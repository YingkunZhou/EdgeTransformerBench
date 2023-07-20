"""
MobileViTv2
Modified from
https://github.com/apple/ml-cvnets
"""

from typing import Optional, Tuple, Union, Dict, Sequence
import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from timm.models.registry import register_model

from mobilevit import ConvLayer2d, LinearLayer, InvertedResidual, GlobalPool

class AdaptiveAvgPool2d(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

    def forward(self, x: Tensor) -> Tensor:
        x = self.global_pool(x)
        x = self.flatten(x)
        return x


class LinearSelfAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        attn_dropout: Optional[float] = 0.0,
        bias: Optional[bool] = True,
        **kwargs
    ) -> None:
        super().__init__()

        self.qkv_proj = ConvLayer2d(
            in_channels=embed_dim,
            out_channels=1 + (2 * embed_dim),
            bias=bias,
            kernel_size=1,
            use_norm=False,
            use_act=False,
        )

        self.attn_dropout = nn.Dropout(p=attn_dropout)
        self.out_proj = ConvLayer2d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            bias=bias,
            kernel_size=1,
            use_norm=False,
            use_act=False,
        )
        self.embed_dim = embed_dim

    def forward(self, x: Tensor) -> Tensor:
        # [B, C, P, N] --> [B, h + 2d, P, N]
        qkv = self.qkv_proj(x)

        # Project x into query, key and value
        # Query --> [B, 1, P, N]
        # value, key --> [B, d, P, N]
        query, key, value = torch.split(
            qkv, split_size_or_sections=[1, self.embed_dim, self.embed_dim], dim=1
        )

        # apply softmax along N dimension
        context_scores = F.softmax(query, dim=-1)
        # Uncomment below line to visualize context scores
        # self.visualize_context_scores(context_scores=context_scores)
        context_scores = self.attn_dropout(context_scores)

        # Compute context vector
        # [B, d, P, N] x [B, 1, P, N] -> [B, d, P, N]
        context_vector = key * context_scores
        # [B, d, P, N] --> [B, d, P, 1]
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

        # combine context vector with values
        # [B, d, P, N] * [B, d, P, 1] --> [B, d, P, N]
        out = F.relu(value) * context_vector.expand_as(value)
        out = self.out_proj(out)
        return out

class LinearAttnFFN(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        ffn_latent_dim: int,
        attn_dropout: Optional[float] = 0.0,
        dropout: Optional[float] = 0.1,
        ffn_dropout: Optional[float] = 0.0,
        norm_layer = nn.GroupNorm,
        **kwargs
    ) -> None:
        super().__init__()

        attn_unit = LinearSelfAttention(
            embed_dim=embed_dim, attn_dropout=attn_dropout,
        )

        self.pre_norm_attn = nn.Sequential(
            norm_layer(num_channels=embed_dim, num_groups=1),
            attn_unit,
            nn.Dropout(p=dropout),
        )

        self.pre_norm_ffn = nn.Sequential(
            norm_layer(num_channels=embed_dim, num_groups=1),
            ConvLayer2d(
                in_channels=embed_dim,
                out_channels=ffn_latent_dim,
                kernel_size=1,
                bias=True,
                use_norm=False,
            ),
            nn.Dropout(p=ffn_dropout),
            ConvLayer2d(
                in_channels=ffn_latent_dim,
                out_channels=embed_dim,
                kernel_size=1,
                bias=True,
                use_norm=False,
                use_act=False,
            ),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        # self-attention
        x = x + self.pre_norm_attn(x)
        # Feed forward network
        x = x + self.pre_norm_ffn(x)
        return x

class MobileViTBlockv2(nn.Module):

    def __init__(
        self,
        in_channels: int,
        attn_unit_dim: int,
        ffn_multiplier: Optional[Union[Sequence[Union[int, float]], int, float]] = 2.0,
        n_attn_blocks: Optional[int] = 2,
        attn_dropout: Optional[float] = 0.0,
        dropout: Optional[float] = 0.0,
        ffn_dropout: Optional[float] = 0.0,
        patch_h: Optional[int] = 8,
        patch_w: Optional[int] = 8,
        conv_ksize: Optional[int] = 3,
        dilation: Optional[int] = 1,
        attn_norm_layer = nn.GroupNorm,
        act_layer=nn.SiLU,
        coreml:bool = False,
        **kwargs
    ) -> None:
        super().__init__()

        cnn_out_dim = attn_unit_dim

        conv_3x3_in = ConvLayer2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=conv_ksize,
            groups=in_channels,
            dilation=dilation,
            act_layer=act_layer,
        )
        conv_1x1_in = ConvLayer2d(
            in_channels=in_channels,
            out_channels=cnn_out_dim,
            kernel_size=1,
            use_norm=False,
            use_act=False,
        )

        self.local_rep = nn.Sequential(conv_3x3_in, conv_1x1_in)

        self.global_rep, attn_unit_dim = self._build_attn_layer(
            d_model=attn_unit_dim,
            ffn_mult=ffn_multiplier,
            n_layers=n_attn_blocks,
            attn_dropout=attn_dropout,
            dropout=dropout,
            ffn_dropout=ffn_dropout,
            attn_norm_layer=attn_norm_layer,
        )

        self.conv_proj = ConvLayer2d(
            in_channels=cnn_out_dim,
            out_channels=in_channels,
            kernel_size=1,
            use_act=False,
        )

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h
        self.cnn_out_dim = cnn_out_dim
        self.coreml = coreml

        if self.coreml:
            # we set persistent to false so that these weights are not part of model's state_dict
            self.register_buffer(
                name="unfolding_weights",
                tensor=self._compute_unfolding_weights(),
                persistent=False,
            )

    def _compute_unfolding_weights(self) -> Tensor:
        # [P_h * P_w, P_h * P_w]
        weights = torch.eye(self.patch_area, dtype=torch.float)
        # [P_h * P_w, P_h * P_w] --> [P_h * P_w, 1, P_h, P_w]
        weights = weights.reshape(
            (self.patch_area, 1, self.patch_h, self.patch_w)
        )
        # [P_h * P_w, 1, P_h, P_w] --> [P_h * P_w * C, 1, P_h, P_w]
        weights = weights.repeat(self.cnn_out_dim, 1, 1, 1)
        return weights

    def _build_attn_layer(
        self,
        d_model: int,
        ffn_mult: Union[Sequence, int, float],
        n_layers: int,
        attn_dropout: float,
        dropout: float,
        ffn_dropout: float,
        attn_norm_layer,
        **kwargs
    ) -> Tuple[nn.Module, int]:

        if isinstance(ffn_mult, Sequence) and len(ffn_mult) == 1:
            ffn_dims = [ffn_mult[0] * d_model] * n_layers
        elif isinstance(ffn_mult, (int, float)):
            ffn_dims = [ffn_mult * d_model] * n_layers
        else:
            raise NotImplementedError

        # ensure that dims are multiple of 16
        ffn_dims = [int((d // 16) * 16) for d in ffn_dims]

        global_rep = [
            LinearAttnFFN(
                embed_dim=d_model,
                ffn_latent_dim=ffn_dims[block_idx],
                attn_dropout=attn_dropout,
                dropout=dropout,
                ffn_dropout=ffn_dropout,
                norm_layer=attn_norm_layer,
            )
            for block_idx in range(n_layers)
        ]
        global_rep.append(attn_norm_layer(num_channels=d_model, num_groups=1))

        return nn.Sequential(*global_rep), d_model

    def unfolding_pytorch(self, feature_map: Tensor) -> Tuple[Tensor, Tuple[int, int]]:

        batch_size, in_channels, img_h, img_w = feature_map.shape

        # [B, C, H, W] --> [B, C, P, N]
        patches = F.unfold(
            feature_map,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )
        patches = patches.reshape(
            batch_size, in_channels, self.patch_area, -1
        )

        return patches, (img_h, img_w)

    def folding_pytorch(self, patches: Tensor, output_size: Tuple[int, int]) -> Tensor:
        batch_size, in_dim, patch_size, n_patches = patches.shape

        # [B, C, P, N]
        patches = patches.reshape(batch_size, in_dim * patch_size, n_patches)

        feature_map = F.fold(
            patches,
            output_size=output_size,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )

        return feature_map

    def unfolding_coreml(self, feature_map: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        # im2col is not implemented in Coreml, so here we hack its implementation using conv2d
        # we compute the weights

        # [B, C, H, W] --> [B, C, P, N]
        batch_size, in_channels, img_h, img_w = feature_map.shape
        #
        patches = F.conv2d(
            feature_map,
            self.unfolding_weights,
            bias=None,
            stride=(self.patch_h, self.patch_w),
            padding=0,
            dilation=1,
            groups=in_channels,
        )
        patches = patches.reshape(
            batch_size, in_channels, self.patch_area, -1
        )
        return patches, (img_h, img_w)

    def folding_coreml(self, patches: Tensor, output_size: Tuple[int, int]) -> Tensor:
        # col2im is not supported on coreml, so tracing fails
        # We hack folding function via pixel_shuffle to enable coreml tracing
        # col2im is also not supported on onnx:
        # Exporting the operator 'aten::col2im' to ONNX opset version 14 is not supported.
        # Support for this operator was added in version 18, try exporting with this version. ?!
        # https://github.com/onnx/onnx/pull/3948
        batch_size, in_dim, patch_size, n_patches = patches.shape

        n_patches_h = output_size[0] // self.patch_h
        n_patches_w = output_size[1] // self.patch_w

        feature_map = patches.reshape(
            batch_size, in_dim * self.patch_area, n_patches_h, n_patches_w
        )
        assert (
            self.patch_h == self.patch_w
        ), "For Coreml, we need patch_h and patch_w are the same"
        feature_map = F.pixel_shuffle(feature_map, upscale_factor=self.patch_h)
        return feature_map

    def resize_input_if_needed(self, x):
        batch_size, in_channels, orig_h, orig_w = x.shape
        if orig_h % self.patch_h != 0 or orig_w % self.patch_w != 0:
            new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
            new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)
            x = F.interpolate(
                x, size=(new_h, new_w), mode="bilinear", align_corners=True
            )
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.resize_input_if_needed(x)

        fm = self.local_rep(x)

        # convert feature map to patches
        if self.coreml:
            patches, output_size = self.unfolding_coreml(fm)
        else:
            patches, output_size = self.unfolding_pytorch(fm)

        # learn global representations on all patches
        patches = self.global_rep(patches)

        # [B x Patch x Patches x C] --> [B x C x Patches x Patch]
        if self.coreml:
            fm = self.folding_coreml(patches=patches, output_size=output_size)
        else:
            fm = self.folding_pytorch(patches=patches, output_size=output_size)
        fm = self.conv_proj(fm)

        return fm


class MobileViTv2(nn.Module):

    def __init__(
            self,
            model_cfg: Dict,
            num_classes: int = 1000,
            act_layer=nn.SiLU,
            coreml: bool = False,
            **kwargs
    ):

        super().__init__()

        self.act_layer = act_layer
        self.coreml = coreml

        image_channels = model_cfg['layer0']['img_channels']
        out_channels = model_cfg['layer0']['out_channels']

        self.conv_1 = ConvLayer2d(
            in_channels=image_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            act_layer=act_layer,
        )

        self.layer_1, out_channels = self._make_layer(
            input_channel=out_channels, cfg=model_cfg["layer1"])
        self.layer_2, out_channels = self._make_layer(
            input_channel=out_channels, cfg=model_cfg["layer2"])
        self.layer_3, out_channels = self._make_layer(
            input_channel=out_channels, cfg=model_cfg["layer3"])
        self.layer_4, out_channels = self._make_layer(
            input_channel=out_channels, cfg=model_cfg["layer4"])
        self.layer_5, out_channels = self._make_layer(
            input_channel=out_channels, cfg=model_cfg["layer5"])

        if True:
            self.classifier = nn.Sequential(
                AdaptiveAvgPool2d(),
                nn.Linear(in_features=out_channels, out_features=num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                GlobalPool(keep_dim=False),
                LinearLayer(in_features=out_channels, out_features=num_classes),
            )

        # weight init
        self.apply(self.init_parameters)

    def _make_layer(self, input_channel, cfg: Dict) -> Tuple[nn.Sequential, int]:
        block_type = cfg.get("block_type", "mobilevit")
        if block_type.lower() == "mobilevit":
            return self._make_mit_layer(input_channel=input_channel, cfg=cfg)
        else:
            return self._make_mobilenet_layer(input_channel=input_channel, cfg=cfg)

    def _make_mobilenet_layer(
            self, input_channel: int, cfg: Dict
        ) -> Tuple[nn.Sequential, int]:
        output_channels = cfg.get("out_channels")
        num_blocks = cfg.get("num_blocks", 2)
        expand_ratio = cfg.get("expand_ratio", 4)
        block = []

        for i in range(num_blocks):
            stride = cfg.get("stride", 1) if i == 0 else 1

            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=output_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                act_layer=self.act_layer,
            )
            block.append(layer)
            input_channel = output_channels

        return nn.Sequential(*block), input_channel

    def _make_mit_layer(
            self, input_channel, cfg: Dict,
            dilate: Optional[bool] = False,
        ) -> Tuple[nn.Sequential, int]:
        stride = cfg.get("stride", 1)
        block = []

        if stride == 2:
            if dilate:
                self.dilation *= 2
                stride = 1

            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=cfg.get("out_channels"),
                stride=stride,
                act_layer=self.act_layer,
                expand_ratio=cfg.get("mv_expand_ratio", 4),
            )

            block.append(layer)
            input_channel = cfg.get("out_channels")

        block.append(MobileViTBlockv2(
            in_channels=input_channel,
            attn_unit_dim=cfg["attn_unit_dim"],
            ffn_multiplier=cfg.get("ffn_multiplier"),
            n_attn_blocks=cfg.get("attn_blocks", 1),
            patch_h=cfg.get("patch_h", 2),
            patch_w=cfg.get("patch_w", 2),
            conv_ksize=3,
            act_layer=self.act_layer,
            coreml=self.coreml,
        ))

        return nn.Sequential(*block), input_channel

    @staticmethod
    def init_parameters(m):
        if isinstance(m, nn.Conv2d):
            if m.weight is not None:
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Linear,)):
            if m.weight is not None:
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        else:
            pass

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_1(x)
        x = self.layer_1(x)
        x = self.layer_2(x)

        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.classifier(x)
        return x

@register_model
def mobilevitv2_050(coreml=True, **kwargs):
    config = get_config("0.50")
    m = MobileViTv2(model_cfg=config, coreml=coreml, **kwargs)
    return m

@register_model
def mobilevitv2_075(coreml=True, **kwargs):
    config = get_config("0.75")
    m = MobileViTv2(model_cfg=config, coreml=coreml, **kwargs)
    return m

@register_model
def mobilevitv2_100(coreml=True, **kwargs):
    config = get_config("1.00")
    m = MobileViTv2(model_cfg=config, coreml=coreml, **kwargs)
    return m

@register_model
def mobilevitv2_125(coreml=True, **kwargs):
    config = get_config("1.25")
    m = MobileViTv2(model_cfg=config, coreml=coreml, **kwargs)
    return m

@register_model
def mobilevitv2_150(coreml=True, **kwargs):
    config = get_config("1.50")
    m = MobileViTv2(model_cfg=config, coreml=coreml, **kwargs)
    return m

@register_model
def mobilevitv2_175(coreml=True, **kwargs):
    config = get_config("1.75")
    m = MobileViTv2(model_cfg=config, coreml=coreml, **kwargs)
    return m

@register_model
def mobilevitv2_200(coreml=True, **kwargs):
    config = get_config("2.00")
    m = MobileViTv2(model_cfg=config, coreml=coreml, **kwargs)
    return m

from typing import Union, Optional

def make_divisible(
    v: Union[float, int],
    divisor: Optional[int] = 8,
    min_value: Optional[Union[float, int]] = None,
) -> Union[float, int]:

    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def bound_fn(
    min_val: Union[float, int], max_val: Union[float, int], value: Union[float, int]
) -> Union[float, int]:
    return max(min_val, min(max_val, value))

def get_config(mode: str = "0.5") -> dict:

    width_multiplier = float(mode)

    ffn_multiplier = (
        2  # bound_fn(min_val=2.0, max_val=4.0, value=2.0 * width_multiplier)
    )
    mv2_exp_mult = 2  # max(1.0, min(2.0, 2.0 * width_multiplier))

    layer_0_dim = bound_fn(min_val=16, max_val=64, value=32 * width_multiplier)
    layer_0_dim = int(make_divisible(layer_0_dim, divisor=8, min_value=16))
    config = {
        "layer0": {
            "img_channels": 3,
            "out_channels": layer_0_dim,
        },
        "layer1": {
            "out_channels": int(make_divisible(64 * width_multiplier, divisor=16)),
            "expand_ratio": mv2_exp_mult,
            "num_blocks": 1,
            "stride": 1,
            "block_type": "mv2",
        },
        "layer2": {
            "out_channels": int(make_divisible(128 * width_multiplier, divisor=8)),
            "expand_ratio": mv2_exp_mult,
            "num_blocks": 2,
            "stride": 2,
            "block_type": "mv2",
        },
        "layer3": {  # 28x28
            "out_channels": int(make_divisible(256 * width_multiplier, divisor=8)),
            "attn_unit_dim": int(make_divisible(128 * width_multiplier, divisor=8)),
            "ffn_multiplier": ffn_multiplier,
            "attn_blocks": 2,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": mv2_exp_mult,
            "block_type": "mobilevit",
        },
        "layer4": {  # 14x14
            "out_channels": int(make_divisible(384 * width_multiplier, divisor=8)),
            "attn_unit_dim": int(make_divisible(192 * width_multiplier, divisor=8)),
            "ffn_multiplier": ffn_multiplier,
            "attn_blocks": 4,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": mv2_exp_mult,
            "block_type": "mobilevit",
        },
        "layer5": {  # 7x7
            "out_channels": int(make_divisible(512 * width_multiplier, divisor=8)),
            "attn_unit_dim": int(make_divisible(256 * width_multiplier, divisor=8)),
            "ffn_multiplier": ffn_multiplier,
            "attn_blocks": 3,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": mv2_exp_mult,
            "block_type": "mobilevit",
        },
        "last_layer_exp_factor": 4,
    }

    return config