"""
MobileViT
Modified from
https://github.com/apple/ml-cvnets
"""

from typing import Optional, Tuple, Union, Dict, List
import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from timm.models.registry import register_model

class GlobalPool(nn.Module):

    def __init__(
        self,
        keep_dim: Optional[bool] = False,
        **kwargs
    ) -> None:
        super().__init__()
        self.keep_dim = keep_dim

    def _global_pool(self, x: Tensor, dims: List):
        # default is mean
        # same as AdaptiveAvgPool
        return torch.mean(x, dim=dims, keepdim=self.keep_dim)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 4:
            dims = [-2, -1]
        elif x.dim() == 5:
            dims = [-3, -2, -1]
        else:
            raise NotImplementedError("Currently 2D and 3D global pooling supported")
        return self._global_pool(x, dims=dims)


class LinearLayer(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: Optional[bool] = True,
        **kwargs
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features)) if bias else None

        self.in_features = in_features
        self.out_features = out_features

        self.reset_params()

    def reset_params(self):
        if self.weight is not None:
            torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, weight=self.weight, bias=self.bias)


class ConvLayer2d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = 1,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: Optional[int] = 1,
        bias: Optional[bool] = False,
        use_norm: Optional[bool] = True,
        norm_layer = nn.BatchNorm2d,
        use_act: Optional[bool] = True,
        act_layer = nn.SiLU,
        **kwargs
    ) -> None:
        super().__init__()

        block = nn.Sequential()

        block.add_module(
            name="conv",
            module=nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                groups=groups,
                padding=int((kernel_size - 1) / 2),
                bias=bias,
                dilation=dilation,
            ),
        )

        if use_norm:
            block.add_module(
                name="norm",
                module=norm_layer(
                    num_features=out_channels,
                    momentum=0.1,
                ),
            )

        if use_act:
            block.add_module(
                name="act",
                module=act_layer(),
            )

        self.block = block

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class MultiHeadAttention(nn.Module):

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            attn_dropout: float = 0.0,
            bias: bool = True,
            coreml: Optional[bool] = False,
    ) -> None:
        super().__init__()
        output_dim = embed_dim
        if embed_dim % num_heads != 0:
            raise ValueError(
                "Embedding dim must be divisible by number of heads in {}. Got: embed_dim={} and num_heads={}".format(
                    self.__class__.__name__, embed_dim, num_heads
                )
            )

        self.qkv_proj = nn.Linear(
            in_features=embed_dim, out_features=3 * embed_dim, bias=bias
        )

        self.attn_dropout = nn.Dropout(p=attn_dropout)
        self.out_proj = nn.Linear(
            in_features=embed_dim, out_features=output_dim, bias=bias
        )

        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.coreml = coreml
        self.use_pytorch_mha = False # TODO: ???

    def forward_default(self, x_q: Tensor) -> Tensor:
        # [N, S, C]
        b_sz, S_len, in_channels = x_q.shape

        # self-attention
        # [N, S, C] --> [N, S, 3C] --> [N, S, 3, h, c] where C = hc
        qkv = self.qkv_proj(x_q).reshape(b_sz, S_len, 3, self.num_heads, -1)
        # [N, S, 3, h, c] --> [N, h, 3, S, C]
        qkv = qkv.transpose(1, 3).contiguous()

        # [N, h, 3, S, C] --> [N, h, S, C] x 3
        query, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        query = query * self.scaling

        # [N h, T, c] --> [N, h, c, T]
        key = key.transpose(-1, -2)

        # QK^T
        # [N, h, S, c] x [N, h, c, T] --> [N, h, S, T]
        attn = torch.matmul(query, key)

        if True: # TODO
            attn = self.softmax(attn)
            attn = self.attn_dropout(attn)
        else:
            attn_dtype = attn.dtype
            attn_as_float = self.softmax(attn.float())
            attn = attn_as_float.to(attn_dtype)
            attn = self.attn_dropout(attn)

        # weighted sum
        # [N, h, S, T] x [N, h, T, c] --> [N, h, S, c]
        out = torch.matmul(attn, value)

        # [N, h, S, c] --> [N, S, h, c] --> [N, S, C]
        out = out.transpose(1, 2).reshape(b_sz, S_len, -1)
        out = self.out_proj(out)

        return out

    def forward_pytorch(self, x_q: Tensor) -> Tensor:
        """
        coremltools - RuntimeError: PyTorch convert function for op 'scaled_dot_product_attention' not implemented.
        """
        out, _ = F.multi_head_attention_forward(
            query=x_q, key=x_q, value=x_q,
            embed_dim_to_check=self.embed_dim,
            num_heads=self.num_heads,
            in_proj_weight=torch.empty([0]),
            in_proj_bias=self.qkv_proj.bias,
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=self.attn_dropout.p,
            out_proj_weight=self.out_proj.weight,
            out_proj_bias=self.out_proj.bias,
            training=self.training,
            key_padding_mask=None,
            need_weights=False,
            attn_mask=None,
            use_separate_proj_weight=True, # FIXME!
            q_proj_weight=self.qkv_proj.weight[: self.embed_dim, ...],
            k_proj_weight=self.qkv_proj.weight[self.embed_dim: 2 * self.embed_dim, ...],
            v_proj_weight=self.qkv_proj.weight[2 * self.embed_dim:, ...],
        )
        return out

    def forward_tracing(self, x_q: Tensor) -> Tensor:
        # [N, S, C] --> # [N, S, 3C] Here, T=S
        qkv = self.qkv_proj(x_q)
        # # [N, S, 3C] --> # [N, S, C] x 3
        query, key, value = torch.chunk(qkv, chunks=3, dim=-1)

        query = query * self.scaling

        # [N, S, C] --> [N, S, c] x h, where C = c * h
        query = torch.chunk(query, chunks=self.num_heads, dim=-1)

        # [N, T, C] --> [N, T, c] x h, where C = c * h
        value = torch.chunk(value, chunks=self.num_heads, dim=-1)
        # [N, T, C] --> [N, T, c] x h, where C = c * h
        key = torch.chunk(key, chunks=self.num_heads, dim=-1)

        wt_out = []
        for h in range(self.num_heads):
            attn_h = torch.matmul(query[h], key[h].transpose(-1, -2))
            attn_h = self.softmax(attn_h)
            attn_h = self.attn_dropout(attn_h)
            out_h = torch.matmul(attn_h, value[h])
            wt_out.append(out_h)

        wt_out = torch.cat(wt_out, dim=-1)
        wt_out = self.out_proj(wt_out)
        return wt_out

    def forward(self, x_q: Tensor) -> Tensor:
        if self.coreml:
            # For CoreML, we follow batch-first format. Make sure the input is of the form
            # [Batch , Sequence, Hidden_dim]
            return self.forward_tracing(x_q=x_q)
        elif self.use_pytorch_mha:
            # pytorch uses sequence-first format. Make sure that input is of the form [Sequence, Batch, Hidden dim]
            return self.forward_pytorch(x_q=x_q)
        else:
            # our default implementation format follows batch-first format. Make sure the input is of the form
            # [Batch , Sequence, Hidden_dim]
            return self.forward_default(x_q=x_q)


class TransformerEncoder(nn.Module):

    def __init__(
            self,
            embed_dim: int,
            ffn_latent_dim: int,
            num_heads: Optional[int] = 8,
            attn_dropout: Optional[float] = 0.0,
            dropout: Optional[float] = 0.0,
            ffn_dropout: Optional[float] = 0.0,
            transformer_norm_layer = nn.LayerNorm,
            act_layer = nn.SiLU,
            coreml = False,
            **kwargs
    ) -> None:
        super().__init__()

        attn_unit = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            coreml = coreml
        )

        self.pre_norm_mha = nn.Sequential(
            transformer_norm_layer(embed_dim),
            attn_unit,
            nn.Dropout(p=dropout)
        )

        self.pre_norm_ffn = nn.Sequential(
            transformer_norm_layer(embed_dim),
            nn.Linear(in_features=embed_dim, out_features=ffn_latent_dim),
            act_layer(),
            nn.Dropout(p=ffn_dropout),
            nn.Linear(in_features=ffn_latent_dim, out_features=embed_dim),
            nn.Dropout(p=dropout)
        )

    def forward(self, x: Tensor) -> Tensor:
        # multi-head attention
        x = x + self.pre_norm_mha(x)
        # feed forward network
        x = x + self.pre_norm_ffn(x)
        return x


class InvertedResidual(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: Union[int, float],
        act_layer=nn.SiLU,
        dilation: int = 1,
        skip_connection: Optional[bool] = True,
        **kwargs
    ) -> None:
        assert stride in [1, 2]
        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)

        super().__init__()

        block = nn.Sequential()
        if expand_ratio != 1:
            block.add_module(
                name="exp_1x1",
                module=ConvLayer2d(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=1,
                    act_layer=act_layer,
                ),
            )

        block.add_module(
            name="conv_3x3",
            module=ConvLayer2d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                stride=stride,
                kernel_size=3,
                groups=hidden_dim,
                dilation=dilation,
                act_layer=act_layer,
            ),
        )

        block.add_module(
            name="red_1x1",
            module=ConvLayer2d(
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                use_act=False,
            ),
        )

        self.block = block
        self.stride = stride
        self.use_res_connect = (
            self.stride == 1 and in_channels == out_channels and skip_connection
        )

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)


class MobileViTBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        transformer_dim: int,
        ffn_dim: int,
        n_transformer_blocks: int = 2,
        head_dim: int = 32,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        patch_h: int = 8,
        patch_w: int = 8,
        transformer_norm_layer = nn.LayerNorm,
        conv_ksize: Optional[int] = 3,
        act_layer = nn.SiLU,
        dilation: Optional[int] = 1,
        no_fusion: Optional[bool] = False,
        coreml:bool = False,
        **kwargs
    ) -> None:
        super().__init__()

        conv_3x3_in = ConvLayer2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=conv_ksize,
            dilation=dilation,
            act_layer=act_layer,
        )
        conv_1x1_in = ConvLayer2d(
            in_channels=in_channels,
            out_channels=transformer_dim,
            kernel_size=1,
            use_norm=False,
            use_act=False
        )

        self.local_rep = nn.Sequential()
        self.local_rep.add_module(name="conv_3x3", module=conv_3x3_in)
        self.local_rep.add_module(name="conv_1x1", module=conv_1x1_in)

        conv_1x1_out = ConvLayer2d(
            in_channels=transformer_dim,
            out_channels=in_channels,
            kernel_size=1,
            act_layer=act_layer,
        )
        conv_3x3_out = None
        if not no_fusion:
            conv_3x3_out = ConvLayer2d(
                in_channels=2 * in_channels,
                out_channels=in_channels,
                kernel_size=conv_ksize,
                act_layer=act_layer,
            )

        self.conv_proj = conv_1x1_out
        self.fusion = conv_3x3_out

        assert transformer_dim % head_dim == 0
        num_heads = transformer_dim // head_dim

        global_rep = [
            TransformerEncoder(
                embed_dim=transformer_dim,
                ffn_latent_dim=ffn_dim,
                num_heads=num_heads,
                dropout=dropout,
                attn_dropout=attn_dropout,
                ffn_dropout=ffn_dropout,
                transformer_norm_layer=transformer_norm_layer,
                act_layer=act_layer,
                coreml=coreml,
            )
            for _ in range(n_transformer_blocks)
        ]
        global_rep.append(transformer_norm_layer(transformer_dim))
        self.global_rep = nn.Sequential(*global_rep)

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h

    def unfolding(self, feature_map: Tensor) -> Tuple[Tensor, Dict]:
        patch_w, patch_h = self.patch_w, self.patch_h
        patch_area = int(patch_w * patch_h)
        batch_size, in_channels, orig_h, orig_w = feature_map.shape

        new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
        new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)

        interpolate = False
        if new_w != orig_w or new_h != orig_h:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            feature_map = F.interpolate(
                feature_map, size=(new_h, new_w), mode="bilinear", align_corners=False
            )
            interpolate = True

        # number of patches along width and height
        num_patch_w = new_w // patch_w  # n_w
        num_patch_h = new_h // patch_h  # n_h
        num_patches = num_patch_h * num_patch_w  # N

        # [B, C, H, W] --> [B * C * n_h, p_h, n_w, p_w]
        reshaped_fm = feature_map.reshape(
            batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w
        )
        # [B * C * n_h, p_h, n_w, p_w] --> [B * C * n_h, n_w, p_h, p_w]
        transposed_fm = reshaped_fm.transpose(1, 2)
        # [B * C * n_h, n_w, p_h, p_w] --> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
        reshaped_fm = transposed_fm.reshape(
            batch_size, in_channels, num_patches, patch_area
        )
        # [B, C, N, P] --> [B, P, N, C]
        transposed_fm = reshaped_fm.transpose(1, 3)
        # [B, P, N, C] --> [BP, N, C]
        patches = transposed_fm.reshape(batch_size * patch_area, num_patches, -1)

        info_dict = {
            "orig_size": (orig_h, orig_w),
            "batch_size": batch_size,
            "interpolate": interpolate,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h,
        }

        return patches, info_dict

    def folding(self, patches: Tensor, info_dict: Dict) -> Tensor:
        n_dim = patches.dim()
        assert n_dim == 3, "Tensor should be of shape BPxNxC. Got: {}".format(
            patches.shape
        )
        # [BP, N, C] --> [B, P, N, C]
        patches = patches.contiguous().view(
            info_dict["batch_size"], self.patch_area, info_dict["total_patches"], -1
        )

        batch_size, pixels, num_patches, channels = patches.size()
        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]

        # [B, P, N, C] --> [B, C, N, P]
        patches = patches.transpose(1, 3)

        # [B, C, N, P] --> [B*C*n_h, n_w, p_h, p_w]
        feature_map = patches.reshape(
            batch_size * channels * num_patch_h, num_patch_w, self.patch_h, self.patch_w
        )
        # [B*C*n_h, n_w, p_h, p_w] --> [B*C*n_h, p_h, n_w, p_w]
        feature_map = feature_map.transpose(1, 2)
        # [B*C*n_h, p_h, n_w, p_w] --> [B, C, H, W]
        feature_map = feature_map.reshape(
            batch_size, channels, num_patch_h * self.patch_h, num_patch_w * self.patch_w
        )
        if info_dict["interpolate"]:
            feature_map = F.interpolate(
                feature_map,
                size=info_dict["orig_size"],
                mode="bilinear",
                align_corners=False,
            )
        return feature_map

    def forward(self, x: Tensor) -> Tensor:
        res = x

        fm = self.local_rep(x)

        # convert feature map to patches
        patches, info_dict = self.unfolding(fm)

        # learn global representations
        for transformer_layer in self.global_rep:
            patches = transformer_layer(patches)

        # [B x Patch x Patches x C] --> [B x C x Patches x Patch]
        fm = self.folding(patches=patches, info_dict=info_dict)

        fm = self.conv_proj(fm)

        if self.fusion is not None:
            fm = self.fusion(torch.cat((res, fm), dim=1))
        return fm


class MobileViT(nn.Module):

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

        image_channels = 3
        out_channels = 16

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

        exp_channels = min(model_cfg["last_layer_exp_factor"] * out_channels, 960)
        self.conv_1x1_exp = ConvLayer2d(
            in_channels=out_channels,
            out_channels=exp_channels,
            kernel_size=1,
            act_layer=act_layer,
        )

        self.classifier = nn.Sequential()
        if True:
            self.classifier.add_module(
                name="global_pool",
                module=nn.AdaptiveAvgPool2d(1)
            )
            self.classifier.add_module(
                name="flatten",
                module=nn.Flatten()
            )
        else:
            self.classifier.add_module(
                name="global_pool", module=GlobalPool(keep_dim=False)
            )
        self.classifier.add_module(
            name="dropout",
            module=nn.Dropout(p=0.1, inplace=True)
        )
        if True:
            self.classifier.add_module(
                name="fc",
                module=nn.Linear(in_features=exp_channels, out_features=num_classes)
            )
        else:
            self.classifier.add_module(
                name="fc",
                module=LinearLayer(in_features=exp_channels, out_features=num_classes)
            )

        # weight init
        self.apply(self.init_parameters)

    def _make_layer(
        self, input_channel, cfg: Dict,
        dilate: Optional[bool] = False,
    ) -> Tuple[nn.Sequential, int]:
        block_type = cfg.get("block_type", "mobilevit")
        if block_type.lower() == "mobilevit":
            return self._make_mit_layer(
                input_channel=input_channel, cfg=cfg, dilate=dilate
            )
        else:
            return self._make_mobilenet_layer(
                input_channel=input_channel, cfg=cfg
            )

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
                expand_ratio=cfg.get("mv_expand_ratio", 4)
            )

            block.append(layer)
            input_channel = cfg.get("out_channels")

        transformer_dim = cfg["transformer_channels"]
        ffn_dim = cfg.get("ffn_dim")
        num_heads = cfg.get("num_heads", 4)
        head_dim = transformer_dim // num_heads

        if transformer_dim % head_dim != 0:
            raise ValueError("Transformer input dimension should be divisible by head dimension. "
                             "Got {} and {}.".format(transformer_dim, head_dim))

        block.append(
            MobileViTBlock(
            in_channels=input_channel,
            transformer_dim=transformer_dim,
            ffn_dim=ffn_dim,
            n_transformer_blocks=cfg.get("transformer_blocks", 1),
            patch_h=cfg.get("patch_h", 2),
            patch_w=cfg.get("patch_w", 2),
            dropout=0.05,
            head_dim=head_dim,
            no_fusion=False, # TODO
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
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
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
        x = self.conv_1x1_exp(x)
        x = self.classifier(x)
        return x

@register_model
def mobilevit_xx_small(coreml=False, **kwargs):
    config = get_config("xx_small")
    model = MobileViT(model_cfg=config, coreml=coreml, **kwargs)
    return model

@register_model
def mobilevit_x_small(coreml=False, **kwargs):
    config = get_config("x_small")
    model = MobileViT(model_cfg=config, coreml=coreml, **kwargs)
    return model

@register_model
def mobilevit_small(coreml=False, **kwargs):
    config = get_config("small")
    model = MobileViT(model_cfg=config, coreml=coreml, **kwargs)
    return model

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

def get_config(mode: str = "xxs") -> dict:
    if mode == "xx_small":
        mv2_exp_mult = 2
        config = {
            "layer1": {
                "out_channels": 16,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 1,
                "stride": 1,
                "block_type": "mv2",
            },
            "layer2": {
                "out_channels": 24,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 3,
                "stride": 2,
                "block_type": "mv2",
            },
            "layer3": {  # 28x28
                "out_channels": 48,
                "transformer_channels": 64,
                "ffn_dim": 128,
                "transformer_blocks": 2,
                "patch_h": 2,  # 8,
                "patch_w": 2,  # 8,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "num_heads": 4,
                "block_type": "mobilevit",
            },
            "layer4": {  # 14x14
                "out_channels": 64,
                "transformer_channels": 80,
                "ffn_dim": 160,
                "transformer_blocks": 4,
                "patch_h": 2,  # 4,
                "patch_w": 2,  # 4,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "num_heads": 4,
                "block_type": "mobilevit",
            },
            "layer5": {  # 7x7
                "out_channels": 80,
                "transformer_channels": 96,
                "ffn_dim": 192,
                "transformer_blocks": 3,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "num_heads": 4,
                "block_type": "mobilevit",
            },
            "last_layer_exp_factor": 4,
        }
    elif mode == "x_small":
        mv2_exp_mult = 4
        config = {
            "layer1": {
                "out_channels": 32,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 1,
                "stride": 1,
                "block_type": "mv2",
            },
            "layer2": {
                "out_channels": 48,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 3,
                "stride": 2,
                "block_type": "mv2",
            },
            "layer3": {  # 28x28
                "out_channels": 64,
                "transformer_channels": 96,
                "ffn_dim": 192,
                "transformer_blocks": 2,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "num_heads": 4,
                "block_type": "mobilevit",
            },
            "layer4": {  # 14x14
                "out_channels": 80,
                "transformer_channels": 120,
                "ffn_dim": 240,
                "transformer_blocks": 4,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "num_heads": 4,
                "block_type": "mobilevit",
            },
            "layer5": {  # 7x7
                "out_channels": 96,
                "transformer_channels": 144,
                "ffn_dim": 288,
                "transformer_blocks": 3,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "num_heads": 4,
                "block_type": "mobilevit",
            },
            "last_layer_exp_factor": 4,
        }
    elif mode == "small":
        mv2_exp_mult = 4
        config = {
            "layer1": {
                "out_channels": 32,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 1,
                "stride": 1,
                "block_type": "mv2",
            },
            "layer2": {
                "out_channels": 64,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 3,
                "stride": 2,
                "block_type": "mv2",
            },
            "layer3": {  # 28x28
                "out_channels": 96,
                "transformer_channels": 144,
                "ffn_dim": 288,
                "transformer_blocks": 2,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "num_heads": 4,
                "block_type": "mobilevit",
            },
            "layer4": {  # 14x14
                "out_channels": 128,
                "transformer_channels": 192,
                "ffn_dim": 384,
                "transformer_blocks": 4,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "num_heads": 4,
                "block_type": "mobilevit",
            },
            "layer5": {  # 7x7
                "out_channels": 160,
                "transformer_channels": 240,
                "ffn_dim": 480,
                "transformer_blocks": 3,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "num_heads": 4,
                "block_type": "mobilevit",
            },
            "last_layer_exp_factor": 4,
        }
    else:
        raise NotImplementedError

    return config