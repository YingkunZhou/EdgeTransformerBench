"""
Modified from
https://github.com/facebookresearch/LeViT/blob/main/speed_test.py
"""

import argparse
import torch
from timm.models import create_model

import models.levit
import models.levit_c
import models.efficientformer_v2
import models.swiftformer
import models.edgenext
import models.edgenext_bn_hs
import models.emo
import models.mobilevit
import models.mobilevit_v2

torch.autograd.set_grad_enabled(False)

def get_args_parser():
    parser = argparse.ArgumentParser(
        'EdgeTransformerPerf model format conversion script', add_help=False)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--opset-version', default=None, type=int)
    # Model parameters
    parser.set_defaults(pretrained=True)
    parser.add_argument('--fuse', action='store_true', default=False)
    parser.add_argument('--non-pretrained', action='store_false', dest='pretrained')
    parser.add_argument('--weights', default='weights', type=str, help='weigths path')
    parser.add_argument('--only-convert', default='', type=str, help='only test a certain model series')
    parser.add_argument('--format', default='', type=str, help='conversion format')
    parser.add_argument('--debug', default=None, type=str, help='e,g --debug 32,4')

    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser('EdgeTransformerPerf evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    # BackendIsNotSupposedToImplementIt: Unsqueeze version 13 is not implemented.
    # https://github.com/onnx/onnx-tensorflow/issues/997
    opv = args.opset_version
    for name, resolution, extern, opset_version, weight in [
        ('efficientformerv2_s0', 224, False, None, "eformer_s0_450.pth"),
        ('efficientformerv2_s1', 224, False, None, "eformer_s1_450.pth"),
        ('efficientformerv2_s2', 224, False, None, "eformer_s2_450.pth"),

        ('SwiftFormer_XS', 224, False, opv, "SwiftFormer_XS_ckpt.pth"),
        ('SwiftFormer_S' , 224, False, opv, "SwiftFormer_S_ckpt.pth"),
        ('SwiftFormer_L1', 224, False, opv, "SwiftFormer_L1_ckpt.pth"),

        ('EMO_1M', 224, False, opv, "EMO_1M.pth"),
        ('EMO_2M', 224, False, opv, "EMO_2M.pth"),
        ('EMO_5M', 224, False, opv, "EMO_5M.pth"),
        ('EMO_6M', 224, False, opv, "EMO_6M.pth"),

        ('edgenext_xx_small', 256, False, opv, "edgenext_xx_small.pth"),
        ('edgenext_x_small' , 256, False, opv, "edgenext_x_small.pth"),
        ('edgenext_small'   , 256, False, opv, "edgenext_small_usi.pth"),

        ('mobilevitv2_050', 256, False, None, "mobilevitv2-0.5.pt"),
        ('mobilevitv2_075', 256, False, None, "mobilevitv2-0.75.pt"),
        ('mobilevitv2_100', 256, False, None, "mobilevitv2-1.0.pt"),
        ('mobilevitv2_125', 256, False, None, "mobilevitv2-1.25.pt"),
        ('mobilevitv2_150', 256, False, None, "mobilevitv2-1.5.pt"),
        ('mobilevitv2_175', 256, False, None, "mobilevitv2-1.75.pt"),
        ('mobilevitv2_200', 256, False, None, "mobilevitv2-2.0.pt"),

        ('mobilevit_xx_small', 256, False, opv, "mobilevit_xxs.pt"),
        ('mobilevit_x_small' , 256, False, opv, "mobilevit_xs.pt"),
        ('mobilevit_small'   , 256, False, opv, "mobilevit_s.pt"),

        ('LeViT_128S', 224, False, None, "LeViT-128S.pth"),
        ('LeViT_128' , 224, False, None, "LeViT-128.pth"),
        ('LeViT_192' , 224, False, None, "LeViT-192.pth"),
        ('LeViT_256' , 224, False, None, "LeViT-256.pth"),

        ('resnet50', 224, True, None, ""),
        ('mobilenetv3_large_100', 224, True, None, ""),
        ('tf_efficientnetv2_b0' , 224, True, opv, ""),
        ('tf_efficientnetv2_b1' , 240, True, opv, ""),
        ('tf_efficientnetv2_b2' , 260, True, opv, ""),
        ('tf_efficientnetv2_b3' , 300, True, opv, ""),
    ]:
        if args.only_convert and args.only_convert not in name:
            continue

        print(f"Creating model: {name}")
        model = create_model(
            name,
            pretrained=extern and args.pretrained,
        )
        if not extern and args.pretrained:
            # load model weights
            weights_dict = torch.load(args.weights+'/'+weight, map_location="cpu")
            # print(weights_dict.keys())

            if "state_dict" in weights_dict:
                weights_dict = weights_dict["state_dict"]
            elif "model" in weights_dict:
                weights_dict = weights_dict["model"]

            if "LeViT_c_" in name:
                D = model.state_dict()
                for k in weights_dict.keys():
                    if D[k].shape != weights_dict[k].shape:
                        weights_dict[k] = weights_dict[k][:, :, None, None]

            model.load_state_dict(weights_dict)

        if args.fuse:
            models.levit.replace_batchnorm(model)  # TODO: speedup levit

        # TODO: does onnx export need this?
        model.eval()

        channels = 3
        if args.debug:
            name = 'debug'
            shapes = args.debug.split(',')
            if len(shapes) > 0 and shapes[0] != '':
                channels = int(shapes[0])
                if len(shapes) > 1:
                    resolution = int(shapes[1])
            print(channels, resolution)

        inputs = torch.randn(
            1, #args.batch_size, TODO: here we only support single batch size benchmarking
            channels, resolution, resolution,
        )

        if not args.format or args.format == 'onnx':
            torch.onnx.export(
                model,
                inputs,
                '.onnx/'+name+'.onnx',
                export_params=True,
                input_names=['input'],
                output_names=['output'],
                opset_version=opset_version
            )
        if not args.format or args.format == 'pt':
            trace_model = torch.jit.trace(model, inputs)
            trace_model.save(".pt/"+name+'.pt')

