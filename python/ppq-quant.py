"""
reference code:
    - https://github.com/openppl-public/ppq/blob/master/ppq/samples/quantize_onnx_model.py
"""

import os
import argparse

import numpy as np
import torch
from ppq import BaseGraph, QuantizationSettingFactory, TargetPlatform
from ppq.api import export_ppq_graph, quantize_onnx_model
from main import build_dataset

def get_args_parser():
    parser = argparse.ArgumentParser(
        'ppq quantization script', add_help=False)
    parser.add_argument('--batch-size', default=1, type=int)
    # Model parameters
    parser.set_defaults(pretrained=True)
    parser.add_argument('--fuse', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--non-pretrained', action='store_false', dest='pretrained')
    parser.add_argument('--weights', default='weights', type=str, help='weigths path')
    parser.add_argument('--only-convert', default='', type=str, help='only test a certain model series')
    # Dataset parameters
    parser.add_argument('--validation', action='store_true', default=False)
    parser.add_argument('--data-path', default='.ncnn/calibration', type=str, help='dataset path')
    parser.add_argument('--num_workers', default=2, type=int)

    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ppq quantization script', parents=[get_args_parser()])
    args = parser.parse_args()

    for name, resolution in [
        ('efficientformerv2_s0', 224),
        ('efficientformerv2_s1', 224),
        ('efficientformerv2_s2', 224),

        ('SwiftFormer_XS', 224),
        ('SwiftFormer_S' , 224),
        ('SwiftFormer_L1', 224),

        ('EMO_1M', 224),
        ('EMO_2M', 224),
        ('EMO_5M', 224),
        ('EMO_6M', 224),

        ('edgenext_xx_small', 256),
        ('edgenext_x_small' , 256),
        ('edgenext_small'   , 256),

        ('mobilevitv2_050', 256),
        ('mobilevitv2_075', 256),
        ('mobilevitv2_100', 256),
        ('mobilevitv2_125', 256),
        ('mobilevitv2_150', 256),
        ('mobilevitv2_175', 256),
        ('mobilevitv2_200', 256),

        ('mobilevit_xx_small', 256),
        ('mobilevit_x_small' , 256),
        ('mobilevit_small'   , 256),

        ('LeViT_128S', 224),
        ('LeViT_128' , 224),
        ('LeViT_192' , 224),
        ('LeViT_256' , 224),

        ('resnet50', 224),
        ('mobilenetv3_large_100', 224),
        ('tf_efficientnetv2_b0' , 224),
        ('tf_efficientnetv2_b1' , 240),
        ('tf_efficientnetv2_b2' , 260),
        ('tf_efficientnetv2_b3' , 300),
    ]:
        if args.only_convert and args.only_convert not in name:
            continue

        args.usi_eval = False
        args.model = name
        args.input_size = resolution

        DEVICE = 'cuda'
        PLATFORM = TargetPlatform.NCNN_INT8
        quant_setting = QuantizationSettingFactory.ncnn_setting()

        # run quantization
        dataset_val = build_dataset(args)
        calibration_dataset = [i[0] for i in dataset_val]
        calibration_dataloader = torch.utils.data.DataLoader(
            dataset=calibration_dataset,
            batch_size=1, shuffle=True)

        def collate_fn(batch: torch.Tensor) -> torch.Tensor:
            return batch.to(DEVICE)

        # AssertionError: Calibration steps is too large, ppq can quantize your network within 8-512 calibration steps. More calibration steps will greatly delay ppq's calibration procedure. Reset your calib_steps parameter please.
        calib_steps = max(min(512, len(dataset_val)), 8)   # 8 ~ 512
        quantized = quantize_onnx_model(
            onnx_import_file=".onnx/" + args.model + ".sim.onnx",
            calib_dataloader=calibration_dataloader,
            calib_steps=calib_steps, input_shape=[1, 3, resolution, resolution],
            setting=quant_setting, collate_fn=collate_fn,
            platform=PLATFORM, device=DEVICE, verbose=0
        )

        assert isinstance(quantized, BaseGraph)

        # export quantization param file and model file
        export_ppq_graph(graph=quantized, platform=PLATFORM,
                         graph_save_to=args.model + '.onnx',
                         config_save_to=args.model + '.table')
