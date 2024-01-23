# https://github.com/PaddlePaddle/PaddleSlim/blob/develop/demo/quant/quant_post/quant_post.py
import os
import sys
import logging
import paddle
import argparse
import functools
import numpy as np
import paddle
from paddleslim.quant import quant_post_static
from main import build_dataset

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--batch_size',      type=int,  default=32,                 help="Minibatch size.")
parser.add_argument('--batch_num',       type=int,  default=512,                help="Batch number")
parser.add_argument('--only-convert',    type=str,  default='', help='only test a certain model series')
parser.add_argument('--model_filename',  type=str,  default='model.pdmodel',    help="model file name")
parser.add_argument('--params_filename', type=str,  default='model.pdiparams',  help="params file name")
parser.add_argument('--algo',            type=str,  default='avg',              help= "calibration algorithm")
parser.add_argument('--round_type',      type=str,  default='round',            help="The method of converting the quantized weights.")
parser.add_argument('--hist_percent',    type=float,default= 0.9999,            help="The percentile of algo:hist")
parser.add_argument('--is_full_quantize',type=bool, default= True,              help="Whether is full quantization or not.")
parser.add_argument('--bias_correction', type=bool, default= False,             help="Whether to use bias correction")
parser.add_argument('--onnx_format',     type=bool, default= False,             help="Whether to export the quantized model with format of ONNX.")
parser.add_argument('--data_path',       type=str,  default='.ncnn/calibration',help='dataset path')

def quantize(args):
    image = paddle.static.data(name="x2paddle_input",
                               shape=[None, 3, args.input_size, args.input_size],
                               dtype='float32')
    calib_dataset = build_dataset(args)
    calib_dataset = [i[0].numpy() for i in calib_dataset]
    data_loader = paddle.io.DataLoader(
        calib_dataset,
        feed_list=[image],
        return_list=False,
        batch_size=1)

    quant_post_static(
        executor=paddle.static.Executor(paddle.CPUPlace()),
        model_dir=".pdlite/paddle/" + args.model,
        quantize_model_path=".pdlite/quant/" + args.model,
        data_loader=data_loader,
        model_filename=args.model_filename,
        params_filename=args.params_filename,
        batch_size=args.batch_size,
        batch_nums=args.batch_num,
        algo=args.algo,
        round_type=args.round_type,
        hist_percent=args.hist_percent,
        is_full_quantize=args.is_full_quantize,
        bias_correction=args.bias_correction,
        onnx_format=args.onnx_format)


def main():
    args = parser.parse_args()
    for name, resolution, usi_eval in [
        ('efficientformerv2_s0', 224, False),
        ('efficientformerv2_s1', 224, False),
        ('efficientformerv2_s2', 224, False),

        ('EMO_1M', 224, False),
        ('EMO_2M', 224, False),
        ('EMO_5M', 224, False),
        ('EMO_6M', 224, False),

        ('edgenext_xx_small', 256, False),
        ('edgenext_x_small' , 256, False),
        ('edgenext_small'   , 256, True),

        ('mobilevit_xx_small', 256, False),
        ('mobilevit_x_small' , 256, False),
        ('mobilevit_small'   , 256, False),

        ('LeViT_128S', 224, False),
        ('LeViT_128' , 224, False),
        ('LeViT_192' , 224, False),
        ('LeViT_256' , 224, False),

        ('resnet50', 224, False),
        ('mobilenetv3_large_100', 224, False),
    ]:
        if args.only_convert and args.only_convert not in name:
            continue

        args.usi_eval = usi_eval
        args.model = name
        args.input_size = resolution
        quantize(args)


if __name__ == '__main__':
    paddle.enable_static()
    main()