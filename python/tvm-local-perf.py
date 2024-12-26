"""
References:
https://tvm.apache.org/docs/reference/api/python/graph_executor.html
https://github.com/fengbycq/tvm/blob/xin/xinetzone/docs/how_to/profile/papi.ipynb
https://tvm.hyper.ai/docs/arch/arch/debugger/
https://tvm.apache.org/docs/how_to/profile/papi.html ?
https://tvm.apache.org/docs/reference/api/python/runtime/profiling.html ?
"""

import os
import argparse
import subprocess
import torch
from timm.models import create_model
from main import MetricLogger, build_dataset, load_image
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

import copy
from timm.utils import accuracy
import sota.efficientformer_v2
import sota.swiftformer
import sota.edgenext
import sota.edgenext_bn_hs
import sota.emo
import sota.mobilevit
import sota.mobilevit_v2
import sota.levit
import sota.levit_c
import tvm
from tvm.autotvm.measure.measure_methods import request_remote
from tvm.contrib import graph_executor
from tvm.driver import tvmc
import tvm_utils
torch.autograd.set_grad_enabled(False)

import tvm
from tvm import relay
from tvm.relay.testing import mlp
from tvm.runtime import profiler_vm
import numpy as np
from tvm.contrib import graph_executor

def get_args_parser():
    parser = argparse.ArgumentParser(
        'EdgeTransformerPerf model format conversion script', add_help=False)
    parser.add_argument('--batch-size', default=1, type=int)
    # parser.add_argument('--opset-version', default=None, type=int)
    # by: ln -sf ../.ncnn/calibration .
    parser.add_argument('--data-path', default='./imagenet-div50', type=str, help='dataset path')
    parser.add_argument('--model', default='resnet50', type=str, help='model name')
    parser.add_argument('--only-test', default='', type=str, help='only perf a certain model series')
    parser.add_argument('--validation', action='store_true', help='run validation')
    parser.add_argument('--tvm_backend', default='cpu', type=str, help='tvm backend')
    parser.add_argument('--tvm_data_precision', default='fp32', type=str, help='tvm data precision')
    parser.add_argument('--tvm_tune_method', default='None', type=str, help='tvm tune method')

    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser('EdgeTransformerPerf evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    for name, resolution, usi_eval in [
        ('efficientformerv2_s0', 224, False),
        ('efficientformerv2_s1', 224, False),
        ('efficientformerv2_s2', 224, False),

        ('SwiftFormer_XS', 224, False),
        ('SwiftFormer_S' , 224, False),
        ('SwiftFormer_L1', 224, False),

        ('EMO_1M', 224, False),
        ('EMO_2M', 224, False),
        ('EMO_5M', 224, False),
        ('EMO_6M', 224, False),

        ('edgenext_xx_small', 256, False),
        ('edgenext_x_small' , 256, False),
        ('edgenext_small'   , 256, True),

        ('mobilevitv2_050', 256, False),
        ('mobilevitv2_075', 256, False),
        ('mobilevitv2_100', 256, False),
        ('mobilevitv2_125', 256, False),
        ('mobilevitv2_150', 256, False),
        ('mobilevitv2_175', 256, False),
        ('mobilevitv2_200', 256, False),

        ('mobilevit_xx_small', 256, False),
        ('mobilevit_x_small' , 256, False),
        ('mobilevit_small'   , 256, False),

        ('LeViT_128S', 224, False),
        ('LeViT_128' , 224, False),
        ('LeViT_192' , 224, False),
        ('LeViT_256' , 224, False),

        ('resnet50', 224, False),
        ('mobilenetv3_large_100', 224, False),
        ('tf_efficientnetv2_b0' , 224, False),
        ('tf_efficientnetv2_b1' , 240, False),
        ('tf_efficientnetv2_b2' , 260, False),
        ('tf_efficientnetv2_b3' , 300, False),
    ]:
        if args.only_test and args.only_test not in name:
            continue

        args.model = name
        args.input_size = resolution
        args.usi_eval = usi_eval

        lib: tvm.runtime.Module = tvm.runtime.load_module(".tvm/efficientformerv2_s0.tar.so")
        dev = tvm.cpu()
        module = graph_executor.GraphModule(lib["default"](dev))
        input_name = 'input'
        test_images = load_image(args).numpy()
        module.set_input(input_name,test_images)
        module.run()
        # print(module.get_num_outputs())
        outputs=module.get_output(0).numpy()
        outputs = torch.from_numpy(outputs)
        val, idx = outputs.topk(3)
        print(list(zip(idx[0].tolist(), val[0].tolist())))
        times = module.benchmark(
            dev,
            number=1,
            # min_repeat_ms=1000,
            repeat=50,
            end_to_end=True)
        print(times)

        from tvm.contrib.debugger import debug_executor
        gr = debug_executor.create(lib["get_graph_json"](), lib, dev)
        gr.set_input(input_name,test_images)
        gr.run()
        report = gr.profile(data=test_images)
        print(report)