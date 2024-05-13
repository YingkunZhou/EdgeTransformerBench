"""
Modified from
https://github.com/facebookresearch/LeViT/blob/main/speed_test.py
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
    parser.add_argument('--tvm_dev', default='vim3', type=str, help='tvm device name')
    parser.add_argument('--tvm_backend', default='cpu', type=str, help='tvm backend')
    parser.add_argument('--tvm_data_precision', default='fp32', type=str, help='tvm data precision')
    parser.add_argument('--tvm_tune_method', default='None', type=str, help='tvm tune method')

    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser('EdgeTransformerPerf evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    remote_device_name=args.tvm_dev
    remote_device=tvm_utils.find_device_by_name(remote_device_name)
    # remote = request_remote(remote_device_name, "127.0.0.1", port=9190)
    # print(remote.cl().exist)
    # print(remote_device_name)
    if(args.tvm_backend=="cpu"):
        target = remote_device.cpu_target
    elif(args.tvm_backend=="opencl"):
        # target = remote_device.opencl_target
        target = tvm.target.Target(target=remote_device.opencl_target,host=remote_device.cpu_target)
    elif(args.tvm_backend=="vulkan"):
        target = remote_device.vulkan_target
    else:
        raise NotImplementedError
    # target=tvm.target.Target(target)

    # BackendIsNotSupposedToImplementIt: Unsqueeze version 13 is not implemented.
    # https://github.com/onnx/onnx-tensorflow/issues/997
    local_lib_dir = os.path.join(".tvm",remote_device_name,"lib")
    if(os.path.exists(local_lib_dir)==False):
        os.makedirs(local_lib_dir)
    remote_lib_dir = "/home/yangwenhao/EdgeTransformerBench/.tvm/"

    print( f"use{args.data_path}")
    remote = request_remote(remote_device_name, "127.0.0.1", port=9190,timeout=0)
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


        local_lib_dir = os.path.join(".tvm",remote_device_name,"lib")
        if(os.path.exists(local_lib_dir)==False):
            os.makedirs(local_lib_dir)
        # local_lib_filename = name+"_"+args.tvm_tune_method+".tar"
        local_lib_filename = "_".join([name,args.tvm_backend ,args.tvm_data_precision,args.tvm_tune_method])+".tar"
        local_lib_path = os.path.join(local_lib_dir,local_lib_filename)
        remote_lib_path = "/tmp/"+local_lib_filename
        # print(remote_lib_path)
        # if(remote.listdir(remote_lib_path)==False):
        #     remote.upload(local_lib_path,remote_lib_path)
        remote.upload(local_lib_path, remote_lib_path)
        print(f"upload lib success, in {remote_lib_path }")
        rlib = remote.load_module(remote_lib_path)
        # create the remote runtime module
        if args.tvm_backend == 'cpu':
            dev = remote.cpu()
        elif args.tvm_backend == 'opencl':
            dev = remote.cl()
        # print(dev)
        # print(dev.device_type)
        # print(dev.kDLVulkan)
        # print(dev.exist)
        # print(dev.device_id)
        module = graph_executor.GraphModule(rlib["default"](dev))
        input_name = 'input'

        print(f"Running benchmark:{args.model} on {args.tvm_dev} with {args.tvm_backend}, {args.tvm_data_precision}, {args.tvm_tune_method} ")
        if args.validation:
            dataset_val = build_dataset(args)
            data_loader_val = torch.utils.data.DataLoader(
                dataset_val,
                batch_size=args.batch_size,
                shuffle=False
            )
            args.len_dataset_val = len(dataset_val)
            criterion = torch.nn.CrossEntropyLoss()
            metric_logger = MetricLogger(delimiter="  ")
            header = 'Test:'

            dataset_scale = 50000//args.len_dataset_val
            for images, target in metric_logger.log_every(data_loader_val, 50, header):
                batch_size = images.shape[0]
                non_blocking = batch_size > 1
                target = target * dataset_scale + (15 if dataset_scale == 50 else 0)

                # images = images.numpy()

                # module = graph_executor.GraphModule(rlib["default"](dev))
                # set input data

                module.set_input(input_name,images.numpy())#todo set input
                module.run()
                # print(module.get_num_outputs())
                output=module.get_output(0).numpy()
                output = torch.from_numpy(output)
                # for i in range(num_outputs):
                #     output_name = "output_{}".format(i)
                #     outputs[output_name] = module.get_output(i).numpy()
                # result = om_net.forward(images)
                # output = torch.from_numpy(result).unsqueeze(0)

                loss = criterion(output, target)

                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                metric_logger.update(loss=loss.item())
                metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
                metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

            # gather the stats from all processes
            metric_logger.synchronize_between_processes()
            # print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
            #     .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
            # print(output.mean().item(), output.std().item())

            test_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
            print(f"Accuracy on {args.len_dataset_val} test images: {test_stats['acc1']:.1f}%")

        else:

            times=[]
            # module.set_input()
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
                repeat=5,
                end_to_end=False)
            print(times)
