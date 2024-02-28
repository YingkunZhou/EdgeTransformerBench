"""
Modified from
https://github.com/facebookresearch/LeViT/blob/main/speed_test.py
"""

import argparse
import os
import torch
import time
import numpy as np
from PIL import Image
from timm.models import create_model
from main import load_image, build_dataset, evaluate
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
import copy

import sota.efficientformer_v2
import sota.swiftformer
import sota.edgenext
import sota.edgenext_bn_hs
import sota.emo
import sota.mobilevit
import sota.mobilevit_v2
import sota.levit
import sota.levit_c

torch.autograd.set_grad_enabled(False)

WARMUP_SEC = 5
TEST_SEC  = 20

def benchmarking_cpu(model, inputs, args):
    # warmup
    start = time.perf_counter()
    while time.perf_counter() - start < WARMUP_SEC:
        outputs = model(inputs)
        if args.test: break

    val, idx = outputs.topk(3)
    print(list(zip(idx[0].tolist(), val[0].tolist())))
    if args.test: return

    time_list = []
    while sum(time_list) < TEST_SEC:
        start = time.perf_counter()
        model(inputs)
        time_list.append(time.perf_counter() - start)
    time_max = max(time_list) * 1000
    time_min = min(time_list) * 1000
    time_mean   = np.mean(time_list)   * 1000
    time_median = np.median(time_list) * 1000
    print("min = {:7.2f}ms  max = {:7.2f}ms  mean = {:7.2f}ms, median = {:7.2f}ms".format(time_min, time_max, time_mean, time_median))


def benchmarking_cuda(model, inputs, args):
    model = model.cuda()
    inputs = inputs.cuda()
    torch.cuda.empty_cache()
    # warmup
    torch.cuda.synchronize()
    # with torch.cuda.amp.autocast():
    start = time.perf_counter()
    while time.perf_counter() - start < WARMUP_SEC:
        outputs = model(inputs)
        if args.test: break

    val, idx = outputs.topk(3)
    print(list(zip(idx[0].tolist(), val[0].tolist())))
    if args.test: return

    time_list = []

    torch.cuda.synchronize()
    # with torch.cuda.amp.autocast():
    while sum(time_list) < TEST_SEC:
        start = time.perf_counter()
        model(inputs)
        torch.cuda.synchronize()
        time_list.append(time.perf_counter() - start)

    time_max = max(time_list) * 1000
    time_min = min(time_list) * 1000
    time_mean   = np.mean(time_list)   * 1000
    time_median = np.median(time_list) * 1000
    print("min = {:7.2f}ms  max = {:7.2f}ms  mean = {:7.2f}ms, median = {:7.2f}ms".format(time_min, time_max, time_mean, time_median))

def get_args_parser():
    parser = argparse.ArgumentParser(
        'EdgeTransformerPerf evaluation and benchmark script', add_help=False)
    parser.add_argument('--batch-size', default=1, type=int)
    # Model parameters
    parser.set_defaults(pretrained=True)
    parser.add_argument('--fuse', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--non-pretrained', action='store_false', dest='pretrained')
    parser.add_argument('--only-test', default='', type=str, help='only test a certain model series')
    # Dataset parameters
    parser.add_argument('--validation', action='store_true', default=False)
    parser.add_argument('--data-path', default='imagenet-div50', type=str, help='dataset path')
    parser.add_argument('--num_workers', default=2, type=int)
    # Benchmark parameters
    parser.set_defaults(cpu=True)
    parser.add_argument('--no-cpu', action='store_false', dest='cpu')
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--use-eager', action='store_true', default=False)
    parser.add_argument('--use-quant', action='store_true', default=False)
    parser.add_argument('--use-script', action='store_true', default=False)
    parser.add_argument('--use-trace', action='store_true', default=False)
    parser.add_argument('--use-inference', action='store_true', default=False)
    parser.add_argument('--use-compile', action='store_true', default=False)
    parser.add_argument('--use-mobile', action='store_true', default=False)
    parser.add_argument('--use-trt', action='store_true', default=False)

    parser.add_argument('--use_amp', action='store_true', default=False,
                        help="Use PyTorch's AMP (Automatic Mixed Precision) or not") #TODO: much slower?
    parser.add_argument('--backend', default='inductor', type=str, help='pytorch compile backend')

    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser('EdgeTransformerPerf evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True # TODO:?
    device_list = []
    if args.cpu: device_list.append('cpu')
    if args.cuda: device_list.append('cuda:0')

    for device in device_list:

        if 'cuda' in device and not torch.cuda.is_available():
            print("no cuda")
            continue

        if device == 'cpu':
            os.system('echo -n "nb processors "; '
                    'cat /proc/cpuinfo | grep ^processor | wc -l; '
                    'cat /proc/cpuinfo | grep ^"model name" | tail -1')
            print('Using 1 cpu thread')
            torch.set_num_threads(1)
            benchmarking = benchmarking_cpu
        else:
            print(torch.cuda.get_device_name(torch.cuda.current_device()))
            benchmarking = benchmarking_cuda

        for name, resolution, extern, weight in [
            ('efficientformerv2_s0', 224, False, "eformer_s0_450.pth"),
            ('efficientformerv2_s1', 224, False, "eformer_s1_450.pth"),
            ('efficientformerv2_s2', 224, False, "eformer_s2_450.pth"),

            ('SwiftFormer_XS', 224, False, "SwiftFormer_XS_ckpt.pth"),
            ('SwiftFormer_S' , 224, False, "SwiftFormer_S_ckpt.pth"),
            ('SwiftFormer_L1', 224, False, "SwiftFormer_L1_ckpt.pth"),

            ('EMO_1M', 224, False, "EMO_1M.pth"),
            ('EMO_2M', 224, False, "EMO_2M.pth"),
            ('EMO_5M', 224, False, "EMO_5M.pth"),
            ('EMO_6M', 224, False, "EMO_6M.pth"),

            ('edgenext_xx_small', 256, False, "edgenext_xx_small.pth"),
            ('edgenext_x_small' , 256, False, "edgenext_x_small.pth"),
            ('edgenext_small'   , 256, False, "edgenext_small_usi.pth"),

            ('mobilevitv2_050', 256, False, "mobilevitv2-0.5.pt"),
            ('mobilevitv2_075', 256, False, "mobilevitv2-0.75.pt"),
            ('mobilevitv2_100', 256, False, "mobilevitv2-1.0.pt"),
            ('mobilevitv2_125', 256, False, "mobilevitv2-1.25.pt"),
            ('mobilevitv2_150', 256, False, "mobilevitv2-1.5.pt"),
            ('mobilevitv2_175', 256, False, "mobilevitv2-1.75.pt"),
            ('mobilevitv2_200', 256, False, "mobilevitv2-2.0.pt"),

            ('mobilevit_xx_small', 256, False, "mobilevit_xxs.pt"),
            ('mobilevit_x_small' , 256, False, "mobilevit_xs.pt"),
            ('mobilevit_small'   , 256, False, "mobilevit_s.pt"),

            ('LeViT_128S', 224, False, "LeViT-128S.pth"),
            ('LeViT_128' , 224, False, "LeViT-128.pth"),
            ('LeViT_192' , 224, False, "LeViT-192.pth"),
            ('LeViT_256' , 224, False, "LeViT-256.pth"),

            ('resnet50', 224, True, ""),
            ('mobilenetv3_large_100', 224, True, ""),
            ('tf_efficientnetv2_b0' , 224, True, ""),
            ('tf_efficientnetv2_b1' , 240, True, ""),
            ('tf_efficientnetv2_b2' , 260, True, ""),
            ('tf_efficientnetv2_b3' , 300, True, ""),
        ]:
            if args.only_test and args.only_test not in name:
                continue

            args.usi_eval = False
            args.model = name
            args.input_size = resolution

            model = create_model(
                name,
                pretrained=extern and args.pretrained,
            )
            if not extern and args.pretrained:
                # load model weights
                if not os.path.exists('weights'):
                    import subprocess
                    if not os.path.exists('EdgeTransformerPerf-weights.tar'):
                        print("============Downloading weights============")
                        print("============you should install gdown first: pip install gdown============")
                        subprocess.run(['gdown', '19irI6H_c1w2OaDOVzPIj2v0Dy30pq-So'])
                    else:
                        print("============Extracting weights============")
                        subprocess.run(['tar', 'xf', 'EdgeTransformerPerf-weights.tar'])
                weights_dict = torch.load('weights/'+weight, map_location="cpu")
                # print(weights_dict.keys())

                if "state_dict" in weights_dict:
                    args.usi_eval = True
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
                sota.levit.replace_batchnorm(model)  # TODO: acc val speed & acc

            model.to(device)
            model.eval()

            inputs = torch.randn(1, 3, resolution, resolution,)

            if args.use_quant:
                # TODO: if use x86 machine, please replace 'qnnpack' with 'x86'!
                torch.backends.quantized.engine = 'qnnpack'
                # we need to deepcopy if we still want to keep model_fp unchanged after quantization since quantization apis change the input model
                float_model = copy.deepcopy(model)
                float_model.eval()
                qconfig_mapping = get_default_qconfig_mapping('qnnpack')
                # a tuple of one or more example inputs are needed to trace the model
                example_inputs = (inputs,)

                prepared_model = prepare_fx(float_model, qconfig_mapping, example_inputs)  # fuse modules and insert observers
                prepared_model.eval()
                calibration_data_loader = build_dataset(args)
                calibration_data = [torch.unsqueeze(i[0], dim=0) for i in calibration_data_loader]
                calibration_data = calibration_data[:2] # for quick try
                with torch.inference_mode():
                    for i in calibration_data:
                        prepared_model(i)

                quantized_model = convert_fx(prepared_model)  # convert the calibrated model to a quantized model
            if args.use_script:
                script_model = torch.jit.script(model)
            if args.use_trace:
                if not os.path.exists(".pt/" + args.model + ".pt"):
                    print(args.model + " model doesn't exist!!!")
                    continue
                trace_model = torch.jit.load(".pt/" + args.model + ".pt")
            if args.use_inference:
                frozen_model = torch.jit.optimize_for_inference(torch.jit.script(model))
            if args.use_compile:
                import torch._dynamo as dynamo
                # dynamo.config.verbose=True
                # dynamo.config.suppress_errors = True
                compile_model = torch.compile(model, backend=args.backend)
            if args.use_mobile:
                if not os.path.exists(".pt/" + args.model + ".c.ptl"):
                    print(args.model + " model doesn't exist!!!")
                    continue
                mobile_model = torch.jit.load(".pt/" + args.model + ".c.ptl")
            if args.use_trt:
                if not os.path.exists(".pt/" + args.model + ".ts"):
                    print(args.model + " model doesn't exist!!!")
                    continue
                import torch_tensorrt
                trt_model = torch.jit.load(".pt/" + args.model + ".ts").cuda()
                benchmarking = benchmarking_cuda

            print(f"Creating model: {name}")
            if args.validation:
                dataset_val = build_dataset(args)
                sampler_val = torch.utils.data.SequentialSampler(dataset_val)
                data_loader_val = torch.utils.data.DataLoader(
                    dataset_val,
                    sampler=sampler_val,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    drop_last=False
                )
                args.len_dataset_val = len(dataset_val)

                if args.use_eager:
                    evaluate(data_loader_val, model, device, args)
                if args.use_script:
                    evaluate(data_loader_val, script_model, device, args)
                if args.use_trace:
                    evaluate(data_loader_val, trace_model, device, args)
                if args.use_inference:
                    evaluate(data_loader_val, frozen_model, device, args)
                if args.use_compile:
                    evaluate(data_loader_val, compile_model, device, args)
                if args.use_mobile:
                    evaluate(data_loader_val, mobile_model, device, args)
                if args.use_trt:
                    evaluate(data_loader_val, trt_model, 'cuda:0', args)
            else:
                # load test image
                inputs = load_image(args=args).to(device)

                if args.use_eager:
                    benchmarking(model, inputs, args)
                if args.use_quant:
                    benchmarking(quantized_model, inputs, args)
                if args.use_script:
                    benchmarking(script_model, inputs, args)
                if args.use_trace:
                    benchmarking(trace_model, inputs, args)
                if args.use_inference:
                    benchmarking(frozen_model, inputs, args)
                if args.use_compile:
                    benchmarking(compile_model, inputs, args)
                if args.use_mobile:
                    benchmarking(mobile_model, inputs, args)
                if args.use_trt:
                    benchmarking(trt_model, inputs, args)