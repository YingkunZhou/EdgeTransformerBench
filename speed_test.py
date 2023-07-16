"""
Modified from
https://github.com/facebookresearch/LeViT/blob/main/speed_test.py
"""

import argparse
import os
import torch
from torchvision import transforms
import time
import numpy as np
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models import create_model
from main import get_transform, build_dataset, evaluate

import levit
import levit_c
import efficientformer_v2
import swiftformer
import edgenext
import edgenext_bn_hs
import emo
import mobilevit
import mobilevit_v2

torch.autograd.set_grad_enabled(False)

WARMUP_SEC = 5
TEST_SEC  = 20


def benchmarking_cpu(model, inputs):
    # warmup
    start = time.perf_counter()
    while time.perf_counter() - start < WARMUP_SEC:
        outputs = model(inputs)

    val, idx = outputs.topk(3)
    print(list(zip(idx[0].tolist(), val[0].tolist())))

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


def benchmarking_cuda(model, inputs):
    torch.cuda.empty_cache()
    # warmup
    torch.cuda.synchronize()
    # with torch.cuda.amp.autocast():
    start = time.perf_counter()
    while time.perf_counter() - start < WARMUP_SEC:
        outputs = model(inputs)

    val, idx = outputs.topk(3)
    print(list(zip(idx[0].tolist(), val[0].tolist())))

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


def load_image(args):
    data_transform = get_transform(args)
    image = Image.open('daisy.jpg')
    # [N, C, H, W]
    image = data_transform(image)
    # expand batch dimension
    return torch.unsqueeze(image, dim=0)

def get_args_parser():
    parser = argparse.ArgumentParser(
        'LeViT evaluation and speed test script', add_help=False)
    parser.add_argument('--batch-size', default=1, type=int)
    # Model parameters
    parser.set_defaults(pretrained=True)
    parser.add_argument('--fuse', action='store_true', default=False)
    parser.add_argument('--usi_eval', action='store_true', default=False)
    parser.add_argument('--non-pretrained', action='store_false', dest='pretrained')
    parser.add_argument('--weights', default='weights', type=str, help='weigths path')
    parser.add_argument('--only-test', default='zzz', type=str, help='only test a certain model series')
    # Dataset parameters
    parser.add_argument('--data-path', default='imagenet-div50', type=str, help='dataset path')
    parser.add_argument('--num_workers', default=2, type=int)
    # Benchmark parameters
    parser.set_defaults(cpu=True)
    parser.add_argument('--no-cpu', action='store_false', dest='cpu')
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--use_amp', action='store_true', default=False,
                        help="Use PyTorch's AMP (Automatic Mixed Precision) or not") #TODO: much slower?
    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser('EdgeTransformerPerf evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

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
            if args.only_test not in name:
                continue

            args.usi_eval = False
            args.model = name
            args.input_size = resolution

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
                levit.replace_batchnorm(model)  # TODO: acc val speed & acc

            model.to(device)
            model.eval()
            inputs = torch.randn(args.batch_size, 3, resolution,
                             resolution, device=device)
            trace_model = torch.jit.trace(model, inputs)

            if False:
                # load test image
                inputs = load_image(args).to(device)
                benchmarking(trace_model, inputs)
            else:
                dataset_val = build_dataset(args=args)
                sampler_val = torch.utils.data.SequentialSampler(dataset_val)
                data_loader_val = torch.utils.data.DataLoader(
                    dataset_val,
                    sampler=sampler_val,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    drop_last=False
                )

                test_stats = evaluate(data_loader_val, model, device, args)
                print(f"Accuracy on {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
