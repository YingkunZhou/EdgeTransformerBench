import numpy as np
import torch
import argparse
import time
from main import load_image, build_dataset, MetricLogger, WARMUP_SEC, TEST_SEC
from timm.utils import accuracy
import openvino as ov
from openvino.runtime import get_version
from openvino.runtime.utils.types import get_dtype

def get_args_parser():
    parser = argparse.ArgumentParser(
        'EdgeTransformerPerf tflite_runtime evaluation and benchmark script', add_help=False)
    parser.add_argument('--extern-model', default=None, type=str, help='extern model name;resolution')
    parser.add_argument('--batch-size', default=1, type=int)
    # Dataset parameters
    parser.add_argument('--format', default='fp16', type=str)
    parser.add_argument('--validation', action='store_true', default=False)
    parser.add_argument('--data-path', default='imagenet-div50', type=str, help='dataset path')
    parser.add_argument('--num_workers', default=2, type=int)
    # Benchmark parameters
    parser.add_argument('--only-test', default='', type=str, help='only test a certain model series')
    parser.add_argument('--device', default='CPU', type=str, help='backend device name')
    parser.add_argument('--threads', default=1, type=int, help='number of cpu threads')

    return parser

def benchmark(infer_request, image):
    # warmup
    start = time.perf_counter()
    while time.perf_counter() - start < WARMUP_SEC:
        infer_request.infer(image)

    output = next(iter(infer_request.results.values()))
    val, idx = torch.Tensor(output[0]).topk(3)
    print(list(zip(idx.tolist(), val.tolist())))

    time_list = []
    while sum(time_list) < TEST_SEC:
        start = time.perf_counter()
        infer_request.infer(image)
        time_list.append(time.perf_counter() - start)

    time_max = max(time_list) * 1000
    time_min = min(time_list) * 1000
    time_mean   = np.mean(time_list)   * 1000
    time_median = np.median(time_list) * 1000
    print("min = {:7.2f}ms  max = {:7.2f}ms  mean = {:7.2f}ms, median = {:7.2f}ms".format(time_min, time_max, time_mean, time_median))

def evaluate(data_loader, infer_request, args):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    dataset_scale = 50000//args.len_dataset_val
    print(dataset_scale)
    for images, target in metric_logger.log_every(data_loader, 50, header):
        target = target * dataset_scale + (15 if dataset_scale == 50 else 0)

        infer_request.infer(images)
        output = next(iter(infer_request.results.values()))
        output = torch.Tensor(output)

        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        metric_logger.update(loss=loss.item())

        batch_size = images.shape[0]
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    print(output.mean().item(), output.std().item())

    test_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    print(f"Accuracy on {args.len_dataset_val} test images: {test_stats['acc1']:.1f}%")

if __name__ == '__main__':
    parser = get_args_parser()
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
        # ('mobilevitv2_125', 256, False),
        # ('mobilevitv2_150', 256, False),
        # ('mobilevitv2_175', 256, False),
        # ('mobilevitv2_200', 256, False),

        ('mobilevit_xx_small', 256, False),
        ('mobilevit_x_small' , 256, False),
        ('mobilevit_small'   , 256, False),

        ('LeViT_128S', 224, False),
        ('LeViT_128' , 224, False),
        ('LeViT_192' , 224, False),
        # ('LeViT_256' , 224, False),

        ('resnet50', 224, False),
        ('mobilenetv3_large_100', 224, False),
        ('tf_efficientnetv2_b0' , 224, False),
        ('tf_efficientnetv2_b1' , 240, False),
        ('tf_efficientnetv2_b2' , 260, False),
        # ('tf_efficientnetv2_b3' , 300, False),
    ]:
        if args.only_test and args.only_test not in name and args.only_test != 'ALL' and not args.extern_model:
            continue

        if args.extern_model:
            name = args.extern_model.split(',')[0]
            resolution = int(args.extern_model.split(',')[1])

        args.model = name
        args.input_size = resolution
        args.usi_eval = usi_eval
        print(f"Load openvino model: {name}")
        # Optimize for latency. Most of the devices are configured for latency by default,
        # but there are exceptions like GNA
        config = {}
        config['PERFORMANCE_HINT'] = 'LATENCY'
        if args.format == 'fp32':
            config['INFERENCE_PRECISION_HINT'] = 'f32'
            args.format = 'fp16'
        else:
            config['INFERENCE_PRECISION_HINT'] = 'f16'
        if args.device == 'CPU':
            config['INFERENCE_NUM_THREADS'] = args.threads

        core = ov.Core()
        compiled_model = core.compile_model(f'.xml/{args.format}/{name}.xml', args.device, config)
        ireq = compiled_model.create_infer_request()

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
            evaluate(data_loader_val, ireq, args)
        else:
            benchmark(ireq, load_image(args))

        if args.extern_model: break