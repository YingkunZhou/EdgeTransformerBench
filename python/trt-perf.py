import os
import torch
import argparse
import time
import numpy as np
import tensorrt as trt
from main import MetricLogger, accuracy, build_dataset, load_image, WARMUP_SEC, TEST_SEC

def get_args_parser():
    parser = argparse.ArgumentParser(
        'tensorrt evaluation and benchmark script', add_help=False)
    parser.add_argument('--batch-size', default=1, type=int)
    # Model parameters
    parser.set_defaults(pretrained=True)
    parser.add_argument('--format', default='fp16', type=str)
    parser.add_argument('--only-test', default='', type=str, help='only test a certain model series')
    parser.add_argument('--trt-dev', default="gpu", type=str, help='gpu, dla')
    # Dataset parameters
    parser.add_argument('--validation', action='store_true', default=False)
    parser.add_argument('--data-path', default='imagenet-div50', type=str, help='dataset path')

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

        if args.only_test and args.only_test != name:
            continue

        args.model = name
        args.input_size = resolution
        args.usi_eval = usi_eval

        input_shape = (args.batch_size, 3, args.input_size, args.input_size)
        output_shape = (args.batch_size, 1000)

        input_buffer = torch.zeros(input_shape, dtype=torch.float32, device=torch.device('cuda'))
        output_buffer = torch.zeros(output_shape, dtype=torch.float32, device=torch.device('cuda'))
        print(f"Creating TensorRT runtime execution context: {name}")

        if 'mobilevitv2' in args.model:
            if not os.path.exists('.onnx/'+args.model+'.onnx'):
                print(args.model + " model doesn't exist!!!")
                continue

            if os.path.exists('.onnx/'+args.trt_dev+'-'+args.format):
                os.system('.onnx/'+args.trt_dev+'-'+args.format+'/map.sh')

            cmd=['LD_LIBRARY_PATH=.libs/onnxruntime/lib', 'bin/onnxruntime-perf',
                 '--backend', 't', '--only-test']
            cmd.append(args.model)
            if args.format == 'int8':
                cmd.append('--use-int8')
            if args.trt_dev == 'gpu':
                cmd.append('--use-gpu')
            if args.validation:
                cmd.append('--validation')
                cmd.append('--data-path')
                cmd.append(args.data_path)
            os.system(' '.join(cmd))
            continue

        if not os.path.exists('.onnx/'+args.trt_dev+'-'+args.format+'/'+args.model+'.engine'):
            print(args.model + " model doesn't exist!!!")
            continue

        print("loading engine: "+args.trt_dev+'-'+args.format+'/'+args.model+'.engine')

        logger = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(logger)
        with open('.onnx/'+args.trt_dev+'-'+args.format+'/'+args.model+'.engine', 'rb') as f:
            engine_bytes = f.read()
            engine = runtime.deserialize_cuda_engine(engine_bytes)

        input_binding_idx = engine.get_binding_index('input')
        output_binding_idx = engine.get_binding_index('output')

        bindings = [None, None]
        bindings[input_binding_idx] = input_buffer.data_ptr()
        bindings[output_binding_idx] = output_buffer.data_ptr()

        context = engine.create_execution_context()
        context.set_binding_shape(
            input_binding_idx,
            input_shape
        )

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
                input_buffer[0:batch_size].copy_(images)
                context.execute_async_v2(
                    bindings,
                    torch.cuda.current_stream().cuda_stream
                )

                torch.cuda.current_stream().synchronize()
                output = output_buffer[0:batch_size]
                target = target.cuda()

                loss = criterion(output, target)

                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                metric_logger.update(loss=loss.item())
                metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
                metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

            # gather the stats from all processes
            metric_logger.synchronize_between_processes()
            print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
                .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
            print(output.mean().item(), output.std().item())

            test_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
            print(f"Accuracy on {args.len_dataset_val} test images: {test_stats['acc1']:.1f}%")
        else:
            # /usr/src/tensorrt/bin/trtexec --batch=1 --loadEngine=xxx.engine --dumpProfile --separateProfileRun
            # /usr/src/tensorrt/bin/trtexec --batch=1 --loadEngine=xxx.engine --avgRuns=500 --duration=20
            images = load_image(args)
            input_buffer[0:args.batch_size].copy_(images)
            # warmup
            start = time.perf_counter()
            while time.perf_counter() - start < WARMUP_SEC:
                context.execute_async_v2(
                    bindings,
                    torch.cuda.current_stream().cuda_stream
                )
                torch.cuda.current_stream().synchronize()
                outputs = output_buffer[0:args.batch_size]

            val, idx = outputs.topk(3)
            print(list(zip(idx[0].tolist(), val[0].tolist())))

            time_list = []
            while sum(time_list) < TEST_SEC:
                start = time.perf_counter()
                context.execute_async_v2(
                    bindings,
                    torch.cuda.current_stream().cuda_stream
                )
                torch.cuda.current_stream().synchronize()
                time_list.append(time.perf_counter() - start)
            time_max = max(time_list) * 1000
            time_min = min(time_list) * 1000
            time_mean   = np.mean(time_list)   * 1000
            time_median = np.median(time_list) * 1000
            print("min = {:7.2f}ms  max = {:7.2f}ms  mean = {:7.2f}ms, median = {:7.2f}ms".format(time_min, time_max, time_mean, time_median))
