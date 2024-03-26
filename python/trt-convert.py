# /usr/src/tensorrt/bin/trtexec --fp16 --onnx=.onnx/fp32/xxx.onnx --saveEngine=.onnx/gpu-fp16/xxx.engine
# /usr/src/tensorrt/bin/trtexec --fp16 --onnx=.onnx/fp32/xxx.onnx --saveEngine=.onnx/dla-fp16/xxx.engine --useDLACore=0 --allowGPUFallback
# above can work for fp16, but slower
import os
import argparse
import torch

import tensorrt as trt
from main import build_dataset

def get_args_parser():
    parser = argparse.ArgumentParser(
        'tensorrt quantization script', add_help=False)
    parser.add_argument('--batch-size', default=1, type=int)
    # Model parameters
    parser.add_argument('--format', default='fp16', type=str)
    parser.add_argument('--only-convert', default='', type=str, help='only test a certain model series')
    # Dataset parameters
    # cd .onnx; ln -sf ../.ncnn/calibration .; cd ..
    parser.add_argument('--data-path', default='.onnx/calibration', type=str, help='dataset path')
    # TensorRT device parameters
    parser.add_argument('--trt-dev', default="gpu", type=str, help='gpu, dla')

    return parser

class DatasetCalibrator(trt.IInt8Calibrator):

    def __init__(self, input, dataset, algorithm, cache_file):
        super(DatasetCalibrator, self).__init__()
        self.algorithm = algorithm
        self.dataset = dataset
        self.cache_file = cache_file
        self.buffer = torch.zeros_like(input).contiguous()
        self.count = 0

    def get_batch(self, *args, **kwargs):
        if self.count < 100:
            for buffer_idx in range(self.get_batch_size()):

                # get image from dataset
                dataset_idx = self.count % len(self.dataset)
                image, _ = self.dataset[dataset_idx]
                image = image.to(self.buffer.device)

                # copy image to buffer
                self.buffer[buffer_idx].copy_(image)

                # increment total number of images used for calibration
                self.count += 1

            return [int(self.buffer.data_ptr())]
        else:
            return []  # returning None or [] signals to TensorRT that calibration is finished

    def get_algorithm(self):
        return self.algorithm

    def get_batch_size(self):
        return int(self.buffer.shape[0])

    def read_calibration_cache(self, *args, **kwargs):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache, *args, **kwargs):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
            f.flush()
            os.fsync(f)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('tensorrt quantization script', parents=[get_args_parser()])
    args = argparser.parse_args()
    if not os.path.exists('.onnx/'+args.trt_dev+'-'+args.format):
        os.makedirs('.onnx/'+args.trt_dev+'-'+args.format)

    # create logger
    logger = trt.Logger(trt.Logger.INFO)
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
        if args.only_convert and args.only_convert not in name:
            continue

        args.usi_eval = usi_eval
        args.model = name
        args.input_size = resolution

        # create builder
        builder = trt.Builder(logger)

        # create network, enabling explicit batch
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        # parse the ONNX model to generate the TensorRT network
        parser = trt.OnnxParser(network, logger)

        with open('.onnx/fp32/'+args.model+'.onnx', 'rb') as f:
            parser.parse(f.read())

        # define the optimization configuration
        profile = builder.create_optimization_profile()
        profile.set_shape(
            'input',
            (args.batch_size, 3, args.input_size, args.input_size), # min shape
            (args.batch_size, 3, args.input_size, args.input_size), # optimal shape
            (args.batch_size, 3, args.input_size, args.input_size), # max shape
        )

        # define the builder configuration
        config = builder.create_builder_config()
        config.add_optimization_profile(profile)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2147483648)

        # below we choose most suitable calibration algorithm for the model
        if "resnet50" in args.model:
            algorithm = trt.CalibrationAlgoType.MINMAX_CALIBRATION
        elif "efficientnet" in args.model \
            or "mobilevit"  in args.model \
                or "EMO"    in args.model:
            # use ENTROPY_CALIBRATION will cause segmentation fault
            algorithm = trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2
        else:
            # ENTROPY_CALIBRATION is better than ENTROPY_CALIBRATION_2 for accuracy
            algorithm = trt.CalibrationAlgoType.ENTROPY_CALIBRATION

        if args.trt_dev == 'dla':
            config.default_device_type = trt.DeviceType.DLA
            config.DLA_core = 0
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
            # Network built for DLA requires kENTROPY_CALIBRATION_2 calibrator.
            algorithm = trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2

        if args.format == 'int8':
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = DatasetCalibrator(
                torch.zeros(args.batch_size, 3, args.input_size, args.input_size).cuda(),
                build_dataset(args),
                algorithm,
                '.onnx/'+args.trt_dev+'-int8/calib-cache/'+args.model
            )
        else:
            config.set_flag(trt.BuilderFlag.FP16)


        engine_bytes = builder.build_serialized_network(network, config)
        with open('.onnx/'+args.trt_dev+'-'+args.format+'/'+args.model+'.engine', 'wb') as f:
            f.write(engine_bytes)