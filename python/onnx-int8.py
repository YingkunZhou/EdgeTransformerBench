# https://medium.com/@hdpoorna/pytorch-to-quantized-onnx-model-18cf2384ec27
import os
import argparse
import torch
import onnx
from onnxruntime.transformers.optimizer import optimize_model
from onnxruntime import quantization
from main import build_dataset

class QuntizationDataReader(quantization.CalibrationDataReader):
    def __init__(self, torch_ds, batch_size, input_name):

        self.torch_dl = torch.utils.data.DataLoader(torch_ds, batch_size=batch_size, shuffle=False)

        self.input_name = input_name
        self.datasize = 70
        self.enum_data = iter(self.torch_dl)
        self.count = 0

    def to_numpy(self, pt_tensor):
        return pt_tensor.detach().cpu().numpy() if pt_tensor.requires_grad else pt_tensor.cpu().numpy()

    def get_next(self):
        if self.count < self.datasize:
            self.count += 1
            batch = next(self.enum_data, None)
            if batch is not None:
                return {self.input_name: self.to_numpy(batch[0])}
            else:
                return None
        return None

    def rewind(self):
        self.enum_data = iter(self.torch_dl)

def get_args_parser():
    parser = argparse.ArgumentParser(
        'onnx model quantization script', add_help=False)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--format', default='int8', type=str, help='model datatype')
    # Model parameters
    parser.add_argument('--only-convert', default='', type=str, help='only test a certain model series')
    # Dataset parameters
    # cd .onnx; ln -sf ../.ncnn/calibration .; cd ..
    parser.add_argument('--data-path', default='.onnx/calibration', type=str, help='dataset path')

    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser('onnx model quantization script', parents=[get_args_parser()])
    args = parser.parse_args()

    for name, resolution, usi_eval, per_channel in [
        ('efficientformerv2_s0', 224, False, True),
        ('efficientformerv2_s1', 224, False, True),
        ('efficientformerv2_s2', 224, False, True),

        ('SwiftFormer_XS', 224, False, True),
        ('SwiftFormer_S' , 224, False, True),
        ('SwiftFormer_L1', 224, False, True),

        ('EMO_1M', 224, False, False),
        ('EMO_2M', 224, False, False),
        ('EMO_5M', 224, False, False),
        ('EMO_6M', 224, False, False),

        ('edgenext_xx_small', 256, False, False),
        ('edgenext_x_small' , 256, False, False),
        ('edgenext_small'   , 256, True , False),

        ('mobilevitv2_050', 256, False, True),
        ('mobilevitv2_075', 256, False, True),
        ('mobilevitv2_100', 256, False, True),
        ('mobilevitv2_125', 256, False, True),
        ('mobilevitv2_150', 256, False, True),
        ('mobilevitv2_175', 256, False, True),
        ('mobilevitv2_200', 256, False, True),

        ('mobilevit_xx_small', 256, False, False),
        ('mobilevit_x_small' , 256, False, False),
        ('mobilevit_small'   , 256, False, False),

        ('LeViT_128S', 224, False, True),
        ('LeViT_128' , 224, False, True),
        ('LeViT_192' , 224, False, True),
        ('LeViT_256' , 224, False, True),

        ('resnet50', 224, False, True),
        ('mobilenetv3_large_100', 224, False, True),
        ('tf_efficientnetv2_b0' , 224, False, True),
        ('tf_efficientnetv2_b1' , 240, False, True),
        ('tf_efficientnetv2_b2' , 260, False, True),
        ('tf_efficientnetv2_b3' , 300, False, True),
    ]:
        if args.only_convert and args.only_convert not in name:
            continue

        args.usi_eval = usi_eval
        args.model = name
        args.input_size = resolution

        model_fp32_path = '.onnx/fp32/'+args.model+'.onnx'
        model_prep_path = model_fp32_path
        # model_prep_path = '.onnx/'+args.model+'_prep.onnx'
        # quantization.shape_inference.quant_pre_process(model_fp32_path, model_prep_path, skip_symbolic_shape=False)

        if args.format == "fp16":
            # https://github.com/huggingface/diffusers/issues/489#issuecomment-1261577250
            """
            To use ONNX Runtime only and no Python fusion logic, use only_onnxruntime flag and a positive opt_level like
            optimize_model(input, opt_level=1, use_gpu=False, only_onnxruntime=True)
            When opt_level is None, we will choose default optimization level according to model type.
            When opt_level is 0 and only_onnxruntime is False, only python fusion logic is used and onnxruntime is disabled.
            """
            from onnxconverter_common import float16
            model_fp16 = optimize_model(
                model_fp32_path,
                # opt_level=1,
                # only_onnxruntime=True,
            )
            model_fp16.convert_float_to_float16(keep_io_types=True)

            if not os.path.exists(".onnx/fp16"):
                os.makedirs(".onnx/fp16")
            onnx.save(model_fp16.model, ".onnx/fp16/"+args.model+".onnx")

        elif args.format == "int8":
            qdr = QuntizationDataReader(build_dataset(args), batch_size=args.batch_size, input_name='input')

            # If model is targeted to GPU/TRT, symmetric activation and weight are required.
            # If model is targeted to CPU, asymmetric activation and symmetric weight are
            # recommended for balance of performance and accuracy.
            q_static_opts = {"ActivationSymmetric":False, # True for GPU
                             "WeightSymmetric":True}

            if not os.path.exists(".onnx/int8"):
                os.makedirs(".onnx/int8")

            quantization.quantize_static(
                model_input=model_prep_path,
                model_output='.onnx/int8/'+args.model+'.onnx',
                calibration_data_reader=qdr,
                extra_options=q_static_opts,
                per_channel=per_channel, # for better accuracy
                reduce_range=False,
                # https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/calibrate.py#L132
                calibrate_method=quantization.CalibrationMethod.MinMax, # == Entropy?
                # MinMax, Entropy, Percentile, Distribution
            )


        elif args.format == "dynamic":
            if not os.path.exists(".onnx/dynamic"):
                os.makedirs(".onnx/dynamic")
            # https://github.com/microsoft/onnxruntime/issues/15888
            # I have the same problem. I am finding that onnxruntime does not support the ConvInteger layer unfortunately,
            # that means dynamic quantization is not working in onnxruntime if initial model has CNNs inside. Very sad!
            quantization.quantize_dynamic(
                model_input=model_prep_path,
                model_output=".onnx/dynamic/"+args.model+".onnx",
                weight_type=quantization.QuantType.QInt8,
            )