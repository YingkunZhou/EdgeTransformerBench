"""
Reference doc:
    - https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification/cpu/run.py
    - https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/calibrate.py
    - https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/quantize.py
    - https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/notebooks/bert/Bert-GLUE_OnnxRuntime_quantization.ipynb
"""

import numpy
import argparse
import onnxruntime
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static, quantize_dynamic, CalibrationMethod
from main import build_dataset

class DataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder:str, model_name: str):
        self.enum_data = None

        # Use inference session to get input shape.
        session = onnxruntime.InferenceSession(".onnx/prep/"+name+".onnx", None)
        (_, _, height, _) = session.get_inputs()[0].shape

        # Convert image to input data
        args.data_path = calibration_image_folder
        args.model = model_name
        args.input_size = height
        args.usi_eval = model_name == "edgenext_small"
        dataset_val = build_dataset(args)
        dataset_val = [numpy.expand_dims(i[0], axis=0) for i in dataset_val]
        self.nhwc_data_list = numpy.concatenate(
            numpy.expand_dims(dataset_val, axis=0), axis=0
        )
        self.input_name = session.get_inputs()[0].name
        self.datasize = len(self.nhwc_data_list)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                [{self.input_name: nhwc_data} for nhwc_data in self.nhwc_data_list]
            )
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None

def get_args_parser():
    parser = argparse.ArgumentParser(
        'onnx model quantization script', add_help=False)
    # Model parameters
    parser.set_defaults(pretrained=True)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--format', default='fp32', type=str, help='model datatype')
    parser.add_argument('--only-convert', default='', type=str, help='only test a certain model series')
    # Dataset parameters
    parser.add_argument('--data-path', default='.onnx/calibration/', type=str, help='dataset path')
    parser.add_argument(
        "--quant_format",
        default=QuantFormat.QDQ,
        type=QuantFormat.from_string,
        choices=list(QuantFormat),
    )
    parser.add_argument("--per_channel", default=True, type=bool)

    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser('onnx model quantization scrip', parents=[get_args_parser()])
    args = parser.parse_args()

    for name, resolution in [
        ('efficientformerv2_s0', 224),
        ('efficientformerv2_s1', 224),
        ('efficientformerv2_s2', 224),

        ('SwiftFormer_XS', 224),
        ('SwiftFormer_S' , 224),
        ('SwiftFormer_L1', 224),

        ('EMO_1M', 224), # cannot convert, even if converted, still cannnot run
        ('EMO_2M', 224), # cannot convert, even if converted, still cannnot run
        ('EMO_5M', 224), # cannot convert, even if converted, still cannnot run
        ('EMO_6M', 224), # cannot convert, even if converted, still cannnot run

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

        print(name)
        args.model = name
        args.input_size = resolution

        dr = DataReader(
            args.data_path, args.model
        )

        if True:
            # Calibrate and quantize model
            # Turn off model optimization during quantization
            quantize_static(
                ".onnx/prep/"+name+".onnx",
                ".onnx/int8/"+name+".onnx",
                dr,
                quant_format=args.quant_format,
                per_channel=args.per_channel,
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
                calibrate_method=CalibrationMethod.MinMax,
            )
        else:
            # https://github.com/microsoft/onnxruntime/issues/15888
            # I have the same problem. I am finding that onnxruntime does not support the ConvInteger layer unfortunately, that means dynamic quantization is not working in onnxruntime if initial model has CNNs inside. Very sad!
            quantize_dynamic(
                ".onnx/prep/"+name+".onnx",
                ".onnx/dynamic/"+name+".onnx",
                weight_type=QuantType.QInt8,
            )