"""
Reference doc:
    - https://github.com/sithu31296/PyTorch-ONNX-TFLite
    - https://stackoverflow.com/questions/53182177/how-do-you-convert-a-onnx-to-tflite#67357966
    - https://colab.research.google.com/drive/1MwFVErmqU9Z6cTDWLoTvLgrAEBRZUEsA#revisionId=0BwKss6yztf4KSmx5MlFXQnRnS0Z0ZXRNendXSk4xR0lnWXhnPQ
"""

import argparse
import torch
import tensorflow as tf
import numpy as np
from main import build_dataset

def get_args_parser():
    parser = argparse.ArgumentParser(
        'tflite model format conversion script', add_help=False)
    parser.add_argument('--batch-size', default=1, type=int)
    # Model parameters
    parser.set_defaults(pretrained=True)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--non-pretrained', action='store_false', dest='pretrained')
    parser.add_argument('--weights', default='weights', type=str, help='weigths path')
    parser.add_argument('--format', default='fp32', type=str, help='model datatype')
    parser.add_argument('--only-convert', default='', type=str, help='only test a certain model series')
    # Dataset parameters
    parser.add_argument('--validation', action='store_true', default=False)
    parser.add_argument('--data-path', default='.tflite/calibration', type=str, help='dataset path')

    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser('tflite model format conversion script', parents=[get_args_parser()])
    args = parser.parse_args()

    for name, resolution, usi_eval in [
        ('efficientformerv2_s0', 224, False),
        ('efficientformerv2_s1', 224, False),
        ('efficientformerv2_s2', 224, False),

        ('SwiftFormer_XS', 224, False),
        ('SwiftFormer_S' , 224, False),
        ('SwiftFormer_L1', 224, False),

        # ('EMO_1M', 224, False),  # cannot convert, even if converted, still cannnot run
        # ('EMO_2M', 224, False),  # cannot convert, even if converted, still cannnot run
        # ('EMO_5M', 224, False),  # cannot convert, even if converted, still cannnot run
        # ('EMO_6M', 224, False),  # cannot convert, even if converted, still cannnot run

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

        converter = tf.lite.TFLiteConverter.from_saved_model(".tflite/"+name+".pb") # path to the SavedModel directory
        if args.format == "int16":
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS,
                tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
            ]
        else:
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
        # tell converter which type of optimization techniques to use

        dataset_val = build_dataset(args)
        calibration_dataset = [torch.unsqueeze(i[0], dim=0) for i in dataset_val]
        #calibration_dataset = calibration_dataset[:20]
        def representative_dataset():
            for i in calibration_dataset:
                yield [i.numpy().astype(np.float32)]
        #to view the best option for optimization read documentation of tflite about optimization
        #go to this link https://www.tensorflow.org/lite/guide/get_started#4_optimize_your_model_optional
        #https://www.tensorflow.org/lite/performance/post_training_quantization?hl=zh-cn
        #https://www.tensorflow.org/lite/performance/post_training_float16_quant?hl=zh-cn
        if args.format != "fp32":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if args.format == "int16" or args.format == "int8":
            converter.representative_dataset = representative_dataset
        if args.format == "bf16":
            converter.target_spec.supported_types = [tf.float16]
        if args.format == "fp16":
            converter.target_spec.supported_types = [tf.bfloat16]

        tf_lite_model = converter.convert()
        # Save the model.
        open(".tflite/"+args.format+"/"+name+'.tflite', 'wb').write(tf_lite_model)
