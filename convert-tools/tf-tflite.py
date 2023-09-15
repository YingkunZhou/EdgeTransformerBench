"""
Reference doc:
    - https://github.com/sithu31296/PyTorch-ONNX-TFLite
    - https://stackoverflow.com/questions/53182177/how-do-you-convert-a-onnx-to-tflite#67357966
    - https://colab.research.google.com/drive/1MwFVErmqU9Z6cTDWLoTvLgrAEBRZUEsA#revisionId=0BwKss6yztf4KSmx5MlFXQnRnS0Z0ZXRNendXSk4xR0lnWXhnPQ
"""

import argparse
import tensorflow as tf

def get_args_parser():
    parser = argparse.ArgumentParser(
        'EdgeTransformerPerf model format conversion script', add_help=False)
    parser.add_argument('--only-convert', default='', type=str, help='only test a certain model series')
    return parser

if __name__ == '__main__':
    parser=get_args_parser()
    args = parser.parse_args()

    # Convert the model
    for name in [
        'efficientformerv2_s0',
        'efficientformerv2_s1',
        'efficientformerv2_s2',
        'SwiftFormer_XS',
        'SwiftFormer_S' ,
        'SwiftFormer_L1',
        # 'EMO_1M', # cannot convert, even if converted, still cannnot run
        # 'EMO_2M', # cannot convert, even if converted, still cannnot run
        # 'EMO_5M', # cannot convert, even if converted, still cannnot run
        # 'EMO_6M', # cannot convert, even if converted, still cannnot run
        'edgenext_xx_small',
        'edgenext_x_small' ,
        'edgenext_small'   ,
        'mobilevitv2_050',
        'mobilevitv2_075',
        'mobilevitv2_100',
        'mobilevitv2_125',
        'mobilevitv2_150',
        'mobilevitv2_175',
        'mobilevitv2_200',
        'mobilevit_xx_small',
        'mobilevit_x_small' ,
        'mobilevit_small'   ,
        'LeViT_128S',
        'LeViT_128' ,
        'LeViT_192' ,
        'LeViT_256' ,
        'resnet50',
        'mobilenetv3_large_100',
        'tf_efficientnetv2_b0' ,
        'tf_efficientnetv2_b1' ,
        'tf_efficientnetv2_b2' ,
        'tf_efficientnetv2_b3' ,
    ]:
        if args.only_convert and args.only_convert not in name:
            continue

        converter = tf.lite.TFLiteConverter.from_saved_model(".tflite/"+name+".pb") # path to the SavedModel directory
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        # tell converter which type of optimization techniques to use
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # to view the best option for optimization read documentation of tflite about optimization
        # go to this link https://www.tensorflow.org/lite/guide/get_started#4_optimize_your_model_optional

        tf_lite_model = converter.convert()
        # Save the model.
        open(".tflite/"+name+'.tflite', 'wb').write(tf_lite_model)