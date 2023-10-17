"""
Reference doc:
    - https://github.com/sithu31296/PyTorch-ONNX-TFLite
    - https://stackoverflow.com/questions/53182177/how-do-you-convert-a-onnx-to-tflite#67357966
    - https://colab.research.google.com/drive/1MwFVErmqU9Z6cTDWLoTvLgrAEBRZUEsA#revisionId=0BwKss6yztf4KSmx5MlFXQnRnS0Z0ZXRNendXSk4xR0lnWXhnPQ
"""

import argparse
import tensorflow as tf
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
    parser.add_argument('--only-convert', default='', type=str, help='only test a certain model series')
    # Dataset parameters
    parser.add_argument('--validation', action='store_true', default=False)
    parser.add_argument('--data-path', default='.tflite/calibration', type=str, help='dataset path')

    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser('tflite model format conversion script', parents=[get_args_parser()])
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

        args.usi_eval = False
        args.model = name
        args.input_size = resolution

        converter = tf.lite.TFLiteConverter.from_saved_model(".tflite/"+name+".pb") # path to the SavedModel directory
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        # tell converter which type of optimization techniques to use

        dataset_val = build_dataset(args)
        calibration_dataset = [i[0] for i in dataset_val]
        def representative_dataset():
            for i in calibration_dataset:
                yield [i]
        # https://www.tensorflow.org/lite/performance/post_training_quantization?hl=zh-cn
        # https://www.tensorflow.org/lite/performance/post_training_float16_quant?hl=zh-cn
        # 您还可以在 GPU 上评估 fp16 量化模型。要使用降低的精度值执行所有算术，请确保在您的应用中创建 TfLiteGPUDelegateOptions 结构，并将 precision_loss_allowed 设置为 1：
        converter.representative_dataset = representative_dataset
        converter.optimizations = [tf.lite.Optimize.DEFAULT] # 推断时，权重从 8 位精度转换为浮点，并使用浮点内核进行计算。此转换会完成一次并缓存，以减少延迟。
        # 为了进一步改善延迟，“动态范围”算子会根据激活的范围将其动态量化为 8 位，并使用 8 位权重和激活执行计算。此优化提供的延迟接近全定点推断。但是，输出仍使用浮点进行存储，因此使用动态范围算子的加速小于全定点计算。
        #converter.target_spec.supported_types = [tf.float16]
        #converter.target_spec.supported_types = [tf.bfloat16]
        # to view the best option for optimization read documentation of tflite about optimization
        # go to this link https://www.tensorflow.org/lite/guide/get_started#4_optimize_your_model_optional

        tf_lite_model = converter.convert()
        # Save the model.
        open(".tflite/"+name+'.tflite', 'wb').write(tf_lite_model)