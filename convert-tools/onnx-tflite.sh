onnx_tf()
{
    # https://github.com/YingkunZhou/EdgeTransformerPerf/wiki/tensorflow-lite#how-to-convert-model
    MODEL=$1
    ### stage 1: onnx -> tensorflow
    if [ ! -d ".tflite/$MODEL.pb" ]
    then
        onnx-tf convert -i .onnx/fp32/$MODEL.onnx -o .tflite/$MODEL.pb
    fi
    ### stage 2: tensorflow -> tf-lite
    python python/tf-tflite.py --only-convert=$MODEL --format=fp32
    python python/tf-tflite.py --only-convert=$MODEL --format=fp16
    python python/tf-tflite.py --only-convert=$MODEL --format=dynamic
    python python/tf-tflite.py --only-convert=$MODEL --format=int8
}

mkdir -p .tflite/fp32 .tflite/fp16 .tflite/int8 .tflite/dynamic

onnx_tf efficientformerv2_s0
onnx_tf efficientformerv2_s1
onnx_tf efficientformerv2_s2

# --opset-version 12
### gpu error
# ERROR: Unrecognized Write selector
# ERROR: Falling back to OpenGL
# ERROR: TfLiteGpuDelegate Init: Batch size mismatch, expected 1 but got 3136
# INFO: Created 0 GPU delegate kernels.
# ERROR: TfLiteGpuDelegate Prepare: delegate is not initialized
# ERROR: Node number 484 (TfLiteGpuDelegateV2) failed to prepare.
# ERROR: Select TensorFlow op(s), included in the given model, is(are) not supported by this interpreter. Make sure you apply/link the Flex delegate before inference. For the Android, it can be resolved by adding "org.tensorflow:tensorflow-lite-select-tf-ops" dependency. See instructions: https://www.tensorflow.org/lite/guide/ops_select
# ERROR: Node number 14 (FlexErf) failed to prepare.
onnx_tf SwiftFormer_XS
onnx_tf SwiftFormer_S
onnx_tf SwiftFormer_L1

# onnx_tf/handlers/backend/pad.py", line 73:
# constant_values = tensor_dict[node.inputs[2]] if len(
# ERROR: tensorflow/lite/kernels/reshape.cc:92 num_input_elements != num_output_elements (0 != 8)
### need emo.patch
onnx_tf EMO_1M
onnx_tf EMO_2M
onnx_tf EMO_5M
onnx_tf EMO_6M

### gpu error
onnx_tf edgenext_xx_small
onnx_tf edgenext_x_small
onnx_tf edgenext_small

onnx_tf mobilevitv2_050
onnx_tf mobilevitv2_075
onnx_tf mobilevitv2_100
onnx_tf mobilevitv2_125
onnx_tf mobilevitv2_150
onnx_tf mobilevitv2_175
onnx_tf mobilevitv2_200

### gpu error
# ERROR: TfLiteGpuDelegate Init: STRIDED_SLICE: Output height doesn't match
# INFO: Created 0 GPU delegate kernels.
# ERROR: TfLiteGpuDelegate Prepare: delegate is not initialized
# ERROR: Node number 1982 (TfLiteGpuDelegateV2) failed to prepare.
# ERROR: Restored original execution plan after delegate application failure.
onnx_tf mobilevit_xx_small
onnx_tf mobilevit_x_small
onnx_tf mobilevit_small

# --fuse
onnx_tf LeViT_128S
onnx_tf LeViT_128
onnx_tf LeViT_192
onnx_tf LeViT_256

onnx_tf resnet50
onnx_tf mobilenetv3_large_100
# --opset-version 12
### gpu error
onnx_tf tf_efficientnetv2_b0
onnx_tf tf_efficientnetv2_b1
onnx_tf tf_efficientnetv2_b2
onnx_tf tf_efficientnetv2_b3
