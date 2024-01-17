download_calibration()
{
    mkdir -p .tflite/calibration && cd .tflite/calibration
    git clone https://github.com/nihui/imagenet-sample-images.git
    rm imagenet-sample-images/.git -rf
    cd -
}

onnx_tf()
{
    # https://github.com/YingkunZhou/EdgeTransformerPerf/wiki/tensorflow-lite#how-to-convert-model
    TYPE=$1
    MODEL=$2
    mkdir -p .tflite/$TYPE
    onnx-tf convert -i .onnx/fp32/$MODEL.onnx -o .tflite/$MODEL.pb
    python python/tf-tflite.py --only-convert=$MODEL --format=$TYPE
}

download_calibration

TYPE=dynamic

onnx_tf $TYPE efficientformerv2_s0
onnx_tf $TYPE efficientformerv2_s1
onnx_tf $TYPE efficientformerv2_s2

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
onnx_tf $TYPE SwiftFormer_XS
onnx_tf $TYPE SwiftFormer_S
onnx_tf $TYPE SwiftFormer_L1

# onnx_tf/handlers/backend/pad.py", line 73:
# constant_values = tensor_dict[node.inputs[2]] if len(
# ERROR: tensorflow/lite/kernels/reshape.cc:92 num_input_elements != num_output_elements (0 != 8)
#onnx_tf EMO_1M
#onnx_tf EMO_2M
#onnx_tf EMO_5M
#onnx_tf EMO_6M

### gpu error
onnx_tf $TYPE edgenext_xx_small
onnx_tf $TYPE edgenext_x_small
onnx_tf $TYPE edgenext_small

onnx_tf $TYPE mobilevitv2_050
onnx_tf $TYPE mobilevitv2_075
onnx_tf $TYPE mobilevitv2_100
onnx_tf $TYPE mobilevitv2_125
onnx_tf $TYPE mobilevitv2_150
onnx_tf $TYPE mobilevitv2_175
onnx_tf $TYPE mobilevitv2_200

### gpu error
# ERROR: TfLiteGpuDelegate Init: STRIDED_SLICE: Output height doesn't match
# INFO: Created 0 GPU delegate kernels.
# ERROR: TfLiteGpuDelegate Prepare: delegate is not initialized
# ERROR: Node number 1982 (TfLiteGpuDelegateV2) failed to prepare.
# ERROR: Restored original execution plan after delegate application failure.
onnx_tf $TYPE mobilevit_xx_small
onnx_tf $TYPE mobilevit_x_small
onnx_tf $TYPE mobilevit_small

# --fuse
onnx_tf $TYPE LeViT_128S
onnx_tf $TYPE LeViT_128
onnx_tf $TYPE LeViT_192
onnx_tf $TYPE LeViT_256

onnx_tf $TYPE resnet50
onnx_tf $TYPE mobilenetv3_large_100
# --opset-version 12
### gpu error
onnx_tf $TYPE tf_efficientnetv2_b0
onnx_tf $TYPE tf_efficientnetv2_b1
onnx_tf $TYPE tf_efficientnetv2_b2
onnx_tf $TYPE tf_efficientnetv2_b3
