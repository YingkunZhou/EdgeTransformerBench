onnx_tf()
{
    MODEL=$1
    onnx-tf convert -i onnx/$MODEL.onnx -o tflite/$MODEL.pb
    python .tf-tflite.py --only-convert=$MODEL
}

onnx_tf efficientformerv2_s0
onnx_tf efficientformerv2_s1
onnx_tf efficientformerv2_s2

# --opset-version 12
onnx_tf SwiftFormer_XS
onnx_tf SwiftFormer_S
onnx_tf SwiftFormer_L1

# onnx_tf/handlers/backend/pad.py", line 73:
# constant_values = tensor_dict[node.inputs[2]] if len(
#onnx_tf EMO_1M
#onnx_tf EMO_2M
#onnx_tf EMO_5M
#onnx_tf EMO_6M

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
onnx_tf tf_efficientnetv2_b0
onnx_tf tf_efficientnetv2_b1
onnx_tf tf_efficientnetv2_b2
onnx_tf tf_efficientnetv2_b3
