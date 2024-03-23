# USAGE: ./convert-tools/onnx-cann.sh
onnx_cann()
{
    SHAPE=$1
    MODEL=$2
    # source $HOME/Ascend/ascend-toolkit/set_env.sh
    # pip install numpy scipy attrs psutil
    atc --mode 0 --framework 5 --input_format NCHW --soc_version Ascend310B4 \
    --input_shape input:$SHAPE --model .onnx/fp32/$MODEL.onnx --output .cann/fp16/$MODEL

}

mkdir -p .cann/fp16

onnx_cann 1,3,224,224 efficientformerv2_s0
onnx_cann 1,3,224,224 efficientformerv2_s1
onnx_cann 1,3,224,224 efficientformerv2_s2

onnx_cann 1,3,224,224 SwiftFormer_XS
onnx_cann 1,3,224,224 SwiftFormer_S
onnx_cann 1,3,224,224 SwiftFormer_L1

onnx_cann 1,3,224,224 EMO_1M
onnx_cann 1,3,224,224 EMO_2M
onnx_cann 1,3,224,224 EMO_5M
onnx_cann 1,3,224,224 EMO_6M

onnx_cann 1,3,256,256 edgenext_xx_small
onnx_cann 1,3,256,256 edgenext_x_small
onnx_cann 1,3,256,256 edgenext_small

onnx_cann 1,3,256,256 mobilevitv2_050
onnx_cann 1,3,256,256 mobilevitv2_075
onnx_cann 1,3,256,256 mobilevitv2_100
onnx_cann 1,3,256,256 mobilevitv2_125
onnx_cann 1,3,256,256 mobilevitv2_150
onnx_cann 1,3,256,256 mobilevitv2_175
onnx_cann 1,3,256,256 mobilevitv2_200

onnx_cann 1,3,256,256 mobilevit_xx_small
onnx_cann 1,3,256,256 mobilevit_x_small
onnx_cann 1,3,256,256 mobilevit_small

onnx_cann 1,3,224,224 LeViT_128S
onnx_cann 1,3,224,224 LeViT_128
onnx_cann 1,3,224,224 LeViT_192
onnx_cann 1,3,224,224 LeViT_256

onnx_cann 1,3,224,224 resnet50
onnx_cann 1,3,224,224 mobilenetv3_large_100
onnx_cann 1,3,224,224 tf_efficientnetv2_b0
onnx_cann 1,3,240,240 tf_efficientnetv2_b1
onnx_cann 1,3,260,260 tf_efficientnetv2_b2
onnx_cann 1,3,300,300 tf_efficientnetv2_b3