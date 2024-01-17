# USAGE: ./convert-tools/onnx-mnn.sh
onnx_mnn()
{
    MODEL=$1
    #mkdir -p .mnn/int8; .libs/MNN/build/MNNConvert -f ONNX --modelFile .onnx/fp32/$MODEL.onnx --MNNModel .mnn/int8/$MODEL.mnn --bizCode MNN --weightQuantBits 8 --weightQuantAsymmetric true
    #mkdir -p .mnn/fp16; .libs/MNN/build/MNNConvert -f ONNX --modelFile .onnx/fp32/$MODEL.onnx --MNNModel .mnn/fp16/$MODEL.mnn --bizCode MNN --fp16
    mkdir -p .mnn/fp32; .libs/MNN/build/MNNConvert -f ONNX --modelFile .onnx/fp32/$MODEL.onnx --MNNModel .mnn/fp32/$MODEL.mnn --bizCode MNN
}

onnx_mnn efficientformerv2_s0
onnx_mnn efficientformerv2_s1
onnx_mnn efficientformerv2_s2

onnx_mnn SwiftFormer_XS
onnx_mnn SwiftFormer_S
onnx_mnn SwiftFormer_L1

onnx_mnn EMO_1M
onnx_mnn EMO_2M
onnx_mnn EMO_5M
onnx_mnn EMO_6M

onnx_mnn edgenext_xx_small
onnx_mnn edgenext_x_small
onnx_mnn edgenext_small

# Cuda: [1]    795723 segmentation fault (core dumped)  ./mnn_perf --only-test mobilevitv2 --backend=c
onnx_mnn mobilevitv2_050
onnx_mnn mobilevitv2_075
onnx_mnn mobilevitv2_100
onnx_mnn mobilevitv2_125
onnx_mnn mobilevitv2_150
onnx_mnn mobilevitv2_175
onnx_mnn mobilevitv2_200

onnx_mnn mobilevit_xx_small
onnx_mnn mobilevit_x_small
onnx_mnn mobilevit_small

onnx_mnn LeViT_128S
onnx_mnn LeViT_128
onnx_mnn LeViT_192
onnx_mnn LeViT_256

onnx_mnn resnet50
onnx_mnn mobilenetv3_large_100
onnx_mnn tf_efficientnetv2_b0
onnx_mnn tf_efficientnetv2_b1
onnx_mnn tf_efficientnetv2_b2
onnx_mnn tf_efficientnetv2_b3
