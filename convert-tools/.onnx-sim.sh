onnx_sim()
{
    MODEL=$1
    cd .onnx/fp32
    onnxsim $MODEL.onnx $MODEL.sim.onnx
    cd -
}

onnx_sim efficientformerv2_s0
onnx_sim efficientformerv2_s1
onnx_sim efficientformerv2_s2

onnx_sim SwiftFormer_XS
onnx_sim SwiftFormer_S
onnx_sim SwiftFormer_L1

onnx_sim EMO_1M
onnx_sim EMO_2M
onnx_sim EMO_5M
onnx_sim EMO_6M

onnx_sim edgenext_xx_small
onnx_sim edgenext_x_small
onnx_sim edgenext_small

onnx_sim mobilevitv2_050
onnx_sim mobilevitv2_075
onnx_sim mobilevitv2_100
onnx_sim mobilevitv2_125
onnx_sim mobilevitv2_150
onnx_sim mobilevitv2_175
onnx_sim mobilevitv2_200

onnx_sim mobilevit_xx_small
onnx_sim mobilevit_x_small
onnx_sim mobilevit_small

onnx_sim LeViT_128S
onnx_sim LeViT_128
onnx_sim LeViT_192
onnx_sim LeViT_256

onnx_sim resnet50
onnx_sim mobilenetv3_large_100
onnx_sim tf_efficientnetv2_b0
onnx_sim tf_efficientnetv2_b1
onnx_sim tf_efficientnetv2_b2
onnx_sim tf_efficientnetv2_b3