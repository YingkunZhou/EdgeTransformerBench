# USAGE: bash convert-tools/pt-pnnx.sh
pt_ncnn()
{
    MODEL=$2
    SHAPE=$1
    mkdir -p .ncnn/fp32 && cd .ncnn/fp32
    ln -s ../../.pt/$MODEL.pt .
    ../../.libs/ncnn/tools/pnnx/build/src/pnnx $MODEL.pt inputshape=$SHAPE
    cd -
}

pt_ncnn [1,3,224,224] efficientformerv2_s0
pt_ncnn [1,3,224,224] efficientformerv2_s1
pt_ncnn [1,3,224,224] efficientformerv2_s2

# layer load_model 35 normalize_16 failed
#pt_ncnn [1,3,224,224] SwiftFormer_XS
#pt_ncnn [1,3,224,224] SwiftFormer_S
#pt_ncnn [1,3,224,224] SwiftFormer_L1

# [1]    720797 segmentation fault (core dumped)  ./ncnn_perf --only-test EMO_1M
#pt_ncnn [1,3,224,224] EMO_1M
#pt_ncnn [1,3,224,224] EMO_2M
#pt_ncnn [1,3,224,224] EMO_5M
#pt_ncnn [1,3,224,224] EMO_6M

# layer load_model 74 normalize_30 failed
#pt_ncnn [1,3,256,256] edgenext_xx_small
#pt_ncnn [1,3,256,256] edgenext_x_small
#pt_ncnn [1,3,256,256] edgenext_small

pt_ncnn [1,3,256,256] mobilevitv2_050
pt_ncnn [1,3,256,256] mobilevitv2_075
pt_ncnn [1,3,256,256] mobilevitv2_100
pt_ncnn [1,3,256,256] mobilevitv2_125
pt_ncnn [1,3,256,256] mobilevitv2_150
pt_ncnn [1,3,256,256] mobilevitv2_175
pt_ncnn [1,3,256,256] mobilevitv2_200

# (index: 999,  score: -nan), (index: 998,  score: -nan), (index: 997,  score: -nan),
# Panvk (index: 387,  score: 7.382812), (index: 282,  score: 6.496094), (index: 292,  score: 6.316406),
pt_ncnn [1,3,256,256] mobilevit_xx_small
#pt_ncnn [1,3,256,256] mobilevit_x_small
#pt_ncnn [1,3,256,256] mobilevit_small

# layer torch.flatten not exists or registered
#pt_ncnn [1,3,224,224] LeViT_128S
#pt_ncnn [1,3,224,224] LeViT_128
#pt_ncnn [1,3,224,224] LeViT_192
#pt_ncnn [1,3,224,224] LeViT_256

pt_ncnn [1,3,224,224] resnet50
pt_ncnn [1,3,224,224] mobilenetv3_large_100
pt_ncnn [1,3,224,224] tf_efficientnetv2_b0
pt_ncnn [1,3,240,240] tf_efficientnetv2_b1
pt_ncnn [1,3,260,260] tf_efficientnetv2_b2
pt_ncnn [1,3,300,300] tf_efficientnetv2_b3