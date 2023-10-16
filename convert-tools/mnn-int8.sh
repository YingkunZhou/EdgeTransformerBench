download_calibration()
{
    cd .mnn
    ln -sf ../.ncnn/calibration .
    cd -
}

# json file style
#{
#    "name":"efficientformerv2_s0",
#    "width":224,
#    "height":224,
#    "path":"calibration/imagenet-sample-images/",
#    "used_image_num":32,
#    "feature_quantize_method":"KL",
#    "weight_quantize_method":"MAX_ABS"
#}

mnn_int8()
{
    MODEL=$1
    cd .mnn
    ../.libs/MNN/build/quantized.out fp32/$MODEL.mnn quant/$MODEL.mnn quant/$MODEL.json
    cd -
}

download_calibration
mnn_int8 efficientformerv2_s0
mnn_int8 efficientformerv2_s1
mnn_int8 efficientformerv2_s2

mnn_int8 SwiftFormer_XS
mnn_int8 SwiftFormer_S
mnn_int8 SwiftFormer_L1

mnn_int8 EMO_1M
mnn_int8 EMO_2M
mnn_int8 EMO_5M
mnn_int8 EMO_6M

mnn_int8 edgenext_xx_small
mnn_int8 edgenext_x_small
mnn_int8 edgenext_small

mnn_int8 mobilevitv2_050
mnn_int8 mobilevitv2_075
mnn_int8 mobilevitv2_100
mnn_int8 mobilevitv2_125
mnn_int8 mobilevitv2_150
mnn_int8 mobilevitv2_175
mnn_int8 mobilevitv2_200

mnn_int8 mobilevit_xx_small
mnn_int8 mobilevit_x_small
mnn_int8 mobilevit_small

mnn_int8 LeViT_128S
mnn_int8 LeViT_128
mnn_int8 LeViT_192
mnn_int8 LeViT_256

mnn_int8 resnet50
mnn_int8 mobilenetv3_large_100
mnn_int8 tf_efficientnetv2_b0
mnn_int8 tf_efficientnetv2_b1
mnn_int8 tf_efficientnetv2_b2
mnn_int8 tf_efficientnetv2_b3