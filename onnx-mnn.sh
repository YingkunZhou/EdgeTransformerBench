onnx_mnn()
{
    MODEL=$1
    ./MNNConvert -f ONNX --modelFile ../onnx/$MODEL.onnx --MNNModel $MODEL.mnn --bizCode MNN
}

# Vulkan (index: 985,  score: 9.117188), (index: 309,  score: 4.152344), (index: 644,  score: 4.023438),
onnx_mnn efficientformerv2_s0
onnx_mnn efficientformerv2_s1
onnx_mnn efficientformerv2_s2

# Vulkan (index: 985,  score: 20.593750), (index: 108,  score: 9.101562), (index: 308,  score: 8.921875),
onnx_mnn SwiftFormer_XS
onnx_mnn SwiftFormer_S
onnx_mnn SwiftFormer_L1

onnx_mnn EMO_1M
onnx_mnn EMO_2M
onnx_mnn EMO_5M
onnx_mnn EMO_6M

# Vulkan (index: 487,  score: 5.949219), (index: 605,  score: 5.699219), (index: 681,  score: 5.535156),
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

# Vulkan (index: 378,  score: 7.003906), (index: 388,  score: 6.550781), (index: 387,  score: 6.394531),
onnx_mnn mobilevit_xx_small
onnx_mnn mobilevit_x_small
onnx_mnn mobilevit_small

# Vulkan (index: 11,  score: 17200.000000), (index: 28,  score: 15192.000000), (index: 65,  score: 14432.000000),
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
