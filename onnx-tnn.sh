onnx_tnn()
{
    MODEL=$1
    python ~/work/TNN/tools/convert2tnn/converter.py onnx2tnn -optimize -v=v3.0 onnx/$MODEL.onnx -o tnn
}

## for OpenCL
#E/tnn: virtual Status tnn::OpenCLSoftmaxLayerAcc::Init(Context *, LayerParam *, LayerResource *, const std::vector<Blob *> &, const std::vector<Blob *> &) [File source/tnn/device/opencl/acc/opencl_softmax_layer_acc.cc][Line 49] not support axis = -1 in softmax yet!
#E/tnn: virtual Status tnn::DefaultNetwork::InitLayers(NetStructure *, NetResource *) [File source/tnn/core/default_ne
onnx_tnn efficientformerv2_s0
onnx_tnn efficientformerv2_s1
onnx_tnn efficientformerv2_s2

#E/tnn: RawBuffer2ArmBlob [File source/tnn/device/arm/acc/arm_layer_acc.cc][Line 135] RawBuffer2ArmBlob:: unsupported buffer and blob data type: 3, 0
#E/tnn: InitLayers [File source/tnn/core/default_network.cc][Line 321] Error Init layer /network.0/network.0.2/attn/Expand (err: 4096 or 0x1000)
onnx_tnn SwiftFormer_XS
onnx_tnn SwiftFormer_S
onnx_tnn SwiftFormer_L1

#D/tnn: get_node_attr_ai [File tools/onnx2tnn/src/core/onnx_utiliE/tnn: TNNWriteProto [File tools/onnx2tnn/src/core/onnx2tnn.cc][Line 326] error::op convert failed onnx:Mod
#Segmentation fault (core dumped)
# Converter ONNX to TNN model failed!
#onnx_tnn EMO_1M
#onnx_tnn EMO_2M
#onnx_tnn EMO_5M
#onnx_tnn EMO_6M

#E/tnn: RawBuffer2ArmBlob [File source/tnn/device/arm/acc/arm_layer_acc.cc][Line 135] RawBuffer2ArmBlob:: unsupported buffer and blob data type: 3, 0
#E/tnn: InitLayers [File source/tnn/core/default_network.cc][Line 321] Error Init layer /stages.1/stages.1.1/xca/Expand (err: 4096 or 0x1000)
onnx_tnn edgenext_xx_small
onnx_tnn edgenext_x_small
onnx_tnn edgenext_small

#E/tnn: RawBuffer2ArmBlob [File source/tnn/device/arm/acc/arm_layer_acc.cc][Line 135] RawBuffer2ArmBlob:: unsupported buffer and blob data type: 3, 0
#E/tnn: InitLayers [File source/tnn/core/default_network.cc][Line 321] Error Init layer /layer_3/layer_3.1/global_rep/global_rep.0/pre_norm_attn/pre_norm_attn.1/Expand (err: 4096 or 0x1000)
onnx_tnn mobilevitv2_050
onnx_tnn mobilevitv2_075
onnx_tnn mobilevitv2_100
onnx_tnn mobilevitv2_125
onnx_tnn mobilevitv2_150
onnx_tnn mobilevitv2_175
onnx_tnn mobilevitv2_200

#E source/tnn/optimizer/graph_matcher/ir.cc:584 the graph is not connected.
#E/tnn: Optimize [File source/tnn/optimizer/net_optimizer_convert_matmul_to_conv.cc][Line 77] code: 0x6000 msg: source/tnn/optimizer/graph_matcher/ir.cc:584 the graph is not connected.E/tnn: StrideSlice [File source/tnn/utils/dims_function_utils.cc][Line 164] StrideSliceV2Layer param of axes, ends, strides size is invalid
#E/tnn: StrideSlice [File source/tnn/utils/dims_function_utils.cc][Line 164] StrideSliceV2Layer param of axes, ends, strides size is invalid
#E/tnn: StrideSlice [File source/tnn/utils/dims_function_utils.cc][Line 164] StrideSliceV2Layer param of axes, ends, strides size is invalid
#E/tnn: Forward [File source/tnn/core/default_network.cc][Line 603] Forward error code: 0x1000 msg: StrideSliceV2Layer param of axes, ends, strides size is invalid, exit
onnx_tnn mobilevit_xx_small
onnx_tnn mobilevit_x_small
onnx_tnn mobilevit_small

## for OpenCL
#E/tnn: virtual Status tnn::OpenCLSplitVLayerAcc::Init(Context *, LayerParam *, LayerResource *, const std::vector<Blob *> &, const std::vector<Blob *> &) [File source/tnn/device/opencl/acc/opencl_splitv_layer_acc.cc][Line 69] axis=3 is not support in SplitV yet!
#E/tnn: virtual Status tnn::DefaultNetwork::InitLayers(NetStructure *, NetResource *) [File source/tnn/core/default_network.cc][Line 321] Error Init layer /blocks/blocks.0/m/Split (err: 40963 or 0xA003)
onnx_tnn LeViT_128S
onnx_tnn LeViT_128
onnx_tnn LeViT_192
onnx_tnn LeViT_256

onnx_tnn resnet50
onnx_tnn mobilenetv3_large_100

#E/tnn: RawBuffer2ArmBlob [File source/tnn/device/arm/acc/arm_layer_acc.cc][Line 135] RawBuffer2ArmBlob:: unsupported buffer and blob data type: 3, 0
#E/tnn: InitLayers [File source/tnn/core/default_network.cc][Line 321] Error Init layer /blocks/blocks.1/blocks.1.0/conv_exp/Pad (err: 4096 or 0x1000)
onnx_tnn tf_efficientnetv2_b0
onnx_tnn tf_efficientnetv2_b1
onnx_tnn tf_efficientnetv2_b2
onnx_tnn tf_efficientnetv2_b3

# https://github.com/Tencent/TNN/issues/1917
# > Found the problem to be expanding a 1D vector to 3D which is not supported by TNN ARM.