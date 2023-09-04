onnx_pdlite()
{
    TYPE=$1
    MODEL=$2
    x2paddle --framework=onnx --model=onnx/$MODEL.onnx --save_dir=pdlite --to_lite=True --lite_valid_places=$TYPE --lite_model_type=naive_buffer
    mv pdlite/opt.nb pdlite/$MODEL.nb
}

## replace opencl with arm

onnx_pdlite opencl efficientformerv2_s0
onnx_pdlite opencl efficientformerv2_s1
onnx_pdlite opencl efficientformerv2_s2

# [F  9/ 4  0: 3:25. 51 ...e-Lite/lite/kernels/host/cast_compute.cc:164 Run] other has not been implemented transform with dtype3 X, dtype0 Out
onnx_pdlite opencl SwiftFormer_XS
onnx_pdlite opencl SwiftFormer_S
onnx_pdlite opencl SwiftFormer_L1

# Exception: The padding value is wrong!
# Exception: convert failed node:_stage3_1_Pad_output_0, op_type is Pad
#onnx_pdlite opencl EMO_1M
#onnx_pdlite opencl EMO_2M
#onnx_pdlite opencl EMO_5M
#onnx_pdlite opencl EMO_6M

onnx_pdlite opencl edgenext_xx_small
onnx_pdlite opencl edgenext_x_small
onnx_pdlite opencl edgenext_small

# [F  9/ 3  1:42:33.691 ...rk/work/Paddle-Lite/lite/core/op_lite.cc:176 AttachOutput] Check failed: is_dispensable || is_have_output:
# FatalError: `Process abort signal` is detected by the operating system.
#   [TimeInfo: *** Aborted at 1693676553 (unix time) try "date -d @1693676553" if you are using GNU date ***]
#   [SignalInfo: *** SIGABRT (@0x3e800001a20) received by PID 6688 (TID 0x7fd501eaf740) from PID 6688 ***]
#onnx_pdlite opencl mobilevitv2_050
#onnx_pdlite opencl mobilevitv2_075
#onnx_pdlite opencl mobilevitv2_100
#onnx_pdlite opencl mobilevitv2_125
#onnx_pdlite opencl mobilevitv2_150
#onnx_pdlite opencl mobilevitv2_175
#onnx_pdlite opencl mobilevitv2_200

# [F  9/ 4  0:23:55.496 ...p/Paddle-Lite/lite/operators/slice_op.cc:47 InferShapeImpl] Check failed: (param_.axes[i] < in_dims.size()): -1!<3 The index of dimension in axes must be less than the size of input shape.
onnx_pdlite opencl mobilevit_xx_small
onnx_pdlite opencl mobilevit_x_small
onnx_pdlite opencl mobilevit_small

onnx_pdlite opencl LeViT_128S
onnx_pdlite opencl LeViT_128
onnx_pdlite opencl LeViT_192
onnx_pdlite opencl LeViT_256

onnx_pdlite opencl resnet50
onnx_pdlite opencl mobilenetv3_large_100
# Exception: The padding value is wrong!
# Exception: convert failed node:_conv_stem_Pad_output_0, op_type is Pad
#onnx_pdlite opencl tf_efficientnetv2_b0
#onnx_pdlite opencl tf_efficientnetv2_b1
#onnx_pdlite opencl tf_efficientnetv2_b2
#onnx_pdlite opencl tf_efficientnetv2_b3