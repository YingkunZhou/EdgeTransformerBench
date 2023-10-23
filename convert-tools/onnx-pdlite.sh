onnx_pdlite()
{
    TYPE=$1
    MODEL=$2
    x2paddle --framework=onnx --model=.onnx/$MODEL.onnx --save_dir=.pdlite --to_lite=True --lite_valid_places=$TYPE --lite_model_type=naive_buffer
    mv .pdlite/opt.nb .pdlite/$MODEL.nb
    mv .pdlite/inference_model .pdlite/$MODEL
}

## replace opencl with arm
BACK=arm

onnx_pdlite $BACK efficientformerv2_s0
onnx_pdlite $BACK efficientformerv2_s1
onnx_pdlite $BACK efficientformerv2_s2

# [F  9/ 4  0: 3:25. 51 ...e-Lite/lite/kernels/host/cast_compute.cc:164 Run] other has not been implemented transform with dtype3 X, dtype0 Out
onnx_pdlite $BACK SwiftFormer_XS
onnx_pdlite $BACK SwiftFormer_S
onnx_pdlite $BACK SwiftFormer_L1

# Exception: The padding value is wrong!
# Exception: convert failed node:_stage3_1_Pad_output_0, op_type is Pad
#onnx_pdlite $BACK EMO_1M
#onnx_pdlite $BACK EMO_2M
#onnx_pdlite $BACK EMO_5M
#onnx_pdlite $BACK EMO_6M

onnx_pdlite $BACK edgenext_xx_small
onnx_pdlite $BACK edgenext_x_small
onnx_pdlite $BACK edgenext_small

# [F  9/ 3  1:42:33.691 ...rk/work/Paddle-Lite/lite/core/op_lite.cc:176 AttachOutput] Check failed: is_dispensable || is_have_output:
# FatalError: `Process abort signal` is detected by the operating system.
#   [TimeInfo: *** Aborted at 1693676553 (unix time) try "date -d @1693676553" if you are using GNU date ***]
#   [SignalInfo: *** SIGABRT (@0x3e800001a20) received by PID 6688 (TID 0x7fd501eaf740) from PID 6688 ***]
#onnx_pdlite $BACK mobilevitv2_050
#onnx_pdlite $BACK mobilevitv2_075
#onnx_pdlite $BACK mobilevitv2_100
#onnx_pdlite $BACK mobilevitv2_125
#onnx_pdlite $BACK mobilevitv2_150
#onnx_pdlite $BACK mobilevitv2_175
#onnx_pdlite $BACK mobilevitv2_200

# [F  9/ 4  0:23:55.496 ...p/Paddle-Lite/lite/operators/slice_op.cc:47 InferShapeImpl] Check failed: (param_.axes[i] < in_dims.size()): -1!<3 The index of dimension in axes must be less than the size of input shape.
# --opset-version == 9 will fix the problem
onnx_pdlite $BACK mobilevit_xx_small
onnx_pdlite $BACK mobilevit_x_small
onnx_pdlite $BACK mobilevit_small

onnx_pdlite $BACK LeViT_128S
onnx_pdlite $BACK LeViT_128
onnx_pdlite $BACK LeViT_192
onnx_pdlite $BACK LeViT_256

onnx_pdlite $BACK resnet50
onnx_pdlite $BACK mobilenetv3_large_100
# Exception: The padding value is wrong!
# Exception: convert failed node:_conv_stem_Pad_output_0, op_type is Pad
#onnx_pdlite $BACK tf_efficientnetv2_b0
#onnx_pdlite $BACK tf_efficientnetv2_b1
#onnx_pdlite $BACK tf_efficientnetv2_b2
#onnx_pdlite $BACK tf_efficientnetv2_b3