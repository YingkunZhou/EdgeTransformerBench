onnx_pdlite()
{
    MODEL=$1
    # https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/docs/user_guides/x2paddle.md
    #x2paddle --framework=onnx --model=.onnx/fp32/$MODEL.onnx --save_dir=.pdlite
    #mv .pdlite/inference_model .pdlite/$MODEL
    #rm .pdlite/model.pdparams .pdlite/x2paddle_code.py .pdlite/__pycache__ -rf
    mkdir -p .pdlite/fp16
    ../update/Paddle-Lite/build.opt/lite/api/opt --model_dir=.pdlite/$MODEL --valid_targets=arm --optimize_out=.pdlite/fp16/$MODEL --enable_fp16=true
    mkdir -p .pdlite/fp32
    ../update/Paddle-Lite/build.opt/lite/api/opt --model_dir=.pdlite/$MODEL --valid_targets=arm --optimize_out=.pdlite/fp32/$MODEL
    mkdir -p .pdlite/opencl
    ../update/Paddle-Lite/build.opt/lite/api/opt --model_dir=.pdlite/$MODEL --valid_targets=opencl --optimize_out=.pdlite/opencl/$MODEL
    #mkdir -p .pdlite/int16
    #../update/Paddle-Lite/build.opt/lite/api/opt --model_dir=.pdlite/$MODEL --valid_targets=arm --optimize_out=.pdlite/int16/$MODEL --quant_model=true --quant_type=QUANT_INT16
    #mkdir -p .pdlite/int8
    #../update/Paddle-Lite/build.opt/lite/api/opt --model_dir=.pdlite/$MODEL --valid_targets=arm --optimize_out=.pdlite/int8/$MODEL --quant_model=true --quant_type=QUANT_INT8
}

onnx_pdlite efficientformerv2_s0
onnx_pdlite efficientformerv2_s1
onnx_pdlite efficientformerv2_s2

# [F  9/ 4  0: 3:25. 51 ...e-Lite/lite/kernels/host/cast_compute.cc:164 Run] other has not been implemented transform with dtype3 X, dtype0 Out
#onnx_pdlite SwiftFormer_XS
#onnx_pdlite SwiftFormer_S
#onnx_pdlite SwiftFormer_L1

# Exception: The padding value is wrong!
# Exception: convert failed node:_stage3_1_Pad_output_0, op_type is Pad
#onnx_pdlite EMO_1M
#onnx_pdlite EMO_2M
#onnx_pdlite EMO_5M
#onnx_pdlite EMO_6M

onnx_pdlite edgenext_xx_small
onnx_pdlite edgenext_x_small
onnx_pdlite edgenext_small

# [F  9/ 3  1:42:33.691 ...rk/work/Paddle-Lite/lite/core/op_lite.cc:176 AttachOutput] Check failed: is_dispensable || is_have_output:
# FatalError: `Process abort signal` is detected by the operating system.
#   [TimeInfo: *** Aborted at 1693676553 (unix time) try "date -d @1693676553" if you are using GNU date ***]
#   [SignalInfo: *** SIGABRT (@0x3e800001a20) received by PID 6688 (TID 0x7fd501eaf740) from PID 6688 ***]
#onnx_pdlite mobilevitv2_050
#onnx_pdlite mobilevitv2_075
#onnx_pdlite mobilevitv2_100
#onnx_pdlite mobilevitv2_125
#onnx_pdlite mobilevitv2_150
#onnx_pdlite mobilevitv2_175
#onnx_pdlite mobilevitv2_200

# [F  9/ 4  0:23:55.496 ...p/Paddle-Lite/lite/operators/slice_op.cc:47 InferShapeImpl] Check failed: (param_.axes[i] < in_dims.size()): -1!<3 The index of dimension in axes must be less than the size of input shape.
# --opset-version == 9 will fix the problem
onnx_pdlite mobilevit_xx_small
onnx_pdlite mobilevit_x_small
onnx_pdlite mobilevit_small

onnx_pdlite LeViT_128S
onnx_pdlite LeViT_128
onnx_pdlite LeViT_192
onnx_pdlite LeViT_256

onnx_pdlite resnet50
onnx_pdlite mobilenetv3_large_100

# Exception: The padding value is wrong!
# Exception: convert failed node:_conv_stem_Pad_output_0, op_type is Pad
#onnx_pdlite tf_efficientnetv2_b0
#onnx_pdlite tf_efficientnetv2_b1
#onnx_pdlite tf_efficientnetv2_b2
#onnx_pdlite tf_efficientnetv2_b3