FMT=$1
DEV=$2
NOSLEEP=0
SLEEP="${3:-$NOSLEEP}"
EVAL=$4
DATA=$5

rm bin/onnxruntime-perf; TENSORRT_FLAGS="-DUSE_TENSORRT" make bin/onnxruntime-perf
cd .onnx; rm -rf *.onnx; ln -sf fp32/*.onnx .; rm LeViT_256.*  mobilevitv2_1[257]*  mobilevitv2_200.*  tf_efficientnetv2_b3.*; cd ..

python python/trt-perf.py --format $FMT --trt-dev $DEV $EVAL $DATA --only-test efficientformerv2_s0 2>/dev/null ; sleep $SLEEP
python python/trt-perf.py --format $FMT --trt-dev $DEV $EVAL $DATA --only-test efficientformerv2_s1 2>/dev/null ; sleep $SLEEP
python python/trt-perf.py --format $FMT --trt-dev $DEV $EVAL $DATA --only-test efficientformerv2_s2 2>/dev/null ; sleep $SLEEP
python python/trt-perf.py --format $FMT --trt-dev $DEV $EVAL $DATA --only-test SwiftFormer_XS 2>/dev/null ; sleep $SLEEP
python python/trt-perf.py --format $FMT --trt-dev $DEV $EVAL $DATA --only-test SwiftFormer_S 2>/dev/null ; sleep $SLEEP
python python/trt-perf.py --format $FMT --trt-dev $DEV $EVAL $DATA --only-test SwiftFormer_L1 2>/dev/null ; sleep $SLEEP
python python/trt-perf.py --format $FMT --trt-dev $DEV $EVAL $DATA --only-test EMO_1M 2>/dev/null ; sleep $SLEEP
python python/trt-perf.py --format $FMT --trt-dev $DEV $EVAL $DATA --only-test EMO_2M 2>/dev/null ; sleep $SLEEP
# python python/trt-perf.py --format $FMT --trt-dev $DEV $EVAL $DATA --only-test EMO_5M 2>/dev/null ; sleep $SLEEP
python python/trt-perf.py --format $FMT --trt-dev $DEV $EVAL $DATA --only-test EMO_6M 2>/dev/null ; sleep $SLEEP
python python/trt-perf.py --format $FMT --trt-dev $DEV $EVAL $DATA --only-test edgenext_xx_small 2>/dev/null ; sleep $SLEEP
python python/trt-perf.py --format $FMT --trt-dev $DEV $EVAL $DATA --only-test edgenext_x_small 2>/dev/null ; sleep $SLEEP
python python/trt-perf.py --format $FMT --trt-dev $DEV $EVAL $DATA --only-test edgenext_small 2>/dev/null ; sleep $SLEEP
python python/trt-perf.py --format $FMT --trt-dev $DEV $EVAL $DATA --only-test mobilevitv2_050 2>/dev/null ; sleep $SLEEP
python python/trt-perf.py --format $FMT --trt-dev $DEV $EVAL $DATA --only-test mobilevitv2_075 2>/dev/null ; sleep $SLEEP
python python/trt-perf.py --format $FMT --trt-dev $DEV $EVAL $DATA --only-test mobilevitv2_100 2>/dev/null ; sleep $SLEEP
# python python/trt-perf.py --format $FMT --trt-dev $DEV $EVAL $DATA --only-test mobilevitv2_125 2>/dev/null ; sleep $SLEEP
# python python/trt-perf.py --format $FMT --trt-dev $DEV $EVAL $DATA --only-test mobilevitv2_150 2>/dev/null ; sleep $SLEEP
# python python/trt-perf.py --format $FMT --trt-dev $DEV $EVAL $DATA --only-test mobilevitv2_175 2>/dev/null ; sleep $SLEEP
# python python/trt-perf.py --format $FMT --trt-dev $DEV $EVAL $DATA --only-test mobilevitv2_200 2>/dev/null ; sleep $SLEEP
python python/trt-perf.py --format $FMT --trt-dev $DEV $EVAL $DATA --only-test mobilevit_xx_small 2>/dev/null ; sleep $SLEEP
python python/trt-perf.py --format $FMT --trt-dev $DEV $EVAL $DATA --only-test mobilevit_x_small 2>/dev/null ; sleep $SLEEP
python python/trt-perf.py --format $FMT --trt-dev $DEV $EVAL $DATA --only-test mobilevit_small 2>/dev/null ; sleep $SLEEP
python python/trt-perf.py --format $FMT --trt-dev $DEV $EVAL $DATA --only-test LeViT_128S 2>/dev/null ; sleep $SLEEP
python python/trt-perf.py --format $FMT --trt-dev $DEV $EVAL $DATA --only-test LeViT_128 2>/dev/null ; sleep $SLEEP
python python/trt-perf.py --format $FMT --trt-dev $DEV $EVAL $DATA --only-test LeViT_192 2>/dev/null ; sleep $SLEEP
# python python/trt-perf.py --format $FMT --trt-dev $DEV $EVAL $DATA --only-test LeViT_256 2>/dev/null ; sleep $SLEEP
python python/trt-perf.py --format $FMT --trt-dev $DEV $EVAL $DATA --only-test resnet50 2>/dev/null ; sleep $SLEEP
python python/trt-perf.py --format $FMT --trt-dev $DEV $EVAL $DATA --only-test mobilenetv3_large_100 2>/dev/null ; sleep $SLEEP
python python/trt-perf.py --format $FMT --trt-dev $DEV $EVAL $DATA --only-test tf_efficientnetv2_b0 2>/dev/null ; sleep $SLEEP
python python/trt-perf.py --format $FMT --trt-dev $DEV $EVAL $DATA --only-test tf_efficientnetv2_b1 2>/dev/null ; sleep $SLEEP
python python/trt-perf.py --format $FMT --trt-dev $DEV $EVAL $DATA --only-test tf_efficientnetv2_b2 2>/dev/null ; sleep $SLEEP
# python python/trt-perf.py --format $FMT --trt-dev $DEV $EVAL $DATA --only-test tf_efficientnetv2_b3 2>/dev/null ; sleep $SLEEP