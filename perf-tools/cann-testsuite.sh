FMT=$1
NOSLEEP=0
SLEEP="${2:-$NOSLEEP}"
EVAL=$3
DATA=$4

# USAGE: bash perf-tools/cann-testsuite.sh

if [ ! -d ".cann" ]
then
    [ ! -f "cann-models.tar.gz" ] && gdown 12kU9snxMNiZiGV_qPdwhPbeqyPqTV0GL
    tar xf cann-models.tar.gz;
fi

python python/cann-perf.py --format $FMT $EVAL $DATA --only-test efficientformerv2_s0 2>/dev/null ; sleep $SLEEP
python python/cann-perf.py --format $FMT $EVAL $DATA --only-test efficientformerv2_s1 2>/dev/null ; sleep $SLEEP
python python/cann-perf.py --format $FMT $EVAL $DATA --only-test efficientformerv2_s2 2>/dev/null ; sleep $SLEEP
python python/cann-perf.py --format $FMT $EVAL $DATA --only-test SwiftFormer_XS 2>/dev/null ; sleep $SLEEP
python python/cann-perf.py --format $FMT $EVAL $DATA --only-test SwiftFormer_S 2>/dev/null ; sleep $SLEEP
python python/cann-perf.py --format $FMT $EVAL $DATA --only-test SwiftFormer_L1 2>/dev/null ; sleep $SLEEP
python python/cann-perf.py --format $FMT $EVAL $DATA --only-test EMO_1M 2>/dev/null ; sleep $SLEEP
python python/cann-perf.py --format $FMT $EVAL $DATA --only-test EMO_2M 2>/dev/null ; sleep $SLEEP
# python python/cann-perf.py --format $FMT $EVAL $DATA --only-test EMO_5M 2>/dev/null ; sleep $SLEEP
python python/cann-perf.py --format $FMT $EVAL $DATA --only-test EMO_6M 2>/dev/null ; sleep $SLEEP
python python/cann-perf.py --format $FMT $EVAL $DATA --only-test edgenext_xx_small 2>/dev/null ; sleep $SLEEP
python python/cann-perf.py --format $FMT $EVAL $DATA --only-test edgenext_x_small 2>/dev/null ; sleep $SLEEP
python python/cann-perf.py --format $FMT $EVAL $DATA --only-test edgenext_small 2>/dev/null ; sleep $SLEEP
python python/cann-perf.py --format $FMT $EVAL $DATA --only-test mobilevitv2_050 2>/dev/null ; sleep $SLEEP
python python/cann-perf.py --format $FMT $EVAL $DATA --only-test mobilevitv2_075 2>/dev/null ; sleep $SLEEP
python python/cann-perf.py --format $FMT $EVAL $DATA --only-test mobilevitv2_100 2>/dev/null ; sleep $SLEEP
# python python/cann-perf.py --format $FMT $EVAL $DATA --only-test mobilevitv2_125 2>/dev/null ; sleep $SLEEP
# python python/cann-perf.py --format $FMT $EVAL $DATA --only-test mobilevitv2_150 2>/dev/null ; sleep $SLEEP
# python python/cann-perf.py --format $FMT $EVAL $DATA --only-test mobilevitv2_175 2>/dev/null ; sleep $SLEEP
# python python/cann-perf.py --format $FMT $EVAL $DATA --only-test mobilevitv2_200 2>/dev/null ; sleep $SLEEP
python python/cann-perf.py --format $FMT $EVAL $DATA --only-test mobilevit_xx_small 2>/dev/null ; sleep $SLEEP
python python/cann-perf.py --format $FMT $EVAL $DATA --only-test mobilevit_x_small 2>/dev/null ; sleep $SLEEP
python python/cann-perf.py --format $FMT $EVAL $DATA --only-test mobilevit_small 2>/dev/null ; sleep $SLEEP
python python/cann-perf.py --format $FMT $EVAL $DATA --only-test LeViT_128S 2>/dev/null ; sleep $SLEEP
python python/cann-perf.py --format $FMT $EVAL $DATA --only-test LeViT_128 2>/dev/null ; sleep $SLEEP
python python/cann-perf.py --format $FMT $EVAL $DATA --only-test LeViT_192 2>/dev/null ; sleep $SLEEP
# python python/cann-perf.py --format $FMT $EVAL $DATA --only-test LeViT_256 2>/dev/null ; sleep $SLEEP
python python/cann-perf.py --format $FMT $EVAL $DATA --only-test resnet50 2>/dev/null ; sleep $SLEEP
python python/cann-perf.py --format $FMT $EVAL $DATA --only-test mobilenetv3_large_100 2>/dev/null ; sleep $SLEEP
python python/cann-perf.py --format $FMT $EVAL $DATA --only-test tf_efficientnetv2_b0 2>/dev/null ; sleep $SLEEP
python python/cann-perf.py --format $FMT $EVAL $DATA --only-test tf_efficientnetv2_b1 2>/dev/null ; sleep $SLEEP
python python/cann-perf.py --format $FMT $EVAL $DATA --only-test tf_efficientnetv2_b2 2>/dev/null ; sleep $SLEEP
# python python/cann-perf.py --format $FMT $EVAL $DATA --only-test tf_efficientnetv2_b3 2>/dev/null ; sleep $SLEEP