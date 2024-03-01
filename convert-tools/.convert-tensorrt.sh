FORMAT=$1
DEV=$2

python python/trt-convert.py --format $FORMAT --trt-dev $DEV --only-convert efficientformerv2_s0
python python/trt-convert.py --format $FORMAT --trt-dev $DEV --only-convert efficientformerv2_s1
python python/trt-convert.py --format $FORMAT --trt-dev $DEV --only-convert efficientformerv2_s2
python python/trt-convert.py --format $FORMAT --trt-dev $DEV --only-convert SwiftFormer_XS
python python/trt-convert.py --format $FORMAT --trt-dev $DEV --only-convert SwiftFormer_S
python python/trt-convert.py --format $FORMAT --trt-dev $DEV --only-convert SwiftFormer_L1
python python/trt-convert.py --format $FORMAT --trt-dev $DEV --only-convert EMO_1M
python python/trt-convert.py --format $FORMAT --trt-dev $DEV --only-convert EMO_2M
python python/trt-convert.py --format $FORMAT --trt-dev $DEV --only-convert EMO_5M
python python/trt-convert.py --format $FORMAT --trt-dev $DEV --only-convert EMO_6M
python python/trt-convert.py --format $FORMAT --trt-dev $DEV --only-convert edgenext_xx_small
python python/trt-convert.py --format $FORMAT --trt-dev $DEV --only-convert edgenext_x_small
python python/trt-convert.py --format $FORMAT --trt-dev $DEV --only-convert edgenext_small
python python/trt-convert.py --format $FORMAT --trt-dev $DEV --only-convert mobilevitv2_050
python python/trt-convert.py --format $FORMAT --trt-dev $DEV --only-convert mobilevitv2_075
python python/trt-convert.py --format $FORMAT --trt-dev $DEV --only-convert mobilevitv2_100
python python/trt-convert.py --format $FORMAT --trt-dev $DEV --only-convert mobilevitv2_125
python python/trt-convert.py --format $FORMAT --trt-dev $DEV --only-convert mobilevitv2_150
python python/trt-convert.py --format $FORMAT --trt-dev $DEV --only-convert mobilevitv2_175
python python/trt-convert.py --format $FORMAT --trt-dev $DEV --only-convert mobilevitv2_200
python python/trt-convert.py --format $FORMAT --trt-dev $DEV --only-convert mobilevit_xx_small
python python/trt-convert.py --format $FORMAT --trt-dev $DEV --only-convert mobilevit_x_small
python python/trt-convert.py --format $FORMAT --trt-dev $DEV --only-convert mobilevit_small
python python/trt-convert.py --format $FORMAT --trt-dev $DEV --only-convert LeViT_128
python python/trt-convert.py --format $FORMAT --trt-dev $DEV --only-convert LeViT_192
python python/trt-convert.py --format $FORMAT --trt-dev $DEV --only-convert LeViT_256
python python/trt-convert.py --format $FORMAT --trt-dev $DEV --only-convert resnet50
python python/trt-convert.py --format $FORMAT --trt-dev $DEV --only-convert mobilenetv3_large_100
python python/trt-convert.py --format $FORMAT --trt-dev $DEV --only-convert tf_efficientnetv2_b0
python python/trt-convert.py --format $FORMAT --trt-dev $DEV --only-convert tf_efficientnetv2_b1
python python/trt-convert.py --format $FORMAT --trt-dev $DEV --only-convert tf_efficientnetv2_b2
python python/trt-convert.py --format $FORMAT --trt-dev $DEV --only-convert tf_efficientnetv2_b3