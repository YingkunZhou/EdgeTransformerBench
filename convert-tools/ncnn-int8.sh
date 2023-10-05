# https://github.com/Tencent/ncnn/blob/master/docs/how-to-use-and-FAQ/quantized-int8-inference.md
# https://github.com/Tencent/ncnn/wiki/FAQ-ncnn-produce-wrong-result#pre-process
export LD_LIBRARY_PATH=$PWD/.libs/ncnn/install/lib
download_calibration()
{
    cd .ncnn
    git clone https://github.com/nihui/imagenet-sample-images.git
    find imagenet-sample-images/ -type f > imagelist.txt
    cd -
}

ncnn_int8()
{
    SHAPE=$1
    MODEL=$2
    MEAN=$3
    NORM=$4
    cd .ncnn
    ../.libs/ncnn/install/bin/ncnnoptimize $MODEL.ncnn.param $MODEL.ncnn.bin $MODEL.ncnn.opt.param $MODEL.ncnn.opt.bin 0
    ../.libs/ncnn/install/bin/ncnn2table $MODEL.ncnn.opt.param $MODEL.ncnn.opt.bin imagelist.txt $MODEL.ncnn.table mean=$MEAN norm=$NORM shape=$SHAPE pixel=RGB thread=12 method=eq
    ../.libs/ncnn/install/bin/ncnn2int8 $MODEL.ncnn.opt.param $MODEL.ncnn.opt.bin $MODEL.ncnn.int8.param $MODEL.ncnn.int8.bin $MODEL.ncnn.table
    cd -
}


download_calibration
ncnn_int8 [224,224,3] efficientformerv2_s0 [123.675,116.28,103.53] [0.017124753831663668,0.01750700280112045,0.017429193899782137]
ncnn_int8 [224,224,3] efficientformerv2_s1 [123.675,116.28,103.53] [0.017124753831663668,0.01750700280112045,0.017429193899782137]
ncnn_int8 [224,224,3] efficientformerv2_s2 [123.675,116.28,103.53] [0.017124753831663668,0.01750700280112045,0.017429193899782137]

ncnn_int8 [256,256,3] mobilevitv2_050 [0,0,0] [0.00392156862745098,0.00392156862745098,0.00392156862745098]
ncnn_int8 [256,256,3] mobilevitv2_075 [0,0,0] [0.00392156862745098,0.00392156862745098,0.00392156862745098]
ncnn_int8 [256,256,3] mobilevitv2_100 [0,0,0] [0.00392156862745098,0.00392156862745098,0.00392156862745098]
ncnn_int8 [256,256,3] mobilevitv2_125 [0,0,0] [0.00392156862745098,0.00392156862745098,0.00392156862745098]
ncnn_int8 [256,256,3] mobilevitv2_150 [0,0,0] [0.00392156862745098,0.00392156862745098,0.00392156862745098]
ncnn_int8 [256,256,3] mobilevitv2_175 [0,0,0] [0.00392156862745098,0.00392156862745098,0.00392156862745098]
ncnn_int8 [256,256,3] mobilevitv2_200 [0,0,0] [0.00392156862745098,0.00392156862745098,0.00392156862745098]

ncnn_int8 [224,224,3] resnet50 [123.675,116.28,103.53] [0.017124753831663668,0.01750700280112045,0.017429193899782137]
ncnn_int8 [224,224,3] mobilenetv3_large_100 [123.675,116.28,103.53] [0.017124753831663668,0.01750700280112045,0.017429193899782137]
ncnn_int8 [224,224,3] tf_efficientnetv2_b0 [123.675,116.28,103.53] [0.017124753831663668,0.01750700280112045,0.017429193899782137]
ncnn_int8 [240,240,3] tf_efficientnetv2_b1 [123.675,116.28,103.53] [0.017124753831663668,0.01750700280112045,0.017429193899782137]
ncnn_int8 [260,260,3] tf_efficientnetv2_b2 [123.675,116.28,103.53] [0.017124753831663668,0.01750700280112045,0.017429193899782137]
ncnn_int8 [300,300,3] tf_efficientnetv2_b3 [127.5,127.5,127.5] [0.00784313725490196,0.00784313725490196,0.00784313725490196]