# USAGE: bash convert-tools/ncnn-int8.sh
# https://github.com/Tencent/ncnn/blob/master/docs/how-to-use-and-FAQ/quantized-int8-inference.md
# https://github.com/Tencent/ncnn/wiki/FAQ-ncnn-produce-wrong-result#pre-process
export LD_LIBRARY_PATH=$PWD/.libs/ncnn/install/lib

download_calibration()
{
    cd .ncnn
    mkdir -p calibration && cd calibration
    git clone https://github.com/nihui/imagenet-sample-images.git
    rm imagenet-sample-images/.git -rf
    cd ..
    find calibration/imagenet-sample-images -type f > imagelist.txt
    cd ..
}

ncnn_int8()
{
    SHAPE=$1
    PIXEL=$2
    METHOD=$3
    MODEL=$4
    THREAD=12

    cd .ncnn
    mkdir -p $METHOD-int8 opt
    ../.libs/ncnn/install/bin/ncnnoptimize \
        fp32/$MODEL.ncnn.param fp32/$MODEL.ncnn.bin opt/$MODEL.ncnn.param opt/$MODEL.ncnn.bin 0
    ../.libs/ncnn/install/bin/ncnn2table \
        opt/$MODEL.ncnn.param opt/$MODEL.ncnn.bin imagelist.txt $METHOD-int8/$MODEL.ncnn.table \
        mean=0 norm=0 shape=$SHAPE pixel=$PIXEL thread=$THREAD method=$METHOD
    ../.libs/ncnn/install/bin/ncnn2int8 \
        opt/$MODEL.ncnn.param opt/$MODEL.ncnn.bin \
        $METHOD-int8/$MODEL.ncnn.param $METHOD-int8/$MODEL.ncnn.bin $METHOD-int8/$MODEL.ncnn.table
    cd ..
}

# TODO: please first patch `ncnn-tools-quantize.patch` to ncnn repo and rebuild quantize tools
download_calibration
METHOD=kl # we use METHOD=kl(,aciq,eq) as default
ncnn_int8 [224,224,3] RAW $METHOD efficientformerv2_s0
ncnn_int8 [224,224,3] RAW $METHOD efficientformerv2_s1
ncnn_int8 [224,224,3] RAW $METHOD efficientformerv2_s2
ncnn_int8 [256,256,3] RGB $METHOD mobilevitv2_050
ncnn_int8 [256,256,3] RGB $METHOD mobilevitv2_075
ncnn_int8 [256,256,3] RGB $METHOD mobilevitv2_100
ncnn_int8 [256,256,3] RGB $METHOD mobilevitv2_125
ncnn_int8 [256,256,3] RGB $METHOD mobilevitv2_150
ncnn_int8 [256,256,3] RGB $METHOD mobilevitv2_175
ncnn_int8 [256,256,3] RGB $METHOD mobilevitv2_200
ncnn_int8 [224,224,3] BGR $METHOD resnet50
ncnn_int8 [224,224,3] RAW $METHOD mobilenetv3_large_100
ncnn_int8 [224,224,3] RAW $METHOD tf_efficientnetv2_b0
ncnn_int8 [240,240,3] RAW $METHOD tf_efficientnetv2_b1
ncnn_int8 [260,260,3] RAW $METHOD tf_efficientnetv2_b2
ncnn_int8 [300,300,3] GRAY $METHOD tf_efficientnetv2_b3