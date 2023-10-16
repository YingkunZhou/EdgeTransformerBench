download_calibration()
{
    cd .tnn
    ln -sf ../.ncnn/calibration .
    cd -
}

tnn_int8()
{
    MODEL=$1
    BLOB=$2
    WEIGHT=$3
    MERGE=$4
    BIAS=$5
    SCALE=$6
    cd .tnn
    ../.libs/TNN/platforms/linux/build_quantize/quantization_cmd -p fp32/$MODEL.opt.tnnproto -m fp32/$MODEL.opt.tnnmodel \
    -b $BLOB -w $WEIGHT -t $MERGE -n $BIAS -s $SCALE -i calibration/imagenet-sample-images
    mv model.quantized.tnnmodel int8/$MODEL.opt.tnnmodel; mv model.quantized.tnnproto int8/$MODEL.opt.tnnproto
    cd -
}

download_calibration
tnn_int8 efficientformerv2_s0 2 1 1 123.675,116.28,103.53 0.017124753831663668,0.01750700280112045,0.017429193899782137
#tnn_int8 LeViT_128S 2 1 1 123.675,116.28,103.53 0.017124753831663668,0.01750700280112045,0.017429193899782137
