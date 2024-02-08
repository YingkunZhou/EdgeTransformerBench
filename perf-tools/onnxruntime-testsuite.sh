# USAGE: bash perf-tools/tflite-testsuite.sh
download_model()
{
    if [ ! -d ".onnx" ]
    then
        [ -f "onnx-models.tar.gz" ] && gdown 1wDlAcXP2kU17yjFFUBG2Uwd1O_hemMPl
        tar xf onnx-models.tar.gz;
    fi
}

download_library()
{
    cd .libs
    if [ ! -d "onnxruntime" ]
    then
        [ -f "onnxruntime.tar.gz"] && wget onnxruntime.tar.gz
        tar xf onnxruntime.tar.gz
    fi
}

testsuite()
{
    cd .onnx; rm -rf *.onnx; ln -sf $1/* .; cd ..
    BACK=$2 THREADS=$3 MODEL=ALL make run-onnxruntime-perf 2>/dev/null
    echo " "
}

download_model
download_library

echo ">>>>>>>>>>>nnapi: fp32 model<<<<<<<<<"
testsuite fp32 n 1

echo ">>>>>>>>>>>nnapi: static ptq int8 model<<<<<<<<<"
testsuite int8 n 1

echo ">>>>>>>>>>>qnn: fp32 model<<<<<<<<<"
testsuite fp32 q 1

echo ">>>>>>>>>>>qnn: static ptq int8 model<<<<<<<<<"
testsuite int8 q 1

echo ">>>>>>>>>>>xnnpack: fp32 model<<<<<<<<<"
testsuite fp32 x 1

echo ">>>>>>>>>>>xnnpack: static ptq int8 model<<<<<<<<<"
testsuite int8 x 1

echo ">>>>>>>>>>>CPU: fp32 model<<<<<<<<<"
testsuite fp32 z 1

echo ">>>>>>>>>>>CPU: static ptq int8 model<<<<<<<<<"
testsuite int8 z 1

echo ">>>>>>>>>>>xnnpack: fp32 model multcore<<<<<<<<<"
testsuite fp32 x 4

echo ">>>>>>>>>>>xnnpack: static ptq int8 model multcore<<<<<<<<<"
testsuite int8 x 4

echo ">>>>>>>>>>>CPU: fp32 model multcore<<<<<<<<<"
testsuite fp32 z 4

echo ">>>>>>>>>>>CPU: static ptq int8 model multcore<<<<<<<<<"
testsuite int8 z 4