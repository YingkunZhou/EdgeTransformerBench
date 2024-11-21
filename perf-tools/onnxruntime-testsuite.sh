# USAGE: bash perf-tools/onnxruntime-testsuite.sh
download_model()
{
    if [ ! -d ".onnx" ]
    then
        [ ! -f "onnx-models.tar.gz" ] && gdown 1eS2sGylZeSuUcrSLtrSo2nJafHQvzVIR
        tar xf onnx-models.tar.gz;
    fi
}

download_library()
{
    cd .libs
    if [ ! -d "onnxruntime" ]
    then
        if [ ! -f "onnxruntime.tar.gz" ]
        then
            if uname -a | grep -q Android
            then
                wget https://github.com/YingkunZhou/EdgeTransformerBench/releases/download/v1.1/onnxruntime.tar.gz
            elif uname -m | grep -q x86_64
            then
                wget https://github.com/YingkunZhou/EdgeTransformerBench/releases/download/v2.1/onnxruntime-linux-x64-1.20.0.tgz
            else
                wget https://github.com/YingkunZhou/EdgeTransformerBench/releases/download/v1.0/onnxruntime.tar.gz
            fi
        fi
        if uname -m | grep -q x86_64
        then
            tar xf onnxruntime-linux-x64-1.20.0.tgz
            mv onnxruntime-linux-x64-1.20.0 onnxruntime
        else
            tar xf onnxruntime.tar.gz
        fi
    fi
    cd ..
}

testsuite()
{
    cd .onnx; rm -rf *.onnx; ln -sf $1/*.onnx .; rm LeViT_256.*  mobilevitv2_1[257]*  mobilevitv2_200.*  tf_efficientnetv2_b3.*; cd ..
    BACK=$2 THREADS=$3 MODEL=ALL make run-onnxruntime-perf 2>/dev/null
    echo " "
}

testsuite_xnnpack()
{
    cd .onnx; rm -rf *.onnx; ln -sf $1/*.onnx .; rm efficientformerv2* LeViT_256.*  mobilevitv2_1[257]*  mobilevitv2_200.*  tf_efficientnetv2_b3.*; cd ..
    BACK=$2 THREADS=$3 MODEL=ALL make run-onnxruntime-perf 2>/dev/null
    echo " "
}

NNAPI_testsuite()
{
    echo ">>>>>>>>>>>nnapi: fp32 model<<<<<<<<<"
    testsuite fp32 n 1

    echo ">>>>>>>>>>>nnapi: static ptq int8 model<<<<<<<<<"
    testsuite int8 n 1
}

QNN_testsuite()
{
    echo ">>>>>>>>>>>qnn: fp32 model<<<<<<<<<"
    testsuite fp32 q 1

    echo ">>>>>>>>>>>qnn: static ptq int8 model<<<<<<<<<"
    testsuite int8 q 1
}

CPU_testsuite()
{
    echo ">>>>>>>>>>>xnnpack: fp32 model<<<<<<<<<"
    testsuite_xnnpack fp32 x $1

    echo ">>>>>>>>>>>xnnpack: static ptq int8 model<<<<<<<<<"
    testsuite_xnnpack int8 x $1

    echo ">>>>>>>>>>>CPU: fp32 model<<<<<<<<<"
    testsuite fp32 z $1

    echo ">>>>>>>>>>>CPU: static ptq int8 model<<<<<<<<<"
    testsuite int8 z $1
}

download_model
download_library

if uname -a | grep -q Android
then
    NNAPI_testsuite
    # QNN_testsuite
fi
CPU_testsuite 1
if uname -m | grep -q x86_64
then
    echo "onnxruntime multi-thread support in x86 linux is rubbish!!!"
else
    CPU_testsuite 3 # 3+1 threads
fi
