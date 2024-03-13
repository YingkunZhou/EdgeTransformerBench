# USAGE: bash perf-tools/pdlite-testsuite.sh
download_model()
{
    if [ ! -d ".pdlite" ]
    then
        [ ! -f "pdlite-models.tar.gz" ] && gdown 1KpTZxZ-mo1TEnXIMkOBEgktroI5vqLBw
        tar xf pdlite-models.tar.gz;
        if clinfo >/dev/null
        then
            [ ! -f "pdlite-opencl-models.tar.gz" ] && gdown 1F9ZtUPMG1NKPYOknGU_PMRnqXLukqo5P
            tar xf pdlite-opencl-models.tar.gz;
        fi
    fi
}

download_library()
{
    cd .libs
    if [ ! -d "Paddle-Lite" ]
    then
        if uname -a | grep -q Android
        then
            [ ! -f "Paddle-Lite-ARM82.tar.gz" ] && wget https://github.com/YingkunZhou/EdgeTransformerBench/releases/download/v1.1/Paddle-Lite-ARM82.tar.gz
            tar xf Paddle-Lite-ARM82.tar.gz
        elif cat /proc/cpuinfo | grep -q asimdhp
        then
            [ ! -f "Paddle-Lite-ARM82.tar.gz" ] && wget https://github.com/YingkunZhou/EdgeTransformerBench/releases/download/v1.0/Paddle-Lite-ARM82.tar.gz
            tar xf Paddle-Lite-ARM82.tar.gz
        else
            [ ! -f "Paddle-Lite.tar.gz" ] && wget https://github.com/YingkunZhou/EdgeTransformerBench/releases/download/v1.0/Paddle-Lite.tar.gz
            tar xf Paddle-Lite.tar.gz
        fi
    fi
    cd ..
}

testsuite()
{
    cd .pdlite; rm -rf *.nb; ln -sf $1/*.nb .; rm SwiftFormer* LeViT_256.*  mobilevitv2_1[257]*  mobilevitv2_200.*  tf_efficientnetv2_b3.*; cd ..
    BACK=$2 FP=$3 THREADS=$4 MODEL=ALL make run-pdlite-perf 2>/dev/null
    echo " "
}

testsuite_opencl()
{
    cd .pdlite; rm -rf *.nb; ln -sf opencl/*.nb .; rm LeViT_256.*; cd ..
    BACK=$1 FP=$2 THREADS=$3 MODEL=efficientformerv2_ make run-pdlite-perf 2>/dev/null
    BACK=$1 FP=$2 THREADS=$3 MODEL=EMO_ make run-pdlite-perf 2>/dev/null
    BACK=$1 FP=$2 THREADS=$3 MODEL=edgenext_ make run-pdlite-perf 2>/dev/null
    BACK=$1 FP=$2 THREADS=$3 MODEL=mobilevit_ make run-pdlite-perf 2>/dev/null
    BACK=$1 FP=$2 THREADS=$3 MODEL=LeViT_ make run-pdlite-perf 2>/dev/null
    BACK=$1 FP=$2 THREADS=$3 MODEL=resnet50 make run-pdlite-perf 2>/dev/null
    BACK=$1 FP=$2 THREADS=$3 MODEL=mobilenetv3_large_100 make run-pdlite-perf 2>/dev/null
    echo " "
}

OPENCL_testsuite()
{
    echo ">>>>>>>>>>>opencl: fp16 model + fp32 arith<<<<<<<<<"
    testsuite_opencl o 32 1

    echo ">>>>>>>>>>>opencl: fp16 model + fp16 arith<<<<<<<<<"
    testsuite_opencl o 16 1
}

CPU_testsuite()
{
    echo ">>>>>>>>>>>cpu: fp32<<<<<<<<<"
    testsuite fp32 z 32 $1
    ### fp16
    if cat /proc/cpuinfo | grep -q asimdhp
    then
        echo ">>>>>>>>>>>cpu: fp16<<<<<<<<<"
        testsuite fp16 z 16 $1
    fi
    echo ">>>>>>>>>>>cpu: int8<<<<<<<<<"
    testsuite int8 z 32 $1
}

download_model
download_library

if clinfo >/dev/null
then
    OPENCL_testsuite
fi
CPU_testsuite 1
CPU_testsuite 4