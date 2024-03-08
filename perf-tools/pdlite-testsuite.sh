# USAGE: bash perf-tools/pdlite-testsuite.sh
download_model()
{
    if [ ! -d ".pdlite" ]
    then
        [ -f "pdlite-models.tar.gz" ] || gdown 1KpTZxZ-mo1TEnXIMkOBEgktroI5vqLBw
        tar xf pdlite-models.tar.gz;
        if clinfo >/dev/null
        then
            [ -f "pdlite-opencl-models.tar.gz" ] || gdown 1F9ZtUPMG1NKPYOknGU_PMRnqXLukqo5P
            tar xf pdlite-opencl-models.tar.gz;
        fi
    fi
}

download_library()
{
    cd .libs
    if [ ! -d "Paddle-Lite" ]
    then
        [ -f "Paddle-Lite.tar.gz" ] || wget Paddle-Lite.tar.gz
        tar xf Paddle-Lite.tar.gz
    fi
    cd ..
}

testsuite()
{
    cd .pdlite; rm -rf *.nb; ln -sf $1/*.nb .; rm LeViT_256.*  mobilevitv2_1[257]*  mobilevitv2_200.*  tf_efficientnetv2_b3.*; cd ..
    BACK=$2 FP=$3 THREADS=$4 MODEL=ALL make run-pdlite-perf 2>/dev/null
    echo " "
}

OPENCL_testsuite()
{
    echo ">>>>>>>>>>>opencl: fp16 model + fp32 arith<<<<<<<<<"
    testsuite opencl o 32 1

    echo ">>>>>>>>>>>opencl: fp16 model + fp16 arith<<<<<<<<<"
    testsuite opencl o 16 1
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