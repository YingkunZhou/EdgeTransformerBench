# USAGE: bash perf-tools/mnn-testsuite.sh
download_model()
{
    if [ ! -d ".mnn" ]
    then
        [ ! -f "mnn-models.tar.gz" ] && gdown 1be8LdqVZ_AJuwF2Ycc28jhytZ4kMkoV4
        tar xf mnn-models.tar.gz;
    fi
    # gdown 1XqF9my9TguiKKlaKxvfvGAXTDZh6HUSG # downloading mnn-fp32-models.tar.gz
}

download_library()
{
    cd .libs
    if [ ! -d "MNN" ]
    then
        if [ ! -f "MNN.tar.gz" ]
        then
            if uname -a | grep -q Android
            then
                wget https://github.com/YingkunZhou/EdgeTransformerBench/releases/download/v1.1/MNN.tar.gz
            elif uname -m | grep -q x86_64
            then
                wget https://github.com/YingkunZhou/EdgeTransformerBench/releases/download/v2.1/MNN.tar.gz
            else
                wget https://github.com/YingkunZhou/EdgeTransformerBench/releases/download/v1.0/MNN.tar.gz
            fi
        fi
        tar xf MNN.tar.gz
    fi
    cd ..
}

testsuite()
{
    cd .mnn; rm -rf *.mnn; ln -sf $1/*.mnn .; rm LeViT_256.*  mobilevitv2_1[257]*  mobilevitv2_200.*  tf_efficientnetv2_b3.*; cd ..
    BACK=$2 FP=$3 THREADS=$4 MODEL=ALL make run-mnn-perf 2>/dev/null
    echo " "
}

OPENCL_testsuite()
{
    echo ">>>>>>>>>>>opencl: fp16 model + fp32 arith<<<<<<<<<"
    testsuite fp16 o 32 1

    echo ">>>>>>>>>>>opencl: fp16 model + fp16 arith<<<<<<<<<"
    testsuite fp16 o 16 1
}

VULKAN_testsuite()
{
    echo ">>>>>>>>>>>vulkan: fp16 model + fp32 arith<<<<<<<<<"
    testsuite fp16 v 32 1

    echo ">>>>>>>>>>>vulkan: fp16 model + fp16 arith<<<<<<<<<"
    testsuite fp16 v 16 1
}

CPU_testsuite()
{
    echo ">>>>>>>>>>>cpu: fp16 model + fp32 arith<<<<<<<<<"
    testsuite fp16 z 32 $1

    if uname -m | grep -q x86_64
    then
        echo "intel client don't support avx512 fp16 instr"
    else
        echo ">>>>>>>>>>>cpu: fp16 model + fp16 arith<<<<<<<<<"
        testsuite fp16 z 16 $1
    fi

    echo ">>>>>>>>>>>cpu: int8 model<<<<<<<<<"
    # TODO: unable to run in aipro and m1, because missing SMMLA instruction
    testsuite int8 z 16 $1

}

download_model
download_library

if uname -m | grep -q x86_64
then
    echo "skip opencl & vulkan testsuite in x86 platform"
else
    if vulkaninfo | grep -q Adreno
    then
        cp /vendor/lib64/libOpenCL.so .libs/MNN/install/lib
    fi

    if clinfo >/dev/null
    then
        OPENCL_testsuite
    fi

    if vulkaninfo >/dev/null
    then
        VULKAN_testsuite
    fi
fi
CPU_testsuite 1
CPU_testsuite 4
