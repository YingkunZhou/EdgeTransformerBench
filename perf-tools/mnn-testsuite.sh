# USAGE: bash perf-tools/mnn-testsuite.sh
download_model()
{
    if [ ! -d ".mnn" ]
    then
        [ -f "mnn-models.tar.gz" ] && gdown 1nBOlcsk2E_RXwwmc2gQa0OTzFuQe4v-3
        tar xf mnn-models.tar.gz;
    fi
    # gdown 1XqF9my9TguiKKlaKxvfvGAXTDZh6HUSG # downloading mnn-fp32-models.tar.gz
}

download_library()
{
    cd .libs
    if [ ! -d "MNN" ]
    then
        [ -f "MNN.tar.gz"] && wget MNN.tar.gz
        tar xf MNN.tar.gz
    fi
    cd ..
}

testsuite()
{
    cd .mnn; rm -rf *.mnn; ln -sf $1/* .; rm LeViT_256.*  .mnn/mobilevitv2_1[257]*  mobilevitv2_200.*  tf_efficientnetv2_b3.*; cd ..
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

    echo ">>>>>>>>>>>cpu: fp16 model + fp16 arith<<<<<<<<<"
    testsuite fp16 z 16 $1
}

download_model
download_library

# TODO: CPU & GPU int8
if clinfo >/dev/null
then
    OPENCL_testsuite
fi

if vulkaninfo >/dev/null
then
    VULKAN_testsuite
fi
CPU_testsuite 1
CPU_testsuite 4