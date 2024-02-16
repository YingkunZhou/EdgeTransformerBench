# USAGE: bash perf-tools/ncnn-testsuite.sh
download_model()
{
    if [ ! -d ".ncnn" ]
    then
        [ -f "ncnn-models.tar.gz" ] && gdown 1ApgE-Js-uPaGTAnpbNTERQgAuyR5K0su
        tar xf ncnn-models.tar.gz;
    fi
    # gdown 1SEb9g3zBrBGh_uf0PkEJios_AB_G3ItT # downloading ncnn-fp32-models.tar.gz
}

download_library()
{
    cd .libs
    if [ ! -d "ncnn" ]
    then
        [ -f "ncnn.tar.gz"] && wget ncnn.tar.gz
        tar xf ncnn.tar.gz
    fi
    cd ..
}

testsuite()
{
    cd .ncnn; ln -sf $1/* .; cd ..
    BACK=$2 FP=$3 THREADS=$4 MODEL=ALL make run-ncnn-perf 2>/dev/null
    echo " "
}

VULKAN_testsuite()
{
    echo ">>>>>>>>>>>vulkan: fp16 model + fp32 arith<<<<<<<<<"
    testsuite fp16 v 16 1

    echo ">>>>>>>>>>>vulkan: fp16 model + fp16 arith<<<<<<<<<"
    testsuite fp16 v 16 1
}

CPU_testsuite()
{
    echo ">>>>>>>>>>>cpu: fp16 model + fp32 arith<<<<<<<<<"
    testsuite fp16 z 32 $1

    echo ">>>>>>>>>>>cpu: fp16 model + fp16 arith<<<<<<<<<"
    testsuite fp16 z 16 $1

    echo ">>>>>>>>>>>cpu: int8 model<<<<<<<<<"
    testsuite int8 v 16 $1
}

download_model
download_library
if vulkaninfo >/dev/null
then
    VULKAN_testsuite
fi
CPU_testsuite 1
CPU_testsuite 4