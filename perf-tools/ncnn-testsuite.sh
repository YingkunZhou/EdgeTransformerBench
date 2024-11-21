# USAGE: bash perf-tools/ncnn-testsuite.sh
download_model()
{
    if [ ! -d ".ncnn" ]
    then
        [ ! -f "ncnn-models.tar.gz" ] && gdown 1ApgE-Js-uPaGTAnpbNTERQgAuyR5K0su
        tar xf ncnn-models.tar.gz;
    fi
    # gdown 1SEb9g3zBrBGh_uf0PkEJios_AB_G3ItT # downloading ncnn-fp32-models.tar.gz
}

download_library()
{
    cd .libs
    if [ ! -d "ncnn" ]
    then
        if [ ! -f "ncnn.tar.gz" ]
        then
            if uname -a | grep -q Android
            then
                wget https://github.com/YingkunZhou/EdgeTransformerBench/releases/download/v1.1/ncnn.tar.gz
            elif uname -m | grep -q x86_64
            then
                wget https://github.com/YingkunZhou/EdgeTransformerBench/releases/download/v2.1/ncnn-20240820-ubuntu-2404-shared.zip
            else
                wget https://github.com/YingkunZhou/EdgeTransformerBench/releases/download/v1.0/ncnn.tar.gz
            fi
        fi
        if uname -m | grep -q x86_64
        then
            unzip ncnn-20240820-ubuntu-2404-shared.zip
            mkdir ncnn && cd ncnn
            ln -sf ../ncnn-20240820-ubuntu-2404-shared install
        else
            tar xf ncnn.tar.gz
        fi
    fi
    cd ..
}

testsuite()
{
    cd .ncnn; rm *;  ln -sf $1/* .; rm LeViT_256.*  mobilevitv2_1[257]*  mobilevitv2_200.*  mobilevit_* tf_efficientnetv2_b3.*; cd ..
    BACK=$2 FP=$3 THREADS=$4 MODEL=ALL make run-ncnn-perf 2>/dev/null
    echo " "
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
    testsuite eq-int8 z 16 $1
}

download_model
download_library
if uname -m | grep -q x86_64
then
    echo "skip vulkan testsuite in x86 platform"
elif vulkaninfo >/dev/null
then
    VULKAN_testsuite
fi
CPU_testsuite 1
CPU_testsuite 4