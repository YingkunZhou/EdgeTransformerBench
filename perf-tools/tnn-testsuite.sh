# USAGE: bash perf-tools/tnn-testsuite.sh
download_model()
{
    if [ ! -d ".tnn" ]
    then
        [ ! -f "tnn-models.tar.gz" ] && gdown 1OXrtjeLA0VGt7mLS7nSc1ipgIJ3BK7k5
        tar xf tnn-models.tar.gz;
    fi
    # gdown 1Qz0U4pPslYsdbkW16Y0lWuGsZBNSL0qf # downloading tnn-fp32-models.tar.gz
}

download_library()
{
    cd .libs
    if [ ! -d "TNN" ]
    then
        [ ! -f "TNN.tar.gz"] && wget TNN.tar.gz
        tar xf TNN.tar.gz
    fi
    cd ..
}

testsuite()
{
    cd .tnn; rm -rf *.tnn; ln -sf $1/*.tnn .; rm LeViT_256.*  mobilevitv2_1[257]*  mobilevitv2_200.*  tf_efficientnetv2_b3.*; cd ..
    BACK=$2 FP=$3 THREADS=$4 MODEL=ALL make run-tnn-perf 2>/dev/null
    echo " "
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

CPU_testsuite 1
CPU_testsuite 4