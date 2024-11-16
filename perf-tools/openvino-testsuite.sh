# USAGE: bash perf-tools/mnn-testsuite.sh
download_model()
{
    if [ ! -d ".xml" ]
    then
        [ ! -f "openvino-models.tar.gz" ] && gdown 1Mhc4g5zSww1NRKIPUjWwO_PNVhYRuNPC
        tar xf openvino-models.tar.gz;
    fi
    # gdown 1XqF9my9TguiKKlaKxvfvGAXTDZh6HUSG # downloading mnn-fp32-models.tar.gz
}

testsuite()
{
    cd .xml; rm *; ln -sf $1/* .; rm LeViT_256.*  mobilevitv2_1[257]*  mobilevitv2_200.*  tf_efficientnetv2_b3.*; cd ..
    BACK=$2 FP=$3 THREADS=$4 MODEL=ALL make run-openvino-perf 2>/dev/null
    echo " "
}

testsuite_series()
{
    cd .xml; rm *; ln -sf $1/* .; rm LeViT_256.*  mobilevitv2_1[257]*  mobilevitv2_200.*  tf_efficientnetv2_b3.*; cd ..
    BACK=$2 FP=$3 THREADS=$4 MODEL=efficientformerv2 make run-openvino-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=SwiftFormer make run-openvino-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=EMO make run-openvino-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=edgenext make run-openvino-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=mobilevitv2 make run-openvino-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=mobilevit_ make run-openvino-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=LeViT make run-openvino-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=net make run-openvino-perf 2>/dev/null
    echo " "
}

NPU_testsuite()
{
    echo ">>>>>>>>>>>NPU: fp16 model + fp32? arith<<<<<<<<<"
    testsuite fp16 n 32 1

    echo ">>>>>>>>>>>NPU: fp16 model + fp16 arith<<<<<<<<<"
    testsuite fp16 n 16 1

    echo ">>>>>>>>>>>NPU: fp16 model + int8 arith<<<<<<<<<"
    testsuite_series int8 n 16 1
}

GPU_testsuite()
{
    echo ">>>>>>>>>>>GPU: fp16 model + fp32 arith<<<<<<<<<"
    testsuite fp16 g 32 1

    echo ">>>>>>>>>>>GPU: fp16 model + fp16 arith<<<<<<<<<"
    testsuite fp16 g 16 1

    echo ">>>>>>>>>>>GPU: fp16 model + int8 arith<<<<<<<<<"
    testsuite_series int8 g 16 1
}

CPU_testsuite()
{
    echo ">>>>>>>>>>>CPU: fp16 model + fp32 arith<<<<<<<<<"
    testsuite fp16 z 32 $1

    # echo ">>>>>>>>>>>cpu: fp16 model + fp16 arith<<<<<<<<<"
    # testsuite fp16 z 16 $1

    echo ">>>>>>>>>>>CPU: int8 model<<<<<<<<<"
    testsuite int8 z 16 $1
}

download_model

NPU_testsuite
GPU_testsuite
CPU_testsuite 1
CPU_testsuite 4
