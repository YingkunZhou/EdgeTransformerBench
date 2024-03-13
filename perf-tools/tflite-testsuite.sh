# USAGE: taskset -c 4-7 bash perf-tools/tflite-testsuite.sh
download_model()
{
    if [ ! -d ".tflite" ]
    then
        [ ! -f "tflite-fp32-models.tar.gz" ]   && gdown 1GIC-Q5OMuNZEh5wWM2qvZriboL_Xiov-
        [ ! -f "tflite-onnx-models.tar.gz" ]   && gdown 18ZJBnwhBLac9mprisHIM_9D2w0mSXHog
        [ ! -f "tflite-tinynn-models.tar.gz" ] && gdown 1q5gdahzHoBQSSGXEKjPsybFgHg-yOSbR
        tar xf tflite-fp32-models.tar.gz; \
        tar xf tflite-onnx-models.tar.gz; \
        tar xf tflite-tinynn-models.tar.gz
    fi
}

download_library()
{
    cd .libs
    if [ ! -d "tensorflow" ]
    then
        [ ! -f "tensorflow.tar.gz" ] && wget tensorflow.tar.gz
        tar xf tensorflow.tar.gz
        if uname -a | grep -q Android
        then
            echo "assert all android devices have armv8.2 fp16 isa and opencl support"
        elif clinfo >/dev/null
        then
            if cat /proc/cpuinfo | grep -q asimdhp
            then
                [ ! -f "armnn-v8.2-cl.tar.gz" ] && wget armnn-v8.2-cl.tar.gz
                tar xf armnn-v8.2-cl.tar.gz
            else
                [ ! -f "armnn-v8-cl.tar.gz" ] && wget armnn-v8-cl.tar.gz
                tar xf armnn-v8-cl.tar.gz
            fi
        else
            if cat /proc/cpuinfo | grep -q asimdhp
            then
                [ ! -f "armnn-v8.2.tar.gz" ] && wget armnn-v8.2.tar.gz
                tar xf armnn-v8.2.tar.gz
            else
                [ ! -f "armnn-v8.tar.gz" ] && wget armnn-v8.tar.gz
                tar xf armnn-v8.tar.gz
            fi
        fi
    fi
    cd ..
}

testsuite()
{
    cd .tflite; rm -rf *.tflite; ln -sf $1/*.tflite .; rm LeViT_256*  mobilevitv2_1[257]*  mobilevitv2_200*  tf_efficientnetv2_b3*; cd ..
    BACK=$2 FP=$3 THREADS=$4 MODEL=ALL make run-tflite-perf 2>/dev/null
    echo " "
}

# for tinynn dynamic int8 model
testsuite_mobilevitv2()
{
    cd .tflite; rm -rf *.tflite; ln -sf $1/efficientformerv2* .; ln -sf $1/SwiftFormer* .; ln -sf $1/EMO* .; ln -sf $1/edgenext* .; cd ..
    BACK=$2 FP=$3 THREADS=$4 MODEL=ALL make run-tflite-perf 2>/dev/null
    cd .tflite; rm -rf *.tflite; ln -sf $1/mobilevitv2_0* .; ln -sf $1/mobilevitv2_100* .; cd ..
    BACK=$2 FP=$3 THREADS=$4 MODEL=050 make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=075 make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=100 make run-tflite-perf 2>/dev/null
    cd .tflite; rm -rf *.tflite; ln -sf $1/*.tflite .; rm efficientformerv2* SwiftFormer* EMO* edgenext* mobilevitv2* LeViT_256* tf_efficientnetv2_b3*; cd ..
    BACK=$2 FP=$3 THREADS=$4 MODEL=ALL make run-tflite-perf 2>/dev/null
    echo " "
}

# for below 2GB memory device CPU armnn inference
testsuite_series()
{
    cd .tflite; rm -rf *.tflite; ln -sf $1/*.tflite .; rm LeViT_256* mobilevitv2_1[257]* mobilevitv2_200* tf_efficientnetv2_b3*; cd ..
    BACK=$2 FP=$3 THREADS=$4 MODEL=efficientformerv2 make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=SwiftFormer make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=EMO make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=edgenext make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=mobilevitv2 make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=mobilevit_ make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=LeViT make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=net make run-tflite-perf 2>/dev/null
    echo " "
}

# for below 2GB memory device GPU armnn inference
testsuite_onebyone()
{
    cd .tflite; rm -rf *.tflite; ln -sf $1/*.tflite .; rm LeViT_256* tf_efficientnetv2_b3*; cd ..
    BACK=$2 FP=$3 THREADS=$4 MODEL=efficientformerv2_s0 make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=efficientformerv2_s1 make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=efficientformerv2_s2 make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=SwiftFormer_XS make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=SwiftFormer_S  make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=SwiftFormer_L1 make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=EMO_1M make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=EMO_2M make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=EMO_5M make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=EMO_6M make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=edgenext_xx_small make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=edgenext_x_small  make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=edgenext_small    make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=mobilevitv2_050 make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=mobilevitv2_075 make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=mobilevitv2_100 make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=mobilevit_xx_small make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=mobilevit_x_small  make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=mobilevit_small    make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=LeViT_128 make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=LeViT_192 make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=resnet50 make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=mobilenetv3_large_100 make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=tf_efficientnetv2 make run-tflite-perf 2>/dev/null
    echo " "
}

CPU_testsuite()
{
    ### fp32
    echo ">>>>>>>>>>>xnnpack: tfconvert fp32/fp16 model + fp32 arith<<<<<<<<<"
    testsuite fp16 z 32 $1

    echo ">>>>>>>>>>>xnnpack: tinynn fp32 model + fp32 arith<<<<<<<<<"
    testsuite tinynn-32 z 32 $1

    ### fp16
    if cat /proc/cpuinfo | grep -q asimdhp
    then
        echo ">>>>>>>>>>>xnnpack: tfconvert fp32/fp16 model + fp16 arith<<<<<<<<<"
        testsuite fp16 x 16 $1

        echo ">>>>>>>>>>>xnnpack: tinynn fp32 model + fp16 arith<<<<<<<<<"
        testsuite tinynn-32 x 16 $1
    fi

    ### int8
    echo ">>>>>>>>>>>xnnpack: tfconvert PTQ static int8 model<<<<<<<<<"
    testsuite int8 z 32 $1

    echo ">>>>>>>>>>>xnnpack: tinynn dynamic int8 model<<<<<<<<<"
    testsuite_mobilevitv2 tinynn-d8 z 32 $1

    ##### armnn performance is not good on CPU
    # echo ">>>>>>>>>>>armnn CPU: tfconvert fp32 model + fp32 arith<<<<<<<<<"
    # testsuite_series fp32 a 32 $1

    # echo ">>>>>>>>>>>armnn CPU: tinynn fp32 model + fp32 arith<<<<<<<<<"
    # testsuite_series tinynn-32 a 32 $1

    # if cat /proc/cpuinfo | grep -q asimdhp
    # then
    #     echo ">>>>>>>>>>>armnn CPU: tfconvert fp32 model + fp16 arith<<<<<<<<<"
    #     testsuite_series fp32 a 16 $1

    #     echo ">>>>>>>>>>>armnn CPU: tinynn fp32 model + fp16 arith<<<<<<<<<"
    #     testsuite_series tinynn-32 a 16 $1
    # fi

    # echo ">>>>>>>>>>>armnn CPU: tfconvert ptq static int8 model<<<<<<<<<"
    # testsuite int8 a 32 $1
}

GPU_testsuite()
{
    ### fp32
    echo ">>>>>>>>>>>gpu: tfconvert fp32/fp16 model + fp32 arith<<<<<<<<<"
    testsuite fp16 g 32 1

    echo ">>>>>>>>>>>gpu: tinynn fp32 model + fp32 arith<<<<<<<<<"
    testsuite_series tinynn-32 g 32 1
    if vulkaninfo | grep -q Adreno
    then
        echo "armnn GPU backend is not suitable for Qualcomm Adreno GPU"
    else
        echo ">>>>>>>>>>>armnn GPU: tfconvert fp32 model + fp32 arith<<<<<<<<<"
        testsuite_onebyone fp32 m 32 1

        echo ">>>>>>>>>>>armnn GPU: tinynn fp32 model + fp32 arith<<<<<<<<<"
        testsuite_onebyone tinynn-32 m 32 1
    fi

    ### fp16
    # make sure all opencl/gpu support fp16
    echo ">>>>>>>>>>>gpu: tfconvert fp32/fp16 model + fp16 arith<<<<<<<<<"
    testsuite fp16 g 16 1

    echo ">>>>>>>>>>>gpu: tinynn fp32 model + fp16 arith<<<<<<<<<"
    testsuite_series tinynn-32 g 16 1

    if vulkaninfo | grep -q Adreno
    then
        echo "armnn GPU backend is not suitable for Qualcomm Adreno GPU"
    else
        echo ">>>>>>>>>>>armnn GPU: tfconvert fp32 model + fp16 arith<<<<<<<<<"
        testsuite_onebyone fp32 m 16 1

        echo ">>>>>>>>>>>armnn GPU: tinynn fp32 model + fp16 arith<<<<<<<<<"
        testsuite_onebyone tinynn-32 m 16 1
    fi

    ### int8
    echo ">>>>>>>>>>>gpu: tfconvert PTQ static int8 model<<<<<<<<<"
    testsuite int8 g 16 1

    echo ">>>>>>>>>>>gpu: tinynn dynamic int8 model<<<<<<<<<"
    testsuite_mobilevitv2 tinynn-d8 g 16 1

    if vulkaninfo | grep -q Adreno
    then
        echo "armnn GPU backend is not suitable for Qualcomm Adreno GPU"
    else
        echo ">>>>>>>>>>>armnn GPU: tfconvert ptq static int8 model<<<<<<<<<"
        testsuite_series int8 m 32 1
    fi

    echo ">>>>>>>>>>>gpu: tfconvert dynamic int8 model<<<<<<<<<"
    testsuite dynamic g 16 1
}

NNAPI_testsuite()
{
    echo ">>>>>>>>>>>nnapi: tfconvert fp32/fp16 model<<<<<<<<<"
    testsuite fp16 n 32 1

    echo ">>>>>>>>>>>nnapi: tinynn fp32 model<<<<<<<<<"
    testsuite tinynn-32 n 32 1

    echo ">>>>>>>>>>>nnapi: tfconvert dynamic int8 model<<<<<<<<<"
    testsuite dynamic n 32 1

    echo ">>>>>>>>>>>nnapi: tfconvert PTQ static int8 model<<<<<<<<<"
    testsuite int8 n 32 1

    echo ">>>>>>>>>>>nnapi: tinynn dynamic int8 model<<<<<<<<<"
    testsuite_mobilevitv2 tinynn-d8 n 32 1
}

download_model
download_library

if uname -a | grep -q Android
then
    NNAPI_testsuite
fi

if clinfo >/dev/null
then
    GPU_testsuite
fi

CPU_testsuite 1
CPU_testsuite 4
