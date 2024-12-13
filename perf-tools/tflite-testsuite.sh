# USAGE: taskset -c 4-7 bash perf-tools/tflite-testsuite.sh
download_model()
{
    if [ ! -d ".tflite" ]
    then
        # [ ! -f "tflite-onnx-models.tar.gz" ]   && gdown 18ZJBnwhBLac9mprisHIM_9D2w0mSXHog
        [ ! -f "tflite-litert-models.tar.gz" ] && gdown 14NTdkYJsT7YOtfA7OZXeEf2DoUmFq0rw
        [ ! -f "tflite-tinynn-models.tar.gz" ] && gdown 1q5gdahzHoBQSSGXEKjPsybFgHg-yOSbR
        # tar xf tflite-onnx-models.tar.gz; \
        tar xf tflite-litert-models.tar.gz; \
        tar xf tflite-tinynn-models.tar.gz
        if vulkaninfo | grep -q Mali
        then
            [ ! -f "tflite-fp32-models.tar.gz" ]   && gdown 1GIC-Q5OMuNZEh5wWM2qvZriboL_Xiov-
            tar xf tflite-fp32-models.tar.gz;
        fi

    fi
}

download_library()
{
    cd .libs
    if [ ! -d "tensorflow" ]
    then
        if [ ! -f "tensorflow.tar.gz" ]
        then
            if uname -a | grep -q Android
            then
                wget https://github.com/YingkunZhou/EdgeTransformerBench/releases/download/v1.1/tensorflow.tar.gz
            elif uname -m | grep -q x86_64
            then
                wget https://github.com/YingkunZhou/EdgeTransformerBench/releases/download/v2.1/tensorflow-2.16.1.tar.gz
            else
                wget https://github.com/YingkunZhou/EdgeTransformerBench/releases/download/v1.0/tensorflow.tar.gz
            fi
        fi
        if uname -m | grep -q x86_64
        then
            tar xf tensorflow-2.16.1.tar.gz
            mv tensorflow-2.16.1 tensorflow
        else
            tar xf tensorflow.tar.gz
        fi
        if uname -a | grep -q Android
        then
            echo "assert all android devices have armv8.2 fp16 isa and opencl support"
        elif clinfo >/dev/null
        then
            if cat /proc/cpuinfo | grep -q asimdhp
            then
                [ ! -f "armnn-v8.2-cl.tar.gz" ] && wget https://github.com/YingkunZhou/EdgeTransformerBench/releases/download/v1.0/armnn-v8.2-cl.tar.gz
                tar xf armnn-v8.2-cl.tar.gz
            else
                [ ! -f "armnn-v8-cl.tar.gz" ] && wget https://github.com/YingkunZhou/EdgeTransformerBench/releases/download/v1.0/armnn-v8-cl.tar.gz
                tar xf armnn-v8-cl.tar.gz
            fi
        else
            if cat /proc/cpuinfo | grep -q asimdhp
            then
                [ ! -f "armnn-v8.2.tar.gz" ] && wget https://github.com/YingkunZhou/EdgeTransformerBench/releases/download/v1.0/armnn-v8.2.tar.gz
                tar xf armnn-v8.2.tar.gz
            else
                [ ! -f "armnn-v8.tar.gz" ] && wget https://github.com/YingkunZhou/EdgeTransformerBench/releases/download/v1.0/armnn-v8.tar.gz
                tar xf armnn-v8.tar.gz
            fi
        fi
    fi
    cd ..
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

CPU_testsuite()
{
    ### fp32
    echo ">>>>>>>>>>>xnnpack: litert fp32/fp16 model + fp32 arith<<<<<<<<<"
    testsuite_series litert z 32 $1

    echo ">>>>>>>>>>>xnnpack: tinynn fp32 model + fp32 arith<<<<<<<<<"
    testsuite_series tinynn-32 z 32 $1

    if uname -m | grep -q aarch64
    then
        echo ">>>>>>>>>>>xnnpack: litert fp32/fp16 model + fp32 arith<<<<<<<<<"
        testsuite_series litert a 32 $1

        echo ">>>>>>>>>>>xnnpack: tinynn fp32 model + fp32 arith<<<<<<<<<"
        testsuite_series tinynn-32 a 32 $1
    fi

    ### fp16
    if cat /proc/cpuinfo | grep -q asimdhp
    then
        echo ">>>>>>>>>>>xnnpack: litert fp32/fp16 model + fp16 arith<<<<<<<<<"
        testsuite_series litert x 16 $1

        echo ">>>>>>>>>>>xnnpack: tinynn fp32 model + fp16 arith<<<<<<<<<"
        testsuite_series tinynn-32 x 16 $1

        echo ">>>>>>>>>>>xnnpack: litert fp32/fp16 model + fp16 arith<<<<<<<<<"
        testsuite_series litert a 16 $1

        echo ">>>>>>>>>>>xnnpack: tinynn fp32 model + fp16 arith<<<<<<<<<"
        testsuite_series tinynn-32 a 16 $1
    fi

    echo ">>>>>>>>>>>xnnpack: tinynn dynamic int8 model<<<<<<<<<"
    # only run on xnnpack
    testsuite_mobilevitv2 tinynn-d8 z 32 $1
}

GPU_testsuite()
{
    ### fp32
    echo ">>>>>>>>>>>gpu: litert fp32/fp16 model + fp32 arith<<<<<<<<<"
    testsuite_series litert g 32 1

    echo ">>>>>>>>>>>gpu: tinynn fp32 model + fp32 arith<<<<<<<<<"
    testsuite_series tinynn-32 g 32 1

    if vulkaninfo | grep -q Adreno
    then
        echo "armnn GPU backend is not suitable for Qualcomm Adreno GPU"
    else
        echo ">>>>>>>>>>>armnn GPU: litert fp32 model + fp32 arith<<<<<<<<<"
        testsuite_series litert m 32 1

        echo ">>>>>>>>>>>armnn GPU: tinynn fp32 model + fp32 arith<<<<<<<<<"
        testsuite_series tinynn-32 m 32 1
    fi

    ### fp16
    # make sure all opencl/gpu support fp16
    echo ">>>>>>>>>>>gpu: litert fp32/fp16 model + fp16 arith<<<<<<<<<"
    testsuite_series litert g 16 1

    echo ">>>>>>>>>>>gpu: tinynn fp32 model + fp16 arith<<<<<<<<<"
    testsuite_series tinynn-32 g 16 1

    if vulkaninfo | grep -q Adreno
    then
        echo "armnn GPU backend is not suitable for Qualcomm Adreno GPU"
    else
        echo ">>>>>>>>>>>armnn GPU: litert fp32 model + fp16 arith<<<<<<<<<"
        testsuite_series litert m 16 1

        echo ">>>>>>>>>>>armnn GPU: tinynn fp32 model + fp16 arith<<<<<<<<<"
        testsuite_series tinynn-32 m 16 1
    fi

    ### int8
    echo ">>>>>>>>>>>gpu: tinynn dynamic int8 model<<<<<<<<<"
    testsuite_mobilevitv2 tinynn-d8 g 16 1
}

NNAPI_testsuite()
{
    echo ">>>>>>>>>>>nnapi: litert fp32 model<<<<<<<<<"
    testsuite_series litert n 32 1

    echo ">>>>>>>>>>>nnapi: tinynn fp32 model<<<<<<<<<"
    testsuite_series tinynn-32 n 32 1

    echo ">>>>>>>>>>>nnapi: tinynn dynamic int8 model<<<<<<<<<"
    testsuite_mobilevitv2 tinynn-d8 n 32 1
}

download_model
download_library

if uname -m | grep -q x86_64
then
    echo "skip vulkan & opencl & nnapi testsuite in x86 platform"
else
    if vulkaninfo | grep -q Adreno
    then
        cp /vendor/lib64/libOpenCL.so .libs/tensorflow/install/lib
    fi

    if uname -a | grep -q Android
    then
        NNAPI_testsuite
    fi

    if clinfo >/dev/null
    then
        GPU_testsuite
    fi
fi

CPU_testsuite 1
CPU_testsuite 4
