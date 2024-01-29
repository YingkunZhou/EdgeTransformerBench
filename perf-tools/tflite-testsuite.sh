download_model()
{
    if [ ! -d ".tflite" ]
    then
        [ -f "tflite-fp32-models.tar.gz" ] && wget tflite-fp32-models.tar.gz
        [ -f "tflite-onnx-models.tar.gz" ] && wget tflite-onnx-models.tar.gz
        [ -f "tflite-tinynn-models.tar.gz" ] && wget tflite-tinynn-models.tar.gz
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
        [ -f "tensorflow.tar.gz"] && wget tensorflow.tar.gz
        tar xf tensorflow.tar.gz
        if clinfo
        then
            if cat /proc/cpuinfo | grep asimdhp
            then
                [ -f "armnn-v8.2-cl.tar.gz"] && wget armnn-v8.2-cl.tar.gz
                tar xf armnn-v8.2-cl.tar.gz
            else
                [ -f "armnn-v8-cl.tar.gz"] && wget armnn-v8-cl.tar.gz
                tar xf armnn-v8-cl.tar.gz
            fi
        else
            if cat /proc/cpuinfo | grep asimdhp
            then
                [ -f "armnn-v8.2.tar.gz"] && wget armnn-v8.2.tar.gz
                tar xf armnn-v8.2.tar.gz
            else
                [ -f "armnn-v8.tar.gz"] && wget armnn-v8.tar.gz
                tar xf armnn-v8.tar.gz
            fi
        fi
    fi
    cd ..
}

testsuite()
{
    cd .tflite; rm -rf *.tflite; ln -sf $1/* .; cd ..
    BACK=$2 FP=$3 THREADS=$4 MODEL=ALL make run-tflite-perf 2>/dev/null
    echo " "
}

testsuite_mobilevitv2()
{
    cd .tflite; rm -rf *.tflite; ln -sf $1/* .; cd ..
    BACK=$2 FP=$3 THREADS=$4 MODEL=orm make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=EMO make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=sma make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=050 make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=075 make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=100 make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=125 make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=150 make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=175 make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=200 make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=LeV make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=res make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=tf_ make run-tflite-perf 2>/dev/null
    echo " "
}

# under 2GB memory
testsuite_onebyone()
{
    cd .tflite; rm -rf *.tflite; ln -sf $1/* .; cd ..
    BACK=$2 FP=$3 THREADS=$4 MODEL=s0 make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=s1 make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=s2 make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=XS make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=_S make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=L1 make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=1M make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=2M make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=5M make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=6M make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=ed make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=050 make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=075 make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=100 make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=125 make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=150 make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=175 make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=200 make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=it_x make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=it_x make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=it_s make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=128 make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=196 make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=256 make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=res make run-tflite-perf 2>/dev/null
    BACK=$2 FP=$3 THREADS=$4 MODEL=tf_ make run-tflite-perf 2>/dev/null
    echo " "
}

CPU_testsuite()
{
    ### fp32
    echo ">>>>>>>>>>>xnnpack: tfconvert fp32/fp16 model + fp32 arith<<<<<<<<<"
    testsuite fp16 z 32 $1

    echo ">>>>>>>>>>>xnnpack: tinynn fp32 model + fp32 arith<<<<<<<<<"
    testsuite tinynn-32 z 32 $1

    echo ">>>>>>>>>>>armnn CPU: tfconvert fp32 model + fp32 arith<<<<<<<<<"
    testsuite_onebyone fp32 a 32 $1

    echo ">>>>>>>>>>>armnn CPU: tinynn fp32 model + fp32 arith<<<<<<<<<"
    testsuite_onebyone tinynn-32 a 32 $1

    ### fp16
    if cat /proc/cpuinfo | grep asimdhp
    then
        echo ">>>>>>>>>>>xnnpack: tfconvert fp32/fp16 model + fp16 arith<<<<<<<<<"
        testsuite fp16 x 16 $1

        echo ">>>>>>>>>>>xnnpack: tinynn fp32 model + fp16 arith<<<<<<<<<"
        testsuite tinynn-32 x 16 $1

        echo ">>>>>>>>>>>armnn CPU: tfconvert fp32 model + fp16 arith<<<<<<<<<"
        testsuite_onebyone fp32 a 16 $1

        echo ">>>>>>>>>>>armnn CPU: tinynn fp32 model + fp16 arith<<<<<<<<<"
        testsuite_onebyone tinynn-32 a 16 $1
    fi

    ### int8
    echo ">>>>>>>>>>>xnnpack: tfconvert PTQ static int8 model<<<<<<<<<"
    testsuite int8 z 32 $1

    echo ">>>>>>>>>>>xnnpack: tinynn dynamic int8 model<<<<<<<<<"
    testsuite_mobilevitv2 tinynn-d8 z 32 $1

    echo ">>>>>>>>>>>armnn CPU: tfconvert ptq static int8 model<<<<<<<<<"
    testsuite int8 a 32 $1
}

GPU_testsuite()
{
    ### fp32
    echo ">>>>>>>>>>>gpu: tfconvert fp32/fp16 model + fp32 arith<<<<<<<<<"
    testsuite fp16 g 32 1

    echo ">>>>>>>>>>>gpu: tinynn fp32 model + fp32 arith<<<<<<<<<"
    testsuite tinynn-32 g 32 1

    echo ">>>>>>>>>>>armnn GPU: tfconvert fp32 model + fp32 arith<<<<<<<<<"
    testsuite_onebyone fp32 m 32 1

    echo ">>>>>>>>>>>armnn GPU: tinynn fp32 model + fp32 arith<<<<<<<<<"
    testsuite_onebyone tinynn-32 m 32 1

    ### fp16
    # make sure all opencl/gpu support fp16
    echo ">>>>>>>>>>>gpu: tfconvert fp32/fp16 model + fp16 arith<<<<<<<<<"
    testsuite fp16 g 16 1

    echo ">>>>>>>>>>>gpu: tinynn fp32 model + fp16 arith<<<<<<<<<"
    testsuite tinynn-32 g 16 1

    echo ">>>>>>>>>>>armnn GPU: tfconvert fp32 model + fp16 arith<<<<<<<<<"
    testsuite_onebyone fp32 m 16 1

    echo ">>>>>>>>>>>armnn GPU: tinynn fp32 model + fp16 arith<<<<<<<<<"
    testsuite_onebyone tinynn-32 m 16 1

    ### int8
    #TODO: use 16 is better or not?
    echo ">>>>>>>>>>>gpu: tfconvert dynamic int8 model<<<<<<<<<"
    testsuite dynamic g 32 1

    echo ">>>>>>>>>>>gpu: tfconvert PTQ static int8 model<<<<<<<<<"
    testsuite int8 g 32 1

    echo ">>>>>>>>>>>gpu: tinynn dynamic int8 model<<<<<<<<<"
    testsuite_mobilevitv2 tinynn-d8 g 32 1

    echo ">>>>>>>>>>>armnn CPU: tfconvert ptq static int8 model<<<<<<<<<"
    testsuite int8 m 32 1
}

NNAPI_testsuite()
{
    exit
}

download_model
download_library

if pwd | grep termux
then
    NNAPI_testsuite
fi

if clinfo
then
    GPU_testsuite
fi

CPU_testsuite 1
CPU_testsuite 4
