# export NCNN_LIB=$HOME/work/ncnn/install-vulkan/lib
# export MNN_LIB=$HOME/work/MNN/install/lib
# export TNN_LIB=$HOME/work/TNN/install/lib
# export PDLITE_LIB=$HOME/work/Paddle-Lite/build.lite.android.armv8.clang/inference_lite_lib.android.armv8.opencl/cxx/lib
# export TFLITE_LIB=$HOME/work/tensorflow/lib
# export ONNXRT_LIB=$HOME/work/onnxruntime/lib
# export LD_LIBRARY_PATH=$NCNN_LIB:$MNN_LIB:$TNN_LIB:$PDLITE_LIB:$TFLITE_LIB:$ONNXRT_LIB

all: ncnn-perf mnn-perf tnn-perf pdlite-perf tflite-perf

NCNN_LIB ?= $(HOME)/work/ncnn/install-vulkan/lib
NCNN_INC ?= $(HOME)/work/ncnn/install-vulkan/include/ncnn
FLAGS =  utils.cpp  -std=c++17 `pkg-config --cflags --libs opencv4`

ncnn-perf:
	g++ -O3 -o ncnn-perf ncnn_perf.cpp -I$(NCNN_INC) -L$(NCNN_LIB) -lncnn $(FLAGS)

ncnn-perf-test:
	g++ -O3 -DTEST -o ncnn-perf-test ncnn_perf.cpp -I$(NCNN_INC) -L$(NCNN_LIB) -lncnn $(FLAGS)

MNN_LIB ?= $(HOME)/work/MNN/install/lib
MNN_INC ?= $(HOME)/work/MNN/install/include

mnn-perf:
	g++ -O3 -o mnn-perf mnn_perf.cpp -I$(MNN_INC) -L$(MNN_LIB) -lMNN $(FLAGS)

mnn-perf-test:
	g++ -O3 -DTEST -o mnn-perf-test mnn_perf.cpp -I$(MNN_INC) -L$(MNN_LIB) -lMNN $(FLAGS)

TNN_LIB ?= $(HOME)/work/TNN/install/lib
TNN_INC ?= $(HOME)/work/TNN/install/include

tnn-perf:
	g++ -O3 -o tnn-perf tnn_perf.cpp -I$(TNN_INC) -L$(TNN_LIB) -lTNN $(FLAGS)

tnn-perf-test:
	g++ -O3 -DTEST -o tnn-perf-test tnn_perf.cpp -I$(TNN_INC) -L$(TNN_LIB) -lTNN $(FLAGS)

PDLITE_LIB ?= $(HOME)/work/Paddle-Lite/build.lite.android.armv8.clang/inference_lite_lib.android.armv8.opencl/cxx/lib
PDLITE_INC ?= $(HOME)/work/Paddle-Lite/build.lite.android.armv8.clang/inference_lite_lib.android.armv8.opencl/cxx/include

pdlite-perf:
	g++ -O3 -o pdlite-perf pdlite_perf.cpp -I$(PDLITE_INC)  -L$(PDLITE_LIB) -lpaddle_light_api_shared $(FLAGS)

TFLITE_INC ?= $(HOME)/work/tensorflow/include
TFLITE_LIB ?= $(HOME)/work/tensorflow/lib
ARMNN_FLAGS = -I$(TFLITE_INC)/armnn/delegate/classic/include -I$(TFLITE_INC)/armnn/delegate/common/include -I$(TFLITE_INC)/armnn/include

tflite-perf:
	g++ -O3 -o tflite-perf tflite_perf.cpp -I$(TFLITE_INC) -L$(TFLITE_LIB) $(FLAGS) -ltensorflowlite \
	#-ltensorflowlite_gpu_delegate -DGPU \
	#-lnnapi_util -lnnapi_delegate_no_nnapi_implementation -lnnapi_implementation -DNNAPI \
	#$(ARMNN_FLAGS) -larmnnDelegate -larmnn -DARMNN

tflite-perf-test:
	g++ -O3 -DTEST -o tflite-perf-test tflite_perf.cpp -I$(TFLITE_INC) -L$(TFLITE_LIB) $(FLAGS) -ltensorflowlite \
	#-ltensorflowlite_gpu_delegate -DGPU \
	#-lnnapi_util -lnnapi_delegate_no_nnapi_implementation -lnnapi_implementation -DNNAPI \
	#$(ARMNN_FLAGS) -larmnnDelegate -larmnn -DARMNN

ONNXRT_INC ?= $(HOME)/work/onnxruntime/include
ONNXRT_LIB ?= $(HOME)/work/onnxruntime/lib

onnxruntime-perf:
	g++ -O3 -o onnxruntime-perf onnxruntime_perf.cpp -I$(ONNXRT_INC)  -L$(ONNXRT_LIB) $(FLAGS) -lonnxruntime