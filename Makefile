all: ncnn-perf mnn-perf tnn-perf pdlite-perf tflite-perf onnxruntime-perf

MODEL ?= s1
# make sure no real backend use "z", so that can fall back to CPU exection
BACK ?= z

init:
	./prepare.sh

########################
###### ncnn part #######
########################
NCNN_LIB ?= $(PWD)/.libs/ncnn/install-vulkan/lib
NCNN_INC ?= $(PWD)/.libs/ncnn/install-vulkan/include/ncnn
FLAGS =  src/utils.cpp  -std=c++17 `pkg-config --cflags --libs opencv4`

ncnn-perf: bin/ncnn-perf
ncnn-perf-test: bin/ncnn-perf-test

bin/ncnn-perf: src/ncnn_perf.cpp src/utils.cpp
	g++ -O3 -o bin/ncnn-perf src/ncnn_perf.cpp -I$(NCNN_INC) -L$(NCNN_LIB) -lncnn $(FLAGS)

bin/ncnn-perf-test: src/ncnn_perf.cpp src/utils.cpp
	g++ -O3 -DTEST -o bin/ncnn-perf-test src/ncnn_perf.cpp -I$(NCNN_INC) -L$(NCNN_LIB) -lncnn $(FLAGS)

run-ncnn-perf: bin/ncnn-perf
	LD_LIBRARY_PATH=$(NCNN_LIB) bin/ncnn-perf --only-test $(MODEL) --backend $(BACK)

validation-ncnn: bin/ncnn-perf
	LD_LIBRARY_PATH=$(NCNN_LIB) bin/ncnn-perf --only-test $(MODEL) --backend $(BACK) --validation

test-ncnn-perf: bin/ncnn-perf-test
	LD_LIBRARY_PATH=$(NCNN_LIB) bin/ncnn-perf-test --only-test $(MODEL) --backend $(BACK)

########################
###### mnn part ########
########################
MNN_LIB ?= $(PWD)/.libs/MNN/install/lib
MNN_INC ?= $(PWD)/.libs/MNN/install/include

mnn-perf: bin/mnn-perf
mnn-perf-test: bin/mnn-perf-test

bin/mnn-perf: src/mnn_perf.cpp src/utils.cpp
	g++ -O3 -o bin/mnn-perf src/mnn_perf.cpp -I$(MNN_INC) -L$(MNN_LIB) -lMNN $(FLAGS)

bin/mnn-perf-test: src/mnn_perf.cpp src/utils.cpp
	g++ -O3 -DTEST -o bin/mnn-perf-test src/mnn_perf.cpp -I$(MNN_INC) -L$(MNN_LIB) -lMNN $(FLAGS)

run-mnn-perf: bin/mnn-perf
	LD_LIBRARY_PATH=$(MNN_LIB) bin/mnn-perf --only-test $(MODEL) --backend $(BACK)

validation-mnn: bin/mnn-perf
	LD_LIBRARY_PATH=$(MNN_LIB) bin/mnn-perf --only-test $(MODEL) --backend $(BACK) --validation

test-mnn-perf: bin/mnn-perf-test
	LD_LIBRARY_PATH=$(MNN_LIB) bin/mnn-perf-test --only-test $(MODEL) --backend $(BACK)

########################
###### tnn part ########
########################
TNN_LIB ?= $(PWD)/.libs/TNN/install/lib
TNN_INC ?= $(PWD)/.libs/TNN/install/include

tnn-perf: bin/tnn-perf
tnn-perf-test: bin/tnn-perf-test

bin/tnn-perf: src/tnn_perf.cpp src/utils.cpp
	g++ -O3 -o bin/tnn-perf src/tnn_perf.cpp -I$(TNN_INC) -L$(TNN_LIB) -lTNN $(FLAGS)

bin/tnn-perf-test: src/tnn_perf.cpp src/utils.cpp
	g++ -O3 -DTEST -o bin/tnn-perf-test src/tnn_perf.cpp -I$(TNN_INC) -L$(TNN_LIB) -lTNN $(FLAGS)

run-tnn-perf: bin/tnn-perf
	LD_LIBRARY_PATH=$(TNN_LIB) bin/tnn-perf --only-test $(MODEL) --backend $(BACK)

validation-tnn: bin/tnn-perf
	LD_LIBRARY_PATH=$(TNN_LIB) bin/tnn-perf --only-test $(MODEL) --backend $(BACK) --validation

test-tnn-perf: bin/tnn-perf-test
	LD_LIBRARY_PATH=$(TNN_LIB) bin/tnn-perf-test --only-test $(MODEL) --backend $(BACK)

########################
##### pdlite part ######
########################
PDLITE_LIB ?= $(PWD)/.libs/Paddle-Lite/lib
PDLITE_INC ?= $(PWD)/.libs/Paddle-Lite/include

pdlite-perf: bin/pdlite-perf
pdlite-perf-test: bin/pdlite-perf-test

bin/pdlite-perf: src/pdlite_perf.cpp src/utils.cpp
	g++ -O3 -o bin/pdlite-perf src/pdlite_perf.cpp -I$(PDLITE_INC)  -L$(PDLITE_LIB) -lpaddle_light_api_shared $(FLAGS)

bin/pdlite-perf-test: src/pdlite_perf.cpp src/utils.cpp
	g++ -O3 -DTEST -o bin/pdlite-perf-test src/pdlite_perf.cpp -I$(PDLITE_INC) -L$(PDLITE_LIB) -lpaddle_light_api_shared $(FLAGS)

run-pdlite-perf: bin/pdlite-perf
	LD_LIBRARY_PATH=$(PDLITE_LIB) bin/pdlite-perf --only-test $(MODEL) --backend $(BACK)

validation-pdlite: bin/pdlite-perf
	LD_LIBRARY_PATH=$(PDLITE_LIB) bin/pdlite-perf --only-test $(MODEL) --backend $(BACK) --validation

test-pdlite-perf: bin/pdlite-perf-test
	LD_LIBRARY_PATH=$(PDLITE_LIB) bin/pdlite-perf-test --only-test $(MODEL) --backend $(BACK)

########################
##### tflite part ######
########################
TFLITE_INC ?= $(PWD)/.libs/tensorflow/include
TFLITE_LIB ?= $(PWD)/.libs/tensorflow/lib
ARMNN_FLAGS = -I$(TFLITE_INC)/armnn/delegate/classic/include -I$(TFLITE_INC)/armnn/delegate/common/include -I$(TFLITE_INC)/armnn/include

tflite-perf: bin/tflite-perf
tflite-perf-test: bin/tflite-perf-test

bin/tflite-perf: src/tflite_perf.cpp src/utils.cpp
	g++ -O3 -o bin/tflite-perf src/tflite_perf.cpp -I$(TFLITE_INC) -L$(TFLITE_LIB) $(FLAGS) -ltensorflowlite \
	#-ltensorflowlite_gpu_delegate -DUSE_GPU \
	#-lnnapi_util -lnnapi_delegate_no_nnapi_implementation -lnnapi_implementation -DUSE_NNAPI \
	#$(ARMNN_FLAGS) -larmnnDelegate -larmnn -DUSE_ARMNN

bin/tflite-perf-test: src/tflite_perf.cpp src/utils.cpp
	g++ -O3 -DTEST -o bin/tflite-perf-test src/tflite_perf.cpp -I$(TFLITE_INC) -L$(TFLITE_LIB) $(FLAGS) -ltensorflowlite \
	#-ltensorflowlite_gpu_delegate -DUSE_GPU \
	#-lnnapi_util -lnnapi_delegate_no_nnapi_implementation -lnnapi_implementation -DUSE_NNAPI \
	#$(ARMNN_FLAGS) -larmnnDelegate -larmnn -DUSE_ARMNN

run-tflite-perf: bin/tflite-perf
	LD_LIBRARY_PATH=$(TFLITE_LIB) bin/tflite-perf --only-test $(MODEL) --backend $(BACK)

validation-tflite: bin/tflite-perf
	LD_LIBRARY_PATH=$(TFLITE_LIB) bin/tflite-perf --only-test $(MODEL) --backend $(BACK) --validation

test-tflite-perf: bin/tflite-perf-test
	LD_LIBRARY_PATH=$(TFLITE_LIB) bin/tflite-perf-test --only-test $(MODEL) --backend $(BACK)

########################
### onnxruntime part ###
########################
ONNXRT_INC ?= $(PWD)/.libs/onnxruntime/include
ONNXRT_LIB ?= $(PWD)/.libs/onnxruntime/lib
MORE_FLAGS ?=

tflite-perf: bin/tflite-perf
tflite-perf-test: bin/tflite-perf-test

bin/onnxruntime-perf: src/onnxruntime_perf.cpp src/utils.cpp
	g++ -O3 -o bin/onnxruntime-perf src/onnxruntime_perf.cpp -I$(ONNXRT_INC)  -L$(ONNXRT_LIB) $(FLAGS) -lonnxruntime $(MORE_FLAGS)

bin/onnxruntime-perf-test: src/onnxruntime_perf.cpp src/utils.cpp
	g++ -O3 -DTEST -o bin/onnxruntime-perf-test src/onnxruntime_perf.cpp -I$(ONNXRT_INC)  -L$(ONNXRT_LIB) $(FLAGS) -lonnxruntime $(MORE_FLAGS)

run-onnxruntime-perf: bin/onnxruntime-perf
	LD_LIBRARY_PATH=$(ONNXRT_LIB):$(LD_LIBRARY_PATH) bin/onnxruntime-perf --only-test $(MODEL) --backend $(BACK)

validation-onnxruntime: bin/onnxruntime-perf
	LD_LIBRARY_PATH=$(ONNXRT_LIB):$(LD_LIBRARY_PATH) bin/onnxruntime-perf --only-test $(MODEL) --backend $(BACK) --validation

test-onnxruntime-perf: bin/onnxruntime-perf-test
	LD_LIBRARY_PATH=$(ONNXRT_LIB):$(LD_LIBRARY_PATH) bin/onnxruntime-perf-test --only-test $(MODEL) --backend $(BACK)

########################
##### pytorch part #####
########################
TORCH_LIB ?= $(PWD)/.libs/pytorch/lib
TORCH_INC ?= $(PWD)/.libs/pytorch/include

pytorch-perf: bin/pytorch-perf
pytorch-perf-test: bin/pytorch-perf-test

bin/pytorch-perf: src/pytorch_perf.cpp src/utils.cpp
	g++ -O3 -o bin/pytorch-perf src/pytorch_perf.cpp -I$(TORCH_INC) -L$(TORCH_LIB) -lc10 -ltorch_cpu -ltorch $(FLAGS)

bin/pytorch-perf-test: src/pytorch_perf.cpp src/utils.cpp
	g++ -O3 -DTEST -o bin/pytorch-perf-test src/pytorch_perf.cpp -I$(TORCH_INC) -L$(TORCH_LIB) -lc10 -ltorch_cpu -ltorch $(FLAGS)

run-pytorch-perf: bin/pytorch-perf
	LD_LIBRARY_PATH=$(TORCH_LIB) bin/pytorch-perf --only-test $(MODEL) --backend $(BACK)

validation-pytorch: bin/pytorch-perf
	LD_LIBRARY_PATH=$(TORCH_LIB) bin/pytorch-perf --only-test $(MODEL) --backend $(BACK) --validation

test-pytorch-perf: bin/pytorch-perf-test
	LD_LIBRARY_PATH=$(TORCH_LIB) bin/pytorch-perf-test --only-test $(MODEL) --backend $(BACK)