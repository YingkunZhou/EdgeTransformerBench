MODEL ?= s1
# make sure no real backend use "z", so that can fall back to CPU exection
BACK ?= z
THREADS ?=1
FP ?= 32

all: ncnn-perf mnn-perf tnn-perf pdlite-perf tflite-perf onnxruntime-perf torch-perf
run-all: run-ncnn-perf run-mnn-perf run-tnn-perf run-pdlite-perf run-tflite-perf run-onnxruntime-perf run-torch-perf
test-all: test-ncnn-perf test-mnn-perf test-tnn-perf test-pdlite-perf test-tflite-perf test-onnxruntime-perf test-torch-perf
validation-all: validation-ncnn validation-mnn validation-tnn validation-pdlite validation-tflite validation-onnxruntime validation-torch

init:
	./prepare.sh

DEPS = src/utils.cpp src/evaluate.tcc src/benchmark.tcc src/PillowResize.cc
# DEF="-DUSE_HWC" make tflite-perf-test
DEF ?=
FLAGS = src/utils.cpp src/PillowResize.cc -std=c++17 `pkg-config --cflags --libs opencv4` $(DEF)

########################
###### ncnn part #######
########################
NCNN_LIB ?= $(PWD)/.libs/ncnn/install/lib
NCNN_INC ?= $(PWD)/.libs/ncnn/install/include/ncnn

ncnn-perf: bin/ncnn-perf
ncnn-perf-test: bin/ncnn-perf-test

bin/ncnn-perf: src/ncnn_perf.cpp $(DEPS)
	LD_LIBRARY_PATH=$(NCNN_LIB) $(CXX) -O3 -o bin/ncnn-perf src/ncnn_perf.cpp -I$(NCNN_INC) -L$(NCNN_LIB) -lncnn $(FLAGS)

bin/ncnn-perf-test: src/ncnn_perf.cpp $(DEPS)
	LD_LIBRARY_PATH=$(NCNN_LIB) $(CXX) -O3 -DTEST -o bin/ncnn-perf-test src/ncnn_perf.cpp -I$(NCNN_INC) -L$(NCNN_LIB) -lncnn $(FLAGS)

run-ncnn-perf: bin/ncnn-perf
	LD_LIBRARY_PATH=$(NCNN_LIB):$(LD_LIBRARY_PATH) bin/ncnn-perf --only-test $(MODEL) --backend $(BACK) --threads $(THREADS)

validation-ncnn: bin/ncnn-perf
	LD_LIBRARY_PATH=$(NCNN_LIB):$(LD_LIBRARY_PATH) bin/ncnn-perf --only-test $(MODEL) --backend $(BACK) --validation --threads $(THREADS)

test-ncnn-perf: bin/ncnn-perf-test
	LD_LIBRARY_PATH=$(NCNN_LIB):$(LD_LIBRARY_PATH) bin/ncnn-perf-test --only-test $(MODEL) --backend $(BACK) --threads $(THREADS)

########################
###### mnn part ########
########################
MNN_LIB ?= $(PWD)/.libs/MNN/install/lib
MNN_INC ?= $(PWD)/.libs/MNN/install/include

mnn-perf: bin/mnn-perf
mnn-perf-test: bin/mnn-perf-test

bin/mnn-perf: src/mnn_perf.cpp $(DEPS)
	$(CXX) -O3 -o bin/mnn-perf src/mnn_perf.cpp -I$(MNN_INC) -L$(MNN_LIB) -lMNN $(FLAGS)

bin/mnn-perf-test: src/mnn_perf.cpp $(DEPS)
	$(CXX) -O3 -DTEST -o bin/mnn-perf-test src/mnn_perf.cpp -I$(MNN_INC) -L$(MNN_LIB) -lMNN $(FLAGS)

run-mnn-perf: bin/mnn-perf
	LD_LIBRARY_PATH=$(MNN_LIB):$(LD_LIBRARY_PATH) bin/mnn-perf --only-test $(MODEL) --backend $(BACK) --threads $(THREADS)

validation-mnn: bin/mnn-perf
	LD_LIBRARY_PATH=$(MNN_LIB):$(LD_LIBRARY_PATH) bin/mnn-perf --only-test $(MODEL) --backend $(BACK) --validation --threads $(THREADS)

test-mnn-perf: bin/mnn-perf-test
	LD_LIBRARY_PATH=$(MNN_LIB):$(LD_LIBRARY_PATH) bin/mnn-perf-test --only-test $(MODEL) --backend $(BACK) --threads $(THREADS)

########################
###### tnn part ########
########################
TNN_LIB ?= $(PWD)/.libs/TNN/install/lib
TNN_INC ?= $(PWD)/.libs/TNN/install/include

tnn-perf: bin/tnn-perf
tnn-perf-test: bin/tnn-perf-test

bin/tnn-perf: src/tnn_perf.cpp $(DEPS)
	$(CXX) -O3 -o bin/tnn-perf src/tnn_perf.cpp -I$(TNN_INC) -L$(TNN_LIB) -lTNN $(FLAGS)

bin/tnn-perf-test: src/tnn_perf.cpp $(DEPS)
	$(CXX) -O3 -DTEST -o bin/tnn-perf-test src/tnn_perf.cpp -I$(TNN_INC) -L$(TNN_LIB) -lTNN $(FLAGS)

run-tnn-perf: bin/tnn-perf
	LD_LIBRARY_PATH=$(TNN_LIB):$(LD_LIBRARY_PATH) bin/tnn-perf --only-test $(MODEL) --backend $(BACK) --threads $(THREADS)

validation-tnn: bin/tnn-perf
	LD_LIBRARY_PATH=$(TNN_LIB):$(LD_LIBRARY_PATH) bin/tnn-perf --only-test $(MODEL) --backend $(BACK) --validation --threads $(THREADS)

test-tnn-perf: bin/tnn-perf-test
	LD_LIBRARY_PATH=$(TNN_LIB):$(LD_LIBRARY_PATH) bin/tnn-perf-test --only-test $(MODEL) --backend $(BACK) --threads $(THREADS)

########################
##### pdlite part ######
########################
PDLITE_LIB ?= $(PWD)/.libs/Paddle-Lite/lib
PDLITE_INC ?= $(PWD)/.libs/Paddle-Lite/include

pdlite-perf: bin/pdlite-perf
pdlite-perf-test: bin/pdlite-perf-test

bin/pdlite-perf: src/pdlite_perf.cpp $(DEPS)
	$(CXX) -O3 -o bin/pdlite-perf src/pdlite_perf.cpp -I$(PDLITE_INC)  -L$(PDLITE_LIB) -lpaddle_light_api_shared $(FLAGS)

bin/pdlite-perf-test: src/pdlite_perf.cpp $(DEPS)
	$(CXX) -O3 -DTEST -o bin/pdlite-perf-test src/pdlite_perf.cpp -I$(PDLITE_INC) -L$(PDLITE_LIB) -lpaddle_light_api_shared $(FLAGS)

run-pdlite-perf: bin/pdlite-perf
	LD_LIBRARY_PATH=$(PDLITE_LIB):$(LD_LIBRARY_PATH) bin/pdlite-perf --only-test $(MODEL) --backend $(BACK) --threads $(THREADS)

validation-pdlite: bin/pdlite-perf
	LD_LIBRARY_PATH=$(PDLITE_LIB):$(LD_LIBRARY_PATH) bin/pdlite-perf --only-test $(MODEL) --backend $(BACK) --validation --threads $(THREADS)

test-pdlite-perf: bin/pdlite-perf-test
	LD_LIBRARY_PATH=$(PDLITE_LIB):$(LD_LIBRARY_PATH) bin/pdlite-perf-test --only-test $(MODEL) --backend $(BACK) --threads $(THREADS)

########################
##### tflite part ######
########################
TFLITE_INC ?= $(PWD)/.libs/tensorflow/install/include
TFLITE_LIB ?= $(PWD)/.libs/tensorflow/install/lib

ANDROID := $(shell uname -a | grep -q Android; echo $$?)

NNAPI_FLAGS =
ifeq ($(ANDROID),0)
	NNAPI_FLAGS += -lnnapi_util -lnnapi_delegate_no_nnapi_implementation -lnnapi_implementation -DUSE_NNAPI
endif

ARMNN_FLAGS = -I$(TFLITE_INC)/armnn/delegate/classic/include -I$(TFLITE_INC)/armnn/delegate/common/include \
-I$(TFLITE_INC)/armnn/include -larmnnDelegate -larmnn -DUSE_ARMNN
ifneq ($(ANDROID),0)
	ARMNN_FLAGS += -lflatbuffers
endif

# sudo apt install libgles2-mesa-dev libegl1-mesa-dev xorg-dev
GPU_FLAGS = -ltensorflowlite_gpu_delegate -DUSE_GPU -lGL -lEGL
ifneq ($(ANDROID),0)
	ARMNN_FLAGS += -lEGL
endif

tflite-perf: bin/tflite-perf
tflite-perf-test: bin/tflite-perf-test

bin/tflite-perf: src/tflite_perf.cpp $(DEPS)
	LD_LIBRARY_PATH=$(TFLITE_LIB) $(CXX) -O3 -o bin/tflite-perf src/tflite_perf.cpp -I$(TFLITE_INC) -L$(TFLITE_LIB) $(FLAGS) -ltensorflowlite \
	$(ARMNN_FLAGS) $(GPU_FLAGS) $(NNAPI_FLAGS)

bin/tflite-perf-test: src/tflite_perf.cpp $(DEPS)
	LD_LIBRARY_PATH=$(TFLITE_LIB) $(CXX) -O3 -DTEST -o bin/tflite-perf-test src/tflite_perf.cpp -I$(TFLITE_INC) -L$(TFLITE_LIB) $(FLAGS) -ltensorflowlite \
	$(ARMNN_FLAGS) $(GPU_FLAGS) $(NNAPI_FLAGS)

run-tflite-perf: bin/tflite-perf
	LD_PRELOAD=$(TFLITE_LIB)/libtensorflowlite_flex.so LD_LIBRARY_PATH=$(TFLITE_LIB):$(LD_LIBRARY_PATH) \
	bin/tflite-perf --only-test $(MODEL) --backend $(BACK) --threads $(THREADS) --fp $(FP)

validation-tflite: bin/tflite-perf
	LD_PRELOAD=$(TFLITE_LIB)/libtensorflowlite_flex.so LD_LIBRARY_PATH=$(TFLITE_LIB):$(LD_LIBRARY_PATH) \
	bin/tflite-perf --only-test $(MODEL) --backend $(BACK) --validation --threads $(THREADS) --fp $(FP)

test-tflite-perf: bin/tflite-perf-test
	LD_PRELOAD=$(TFLITE_LIB)/libtensorflowlite_flex.so LD_LIBRARY_PATH=$(TFLITE_LIB):$(LD_LIBRARY_PATH) \
	bin/tflite-perf-test --only-test $(MODEL) --backend $(BACK) --threads $(THREADS) --fp $(FP)

########################
### onnxruntime part ###
########################
ONNXRT_INC ?= $(PWD)/.libs/onnxruntime/include
ONNXRT_LIB ?= $(PWD)/.libs/onnxruntime/lib
MORE_FLAGS ?=

tflite-perf: bin/tflite-perf
tflite-perf-test: bin/tflite-perf-test

bin/onnxruntime-perf: src/onnxruntime_perf.cpp $(DEPS)
	$(CXX) -O3 -o bin/onnxruntime-perf src/onnxruntime_perf.cpp -I$(ONNXRT_INC)  -L$(ONNXRT_LIB) $(FLAGS) -lonnxruntime $(MORE_FLAGS)

bin/onnxruntime-perf-test: src/onnxruntime_perf.cpp $(DEPS)
	$(CXX) -O3 -DTEST -o bin/onnxruntime-perf-test src/onnxruntime_perf.cpp -I$(ONNXRT_INC)  -L$(ONNXRT_LIB) $(FLAGS) -lonnxruntime $(MORE_FLAGS)

run-onnxruntime-perf: bin/onnxruntime-perf
	LD_LIBRARY_PATH=$(ONNXRT_LIB):$(LD_LIBRARY_PATH) bin/onnxruntime-perf --only-test $(MODEL) --backend $(BACK) --threads $(THREADS)

validation-onnxruntime: bin/onnxruntime-perf
	LD_LIBRARY_PATH=$(ONNXRT_LIB):$(LD_LIBRARY_PATH) bin/onnxruntime-perf --only-test $(MODEL) --backend $(BACK) --validation --threads $(THREADS)

test-onnxruntime-perf: bin/onnxruntime-perf-test
	LD_LIBRARY_PATH=$(ONNXRT_LIB):$(LD_LIBRARY_PATH) bin/onnxruntime-perf-test --only-test $(MODEL) --backend $(BACK) --threads $(THREADS)

########################
##### torch part #####
########################
TORCH_LIB ?= $(PWD)/.libs/torch/lib
TORCH_INC ?= $(PWD)/.libs/torch/include

torch-perf: bin/torch-perf
torch-perf-test: bin/torch-perf-test

bin/torch-perf: src/torch_perf.cpp $(DEPS)
	LD_LIBRARY_PATH=$(TORCH_LIB) $(CXX) -O3 -o bin/torch-perf src/torch_perf.cpp -I$(TORCH_INC) -L$(TORCH_LIB) -lc10 -ltorch_cpu -ltorch $(FLAGS)
	#-DUSE_TORCH_MOBILE
	#-D_GLIBCXX_USE_CXX11_ABI=0

bin/torch-perf-test: src/torch_perf.cpp $(DEPS)
	LD_LIBRARY_PATH=$(TORCH_LIB) $(CXX) -O3 -DTEST -o bin/torch-perf-test src/torch_perf.cpp -I$(TORCH_INC) -L$(TORCH_LIB) -lc10 -ltorch_cpu -ltorch $(FLAGS)
	#-DUSE_TORCH_MOBILE
	#-D_GLIBCXX_USE_CXX11_ABI=0

run-torch-perf: bin/torch-perf
	LD_LIBRARY_PATH=$(TORCH_LIB):$(LD_LIBRARY_PATH) bin/torch-perf --only-test $(MODEL) --backend $(BACK) --threads $(THREADS)

validation-torch: bin/torch-perf
	LD_LIBRARY_PATH=$(TORCH_LIB):$(LD_LIBRARY_PATH) bin/torch-perf --only-test $(MODEL) --backend $(BACK) --validation --threads $(THREADS)

test-torch-perf: bin/torch-perf-test
	LD_LIBRARY_PATH=$(TORCH_LIB):$(LD_LIBRARY_PATH) bin/torch-perf-test --only-test $(MODEL) --backend $(BACK) --threads $(THREADS)