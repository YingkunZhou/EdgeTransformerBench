MODEL ?= s1
# make sure no real backend use "z", so that can fall back to CPU exection
BACK ?= z
THREADS ?=1
FP ?= 32
SIZE ?= 224
ANDROID := $(shell uname -a | grep -q Android; echo $$?)

all: ncnn-perf mnn-perf tnn-perf pdlite-perf tflite-perf onnxruntime-perf torch-perf
run-all: run-ncnn-perf run-mnn-perf run-tnn-perf run-pdlite-perf run-tflite-perf run-onnxruntime-perf run-torch-perf
test-all: test-ncnn-perf test-mnn-perf test-tnn-perf test-pdlite-perf test-tflite-perf test-onnxruntime-perf test-torch-perf
validation-all: validation-ncnn validation-mnn validation-tnn validation-pdlite validation-tflite validation-onnxruntime validation-torch

DEPS = src/utils.cpp src/evaluate.tcc src/benchmark.tcc src/PillowResize.cc
# DEF="-DUSE_HWC" make tflite-perf-test
DEF ?=
FLAGS = src/utils.cpp src/PillowResize.cc -std=c++17 `pkg-config --cflags --libs opencv4` $(DEF)
VAL_EXTRA?=

########################
###### ncnn part #######
########################
NCNN_LIB ?= $(PWD)/.libs/ncnn/install/lib
NCNN_INC ?= $(PWD)/.libs/ncnn/install/include/ncnn

ncnn-perf: bin/ncnn-perf
ncnn-perf-test: bin/ncnn-perf-test

bin/ncnn-perf: src/ncnn-perf.cpp $(DEPS)
	LD_LIBRARY_PATH=$(NCNN_LIB) $(CXX) -O3 -o bin/ncnn-perf src/ncnn-perf.cpp -I$(NCNN_INC) -L$(NCNN_LIB) -lncnn $(FLAGS)

bin/ncnn-perf-test: src/ncnn-perf.cpp $(DEPS)
	LD_LIBRARY_PATH=$(NCNN_LIB) $(CXX) -O3 -DTEST -o bin/ncnn-perf-test src/ncnn-perf.cpp -I$(NCNN_INC) -L$(NCNN_LIB) -lncnn $(FLAGS)

run-ncnn-perf: bin/ncnn-perf
	LD_LIBRARY_PATH=$(NCNN_LIB):$(LD_LIBRARY_PATH) bin/ncnn-perf --only-test $(MODEL) --backend $(BACK) --threads $(THREADS) --fp $(FP)

validation-ncnn: bin/ncnn-perf
	LD_LIBRARY_PATH=$(NCNN_LIB):$(LD_LIBRARY_PATH) bin/ncnn-perf --only-test $(MODEL) --backend $(BACK) --validation --threads $(THREADS) --fp $(FP) $(VAL_EXTRA)

ncnn-model-perf: bin/ncnn-perf
	LD_LIBRARY_PATH=$(NCNN_LIB):$(LD_LIBRARY_PATH) bin/ncnn-perf --model $(MODEL) --backend $(BACK) --threads $(THREADS) --fp $(FP) --size $(SIZE)

ncnn-model-validation: bin/ncnn-perf
	LD_LIBRARY_PATH=$(NCNN_LIB):$(LD_LIBRARY_PATH) bin/ncnn-perf --model $(MODEL) --backend $(BACK) --validation --threads $(THREADS) --fp $(FP) --size $(SIZE) $(VAL_EXTRA)

test-ncnn-perf: bin/ncnn-perf-test
	LD_LIBRARY_PATH=$(NCNN_LIB):$(LD_LIBRARY_PATH) bin/ncnn-perf-test --only-test $(MODEL) --backend $(BACK) --threads $(THREADS) --fp $(FP)

ncnn-model-test: bin/ncnn-perf-test
	LD_LIBRARY_PATH=$(NCNN_LIB):$(LD_LIBRARY_PATH) bin/ncnn-perf-test --model $(MODEL) --backend $(BACK) --threads $(THREADS) --fp $(FP) --size $(SIZE)

########################
###### mnn part ########
########################
MNN_LIB ?= $(PWD)/.libs/MNN/install/lib
MNN_INC ?= $(PWD)/.libs/MNN/install/include

mnn-perf: bin/mnn-perf
mnn-perf-test: bin/mnn-perf-test

bin/mnn-perf: src/mnn-perf.cpp $(DEPS)
	$(CXX) -O3 -o bin/mnn-perf src/mnn-perf.cpp -I$(MNN_INC) -L$(MNN_LIB) -lMNN $(FLAGS)

bin/mnn-perf-test: src/mnn-perf.cpp $(DEPS)
	$(CXX) -O3 -DTEST -o bin/mnn-perf-test src/mnn-perf.cpp -I$(MNN_INC) -L$(MNN_LIB) -lMNN $(FLAGS)

run-mnn-perf: bin/mnn-perf
	LD_LIBRARY_PATH=$(MNN_LIB):$(LD_LIBRARY_PATH) bin/mnn-perf --only-test $(MODEL) --backend $(BACK) --threads $(THREADS) --fp $(FP)

validation-mnn: bin/mnn-perf
	LD_LIBRARY_PATH=$(MNN_LIB):$(LD_LIBRARY_PATH) bin/mnn-perf --only-test $(MODEL) --backend $(BACK) --validation --threads $(THREADS) --fp $(FP) $(VAL_EXTRA)

mnn-model-perf: bin/mnn-perf
	LD_LIBRARY_PATH=$(MNN_LIB):$(LD_LIBRARY_PATH) bin/mnn-perf --model $(MODEL) --backend $(BACK) --threads $(THREADS) --fp $(FP) --size $(SIZE)

mnn-model-validation: bin/mnn-perf
	LD_LIBRARY_PATH=$(MNN_LIB):$(LD_LIBRARY_PATH) bin/mnn-perf --model $(MODEL) --backend $(BACK) --validation --threads $(THREADS) --fp $(FP) --size $(SIZE) $(VAL_EXTRA)

test-mnn-perf: bin/mnn-perf-test
	LD_LIBRARY_PATH=$(MNN_LIB):$(LD_LIBRARY_PATH) bin/mnn-perf-test --only-test $(MODEL) --backend $(BACK) --threads $(THREADS) --fp $(FP)

mnn-model-test: bin/mnn-perf-test
	LD_LIBRARY_PATH=$(MNN_LIB):$(LD_LIBRARY_PATH) bin/mnn-perf-test --model $(MODEL) --backend $(BACK) --threads $(THREADS) --fp $(FP) --size $(SIZE)

########################
###### tnn part ########
########################
TNN_LIB ?= $(PWD)/.libs/TNN/install/lib
TNN_INC ?= $(PWD)/.libs/TNN/install/include

tnn-perf: bin/tnn-perf
tnn-perf-test: bin/tnn-perf-test

bin/tnn-perf: src/tnn-perf.cpp $(DEPS)
	$(CXX) -O3 -o bin/tnn-perf src/tnn-perf.cpp -I$(TNN_INC) -L$(TNN_LIB) -lTNN $(FLAGS)

bin/tnn-perf-test: src/tnn-perf.cpp $(DEPS)
	$(CXX) -O3 -DTEST -o bin/tnn-perf-test src/tnn-perf.cpp -I$(TNN_INC) -L$(TNN_LIB) -lTNN $(FLAGS)

run-tnn-perf: bin/tnn-perf
	LD_LIBRARY_PATH=$(TNN_LIB):$(LD_LIBRARY_PATH) bin/tnn-perf --only-test $(MODEL) --backend $(BACK) --threads $(THREADS) --fp $(FP)

validation-tnn: bin/tnn-perf
	LD_LIBRARY_PATH=$(TNN_LIB):$(LD_LIBRARY_PATH) bin/tnn-perf --only-test $(MODEL) --backend $(BACK) --validation --threads $(THREADS) --fp $(FP) $(VAL_EXTRA)

tnn-model-perf: bin/tnn-perf
	LD_LIBRARY_PATH=$(TNN_LIB):$(LD_LIBRARY_PATH) bin/tnn-perf --model $(MODEL) --backend $(BACK) --threads $(THREADS) --fp $(FP) --size $(SIZE)

tnn-model-validation: bin/tnn-perf
	LD_LIBRARY_PATH=$(TNN_LIB):$(LD_LIBRARY_PATH) bin/tnn-perf --model $(MODEL) --backend $(BACK) --validation --threads $(THREADS) --fp $(FP) --size $(SIZE) $(VAL_EXTRA)

test-tnn-perf: bin/tnn-perf-test
	LD_LIBRARY_PATH=$(TNN_LIB):$(LD_LIBRARY_PATH) bin/tnn-perf-test --only-test $(MODEL) --backend $(BACK) --threads $(THREADS) --fp $(FP)

tnn-model-test: bin/tnn-perf-test
	LD_LIBRARY_PATH=$(TNN_LIB):$(LD_LIBRARY_PATH) bin/tnn-perf-test --model $(MODEL) --backend $(BACK) --threads $(THREADS) --fp $(FP) --size $(SIZE)

########################
##### pdlite part ######
########################
PDLITE_LIB ?= $(PWD)/.libs/Paddle-Lite/lib
PDLITE_INC ?= $(PWD)/.libs/Paddle-Lite/include

pdlite-perf: bin/pdlite-perf
pdlite-perf-test: bin/pdlite-perf-test

bin/pdlite-perf: src/pdlite-perf.cpp $(DEPS)
	$(CXX) -O3 -o bin/pdlite-perf src/pdlite-perf.cpp -I$(PDLITE_INC)  -L$(PDLITE_LIB) -lpaddle_light_api_shared $(FLAGS)

bin/pdlite-perf-test: src/pdlite-perf.cpp $(DEPS)
	$(CXX) -O3 -DTEST -o bin/pdlite-perf-test src/pdlite-perf.cpp -I$(PDLITE_INC) -L$(PDLITE_LIB) -lpaddle_light_api_shared $(FLAGS)

run-pdlite-perf: bin/pdlite-perf
	LD_LIBRARY_PATH=$(PDLITE_LIB):$(LD_LIBRARY_PATH) bin/pdlite-perf --only-test $(MODEL) --backend $(BACK) --threads $(THREADS) --fp $(FP)

validation-pdlite: bin/pdlite-perf
	LD_LIBRARY_PATH=$(PDLITE_LIB):$(LD_LIBRARY_PATH) bin/pdlite-perf --only-test $(MODEL) --backend $(BACK) --validation --threads $(THREADS) --fp $(FP) $(VAL_EXTRA)

pdlite-model-perf: bin/pdlite-perf
	LD_LIBRARY_PATH=$(PDLITE_LIB):$(LD_LIBRARY_PATH) bin/pdlite-perf --model $(MODEL) --backend $(BACK) --threads $(THREADS) --fp $(FP) --size $(SIZE)

pdlite-model-validation: bin/pdlite-perf
	LD_LIBRARY_PATH=$(PDLITE_LIB):$(LD_LIBRARY_PATH) bin/pdlite-perf --model $(MODEL) --backend $(BACK) --validation --threads $(THREADS) --fp $(FP) --size $(SIZE) $(VAL_EXTRA)

test-pdlite-perf: bin/pdlite-perf-test
	LD_LIBRARY_PATH=$(PDLITE_LIB):$(LD_LIBRARY_PATH) bin/pdlite-perf-test --only-test $(MODEL) --backend $(BACK) --threads $(THREADS) --fp $(FP)

pdlite-model-test: bin/pdlite-perf-test
	LD_LIBRARY_PATH=$(PDLITE_LIB):$(LD_LIBRARY_PATH) bin/pdlite-perf-test --model $(MODEL) --backend $(BACK) --threads $(THREADS) --fp $(FP) --size $(SIZE)

########################
##### tflite part ######
########################
TFLITE_INC ?= $(PWD)/.libs/tensorflow/install/include
TFLITE_LIB ?= $(PWD)/.libs/tensorflow/install/lib

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
GPU_FLAGS = -ltensorflowlite_gpu_delegate -DUSE_GPU -lEGL
ifneq ($(ANDROID),0)
	ARMNN_FLAGS += -lGL
endif

tflite-perf: bin/tflite-perf
tflite-perf-test: bin/tflite-perf-test

bin/tflite-perf: src/tflite-perf.cpp $(DEPS)
	LD_LIBRARY_PATH=$(TFLITE_LIB) $(CXX) -O3 -o bin/tflite-perf src/tflite-perf.cpp -I$(TFLITE_INC) -L$(TFLITE_LIB) $(FLAGS) -ltensorflowlite \
	$(ARMNN_FLAGS) $(GPU_FLAGS) $(NNAPI_FLAGS)

bin/tflite-perf-test: src/tflite-perf.cpp $(DEPS)
	LD_LIBRARY_PATH=$(TFLITE_LIB) $(CXX) -O3 -DTEST -o bin/tflite-perf-test src/tflite-perf.cpp -I$(TFLITE_INC) -L$(TFLITE_LIB) $(FLAGS) -ltensorflowlite \
	$(ARMNN_FLAGS) $(GPU_FLAGS) $(NNAPI_FLAGS)

run-tflite-perf: bin/tflite-perf
	LD_PRELOAD=$(TFLITE_LIB)/libtensorflowlite_flex.so LD_LIBRARY_PATH=$(TFLITE_LIB):$(LD_LIBRARY_PATH) \
	bin/tflite-perf --only-test $(MODEL) --backend $(BACK) --threads $(THREADS) --fp $(FP)

validation-tflite: bin/tflite-perf
	LD_PRELOAD=$(TFLITE_LIB)/libtensorflowlite_flex.so LD_LIBRARY_PATH=$(TFLITE_LIB):$(LD_LIBRARY_PATH) \
	bin/tflite-perf --only-test $(MODEL) --backend $(BACK) --validation --threads $(THREADS) --fp $(FP) $(VAL_EXTRA)

tflite-model-perf: bin/tflite-perf
	LD_PRELOAD=$(TFLITE_LIB)/libtensorflowlite_flex.so LD_LIBRARY_PATH=$(TFLITE_LIB):$(LD_LIBRARY_PATH) \
	bin/tflite-perf --model $(MODEL) --backend $(BACK) --threads $(THREADS) --fp $(FP) --size $(SIZE)

tflite-model-validation: bin/tflite-perf
	LD_PRELOAD=$(TFLITE_LIB)/libtensorflowlite_flex.so LD_LIBRARY_PATH=$(TFLITE_LIB):$(LD_LIBRARY_PATH) \
	bin/tflite-perf --model $(MODEL) --backend $(BACK) --validation --threads $(THREADS) --fp $(FP) --size $(SIZE) $(VAL_EXTRA)

test-tflite-perf: bin/tflite-perf-test
	LD_PRELOAD=$(TFLITE_LIB)/libtensorflowlite_flex.so LD_LIBRARY_PATH=$(TFLITE_LIB):$(LD_LIBRARY_PATH) \
	bin/tflite-perf-test --only-test $(MODEL) --backend $(BACK) --threads $(THREADS) --fp $(FP)

tflite-model-test: bin/tflite-perf-test
	LD_PRELOAD=$(TFLITE_LIB)/libtensorflowlite_flex.so LD_LIBRARY_PATH=$(TFLITE_LIB):$(LD_LIBRARY_PATH) \
	bin/tflite-perf-test --model $(MODEL) --backend $(BACK) --threads $(THREADS) --fp $(FP) --size $(SIZE)

########################
### onnxruntime part ###
########################
ONNXRT_INC ?= $(PWD)/.libs/onnxruntime/include
ONNXRT_LIB ?= $(PWD)/.libs/onnxruntime/lib

ONNXRT_NNAPI_FLAGS ?=
ifeq ($(ANDROID),0)
	ONNXRT_NNAPI_FLAGS += -DUSE_NNAPI
endif

TENSORRT_FLAGS ?=
ACL_FLAGS ?=
DNNL_FLAGS ?=
COREML_FLAGS ?=

onnxruntime-perf: bin/onnxruntime-perf
onnxruntime-perf-test: bin/tflionnxruntimete-perf-test

bin/onnxruntime-perf: src/onnxruntime-perf.cpp $(DEPS)
	$(CXX) -O3 -o bin/onnxruntime-perf src/onnxruntime-perf.cpp -I$(ONNXRT_INC)  -L$(ONNXRT_LIB) $(FLAGS) -lonnxruntime $(ONNXRT_NNAPI_FLAGS) $(TENSORRT_FLAGS) $(ACL_FLAGS) $(DNNL_FLAGS) $(COREML_FLAGS)

bin/onnxruntime-perf-test: src/onnxruntime-perf.cpp $(DEPS)
	$(CXX) -O3 -DTEST -o bin/onnxruntime-perf-test src/onnxruntime-perf.cpp -I$(ONNXRT_INC)  -L$(ONNXRT_LIB) $(FLAGS) -lonnxruntime $(ONNXRT_NNAPI_FLAGS) $(TENSORRT_FLAGS) $(ACL_FLAGS) $(DNNL_FLAGS) $(COREML_FLAGS)

run-onnxruntime-perf: bin/onnxruntime-perf
	LD_LIBRARY_PATH=$(ONNXRT_LIB):$(LD_LIBRARY_PATH) bin/onnxruntime-perf --only-test $(MODEL) --backend $(BACK) --threads $(THREADS)

validation-onnxruntime: bin/onnxruntime-perf
	LD_LIBRARY_PATH=$(ONNXRT_LIB):$(LD_LIBRARY_PATH) bin/onnxruntime-perf --only-test $(MODEL) --backend $(BACK) --validation --threads $(THREADS) $(VAL_EXTRA)

onnxruntime-model-perf: bin/onnxruntime-perf
	LD_LIBRARY_PATH=$(ONNXRT_LIB):$(LD_LIBRARY_PATH) bin/onnxruntime-perf --model $(MODEL) --backend $(BACK) --threads $(THREADS) --size $(SIZE)

onnxruntime-model-validation: bin/onnxruntime-perf
	LD_LIBRARY_PATH=$(ONNXRT_LIB):$(LD_LIBRARY_PATH) bin/onnxruntime-perf --model $(MODEL) --backend $(BACK) --validation --threads $(THREADS) --size $(SIZE) $(VAL_EXTRA)

test-onnxruntime-perf: bin/onnxruntime-perf-test
	LD_LIBRARY_PATH=$(ONNXRT_LIB):$(LD_LIBRARY_PATH) bin/onnxruntime-perf-test --only-test $(MODEL) --backend $(BACK) --threads $(THREADS)

onnxruntime-model-test: bin/onnxruntime-perf-test
	LD_LIBRARY_PATH=$(ONNXRT_LIB):$(LD_LIBRARY_PATH) bin/onnxruntime-perf-test --model $(MODEL) --backend $(BACK) --threads $(THREADS) --size $(SIZE)

########################
##### torch part #####
########################
TORCH_LIB ?= $(PWD)/.libs/torch/lib
TORCH_INC ?= $(PWD)/.libs/torch/include

torch-perf: bin/torch-perf
torch-perf-test: bin/torch-perf-test

bin/torch-perf: src/torch-perf.cpp $(DEPS)
	LD_LIBRARY_PATH=$(TORCH_LIB) $(CXX) -O3 -o bin/torch-perf src/torch-perf.cpp -I$(TORCH_INC) -L$(TORCH_LIB) -lc10 -ltorch_cpu -ltorch $(FLAGS)
	#-DUSE_TORCH_MOBILE
	#-D_GLIBCXX_USE_CXX11_ABI=0

bin/torch-perf-test: src/torch-perf.cpp $(DEPS)
	LD_LIBRARY_PATH=$(TORCH_LIB) $(CXX) -O3 -DTEST -o bin/torch-perf-test src/torch-perf.cpp -I$(TORCH_INC) -L$(TORCH_LIB) -lc10 -ltorch_cpu -ltorch $(FLAGS)
	#-DUSE_TORCH_MOBILE
	#-D_GLIBCXX_USE_CXX11_ABI=0

run-torch-perf: bin/torch-perf
	LD_LIBRARY_PATH=$(TORCH_LIB):$(LD_LIBRARY_PATH) bin/torch-perf --only-test $(MODEL) --backend $(BACK) --threads $(THREADS)

validation-torch: bin/torch-perf
	LD_LIBRARY_PATH=$(TORCH_LIB):$(LD_LIBRARY_PATH) bin/torch-perf --only-test $(MODEL) --backend $(BACK) --validation --threads $(THREADS) $(VAL_EXTRA)

torch-model-perf: bin/torch-perf
	LD_LIBRARY_PATH=$(TORCH_LIB):$(LD_LIBRARY_PATH) bin/torch-perf --model $(MODEL) --backend $(BACK) --threads $(THREADS) --size $(SIZE)

torch-model-validation: bin/torch-perf
	LD_LIBRARY_PATH=$(TORCH_LIB):$(LD_LIBRARY_PATH) bin/torch-perf --model $(MODEL) --backend $(BACK) --validation --threads $(THREADS) $(VAL_EXTRA) --size $(SIZE)

test-torch-perf: bin/torch-perf-test
	LD_LIBRARY_PATH=$(TORCH_LIB):$(LD_LIBRARY_PATH) bin/torch-perf-test --only-test $(MODEL) --backend $(BACK) --threads $(THREADS)

torch-model-test: bin/torch-perf-test
	LD_LIBRARY_PATH=$(TORCH_LIB):$(LD_LIBRARY_PATH) bin/torch-perf-test --model $(MODEL) --backend $(BACK) --threads $(THREADS) --size $(SIZE)


########################
##### tvm part #####
########################
TVM_LIB ?= $(PWD)/.libs/tvm/install/lib
TVM_INC ?= $(PWD)/.libs/tvm/install/include

tvm-perf: bin/tvm-perf
tvm-perf-test: bin/torch-perf-test

bin/tvm-perf: src/tvm-perf.cpp $(DEPS)
	$(CXX) -O3 -o bin/tvm-perf src/tvm-perf.cpp -I$(TVM_INC) -L$(TVM_LIB) -ldl -pthread -ltvm_runtime $(FLAGS) -DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\>

bin/tvm-perf-test: src/tvm-perf.cpp $(DEPS)
	$(CXX) -O3 -DTEST -o bin/tvm-perf-test src/tvm-perf.cpp -I$(TVM_INC) -L$(TVM_LIB) -ldl -pthread -ltvm_runtime $(FLAGS) -DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\>

run-tvm-perf: bin/tvm-perf
	TVM_NUM_THREADS=$(THREADS) LD_LIBRARY_PATH=$(TVM_LIB):$(LD_LIBRARY_PATH) bin/tvm-perf --only-test $(MODEL) --backend $(BACK)

validation-tvm: bin/tvm-perf
	TVM_NUM_THREADS=$(THREADS) LD_LIBRARY_PATH=$(TVM_LIB):$(LD_LIBRARY_PATH) bin/tvm-perf --only-test $(MODEL) --backend $(BACK) --validation $(VAL_EXTRA)

tvm-model-perf: bin/tvm-perf
	TVM_NUM_THREADS=$(THREADS) LD_LIBRARY_PATH=$(TVM_LIB):$(LD_LIBRARY_PATH) bin/tvm-perf --model $(MODEL) --backend $(BACK) --size $(SIZE)

tvm-model-validation: bin/tvm-perf
	TVM_NUM_THREADS=$(THREADS) LD_LIBRARY_PATH=$(TVM_LIB):$(LD_LIBRARY_PATH) bin/tvm-perf --model $(MODEL) --backend $(BACK) --validation $(VAL_EXTRA) --size $(SIZE)

test-tvm-perf: bin/tvm-perf-test
	TVM_NUM_THREADS=$(THREADS) LD_LIBRARY_PATH=$(TVM_LIB):$(LD_LIBRARY_PATH) bin/tvm-perf-test --only-test $(MODEL) --backend $(BACK)

tvm-model-test: bin/tvm-perf-test
	TVM_NUM_THREADS=$(THREADS) LD_LIBRARY_PATH=$(TVM_LIB):$(LD_LIBRARY_PATH) bin/tvm-perf-test --model $(MODEL) --backend $(BACK) --size $(SIZE)
