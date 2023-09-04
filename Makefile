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

TNN_INC ?= $(HOME)/work/TNN/install/include
TNN_LIB ?= $(HOME)/work/TNN/install/lib

tnn-perf:
	g++ -O3 -o tnn-perf tnn_perf.cpp -I$(TNN_INC) -L$(TNN_LIB) -lTNN $(FLAGS)

tnn-perf-test:
	g++ -O3 -DTEST -o tnn-perf-test tnn_perf.cpp -I$(TNN_INC) -L$(TNN_LIB) -lTNN $(FLAGS)

PDLITE_LIB ?= $(HOME)/work/Paddle-Lite/build.lite.android.armv8.clang/inference_lite_lib.android.armv8.opencl/cxx/lib
PDLITE_INC ?= $(HOME)/work/Paddle-Lite/build.lite.android.armv8.clang/inference_lite_lib.android.armv8.opencl/cxx/include

pdlite-perf:
	g++ -O3 -o pdlite-perf pdlite_perf.cpp -I$(PDLITE_INC)  -L$(PDLITE_LIB) -lpaddle_light_api_shared  $(FLAGS)