```bash
export CC=/usr/bin/clang-16
export CXX=/usr/bin/clang++-16
```

# ncnn

<details>
<summary>Linux</summary>

```bash
git clone https://github.com/Tencent/ncnn.git #--depth=1
cd ncnn
#git submodule sync
git submodule update --init --recursive
mkdir -p build && cd build
cmake -D NCNN_SHARED_LIB=ON -D NCNN_VULKAN=ON .. -D CMAKE_BUILD_TYPE=Release \
-D CMAKE_INSTALL_PREFIX=../install -D NCNN_BUILD_BENCHMARK=OFF
make install -j`nproc`
```
</details>


# mnn

<details>
<summary>Linux</summary>

```bash
git clone https://github.com/alibaba/MNN.git #--depth=1
cd MNN
mkdir -p build && build
cmake -D CMAKE_BUILD_TYPE=Release -D MNN_VULKAN=ON -D MNN_OPENCL=ON .. \
-D CMAKE_INSTALL_PREFIX=../install -D MNN_SEP_BUILD=OFF #-D MNN_OPENGL=ON
make install -j`nproc`
```

```diff
diff --git a/CMakeLists.txt b/CMakeLists.txt
index a5b42a7..a5294ea 100644
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -267,7 +267,7 @@ if(CMAKE_SYSTEM_NAME MATCHES "^Linux")
     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STRICT_ANSI__")
     if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
       # This is to workaround libgcc.a
-      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
+      # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
     endif()
     if(CMAKE_SYSTEM_PROCESSOR MATCHES "^armv7")
         add_definitions(-mfpu=neon)    #please define in project/cross-compile/arm.toolchain.cmake
diff --git a/source/backend/cpu/CPUFixedPoint.hpp b/source/backend/cpu/CPUFixedPoint.hpp
index a5c44f9..e7e8083 100644
--- a/source/backend/cpu/CPUFixedPoint.hpp
+++ b/source/backend/cpu/CPUFixedPoint.hpp
@@ -25,6 +25,7 @@ limitations under the License.
 #ifdef MNN_USE_NEON
 #include <arm_neon.h>
 #endif
+#include <cstdint>

 namespace MNN {
 // Part 1: Low-level integer-arithmetic primitives.
```
</details>

# TNN

<details>
<summary>Linux</summary>

```bash
#sudo apt install libprotoc-dev
#sudo apt install libomp-16-dev
# see https://github.com/YingkunZhou/EdgeTransformerPerf/wiki/tnn for more details
git clone https://github.com/Tencent/TNN.git # --depth=1
mkdir -p build && cd build
cmake -D CMAKE_BUILD_TYPE=Release \
-D CMAKE_SYSTEM_NAME=Linux \
-D CMAKE_C_COMPILER=clang-16 \
-D CMAKE_CXX_COMPILER=clang++-16 \
-D TNN_ARM_ENABLE=ON \
-D TNN_TEST_ENABLE=ON \
-D TNN_CPU_ENABLE=ON \
-D TNN_RK_NPU_ENABLE=OFF \
-D TNN_OPENMP_ENABLE=ON \
-D TNN_OPENCL_ENABLE=ON \
-D CMAKE_SYSTEM_PROCESSOR=aarch64 \
-D TNN_BUILD_SHARED=ON .. \
# -D TNN_CUDA_ENABLE=ON -D TNN_TENSORRT_ENABLE=ON

make -j`nproc`

mkdir -p ../install/include && mkdir -p ../install/lib
cp -a libTNN.so* ../install/lib
cp -r ../include/tnn ../install/include
```

```diff
diff --git a/source/tnn/device/cuda/CMakeLists.txt b/source/tnn/device/cuda/CMakeLists.txt
index 03dc534..9be59fb 100644
--- a/source/tnn/device/cuda/CMakeLists.txt
+++ b/source/tnn/device/cuda/CMakeLists.txt
@@ -16,6 +16,7 @@ include_directories($ENV{CUDNN_ROOT_DIR}/include)

 set(TARGET_ARCH "-gencode arch=compute_75,code=sm_75 \
                  -gencode arch=compute_70,code=sm_70 \
+                 -gencode arch=compute_87,code=sm_87 \
                  -gencode arch=compute_61,code=sm_61 \
                  -gencode arch=compute_60,code=sm_60 \
                  -gencode arch=compute_53,code=sm_53")
diff --git a/source/tnn/utils/data_type_utils.cc b/source/tnn/utils/data_type_utils.cc
index 1b11af6..febf16f 100644
--- a/source/tnn/utils/data_type_utils.cc
+++ b/source/tnn/utils/data_type_utils.cc
@@ -15,6 +15,7 @@
 #include "tnn/utils/data_type_utils.h"
 #include <limits.h>
 #include "tnn/core/macro.h"
+#include <cstdint>

 namespace TNN_NS {


```
</details>


# paddle lite

<details>
<summary>Linux</summary>

```bash
git clone https://github.com/PaddlePaddle/Paddle-Lite.git #--depth=1
cd Paddle-Lite
./lite/tools/build_linux.sh --arch=armv8 --with_extra=ON --toolchain=clang
```

```diff
diff --git a/lite/api/paddle_place.h b/lite/api/paddle_place.h
index c5757b8..abed5b0 100644
--- a/lite/api/paddle_place.h
+++ b/lite/api/paddle_place.h
@@ -15,6 +15,7 @@
 #pragma once
 #include <set>
 #include <string>
+#include <cstdint>

 // Generic helper definitions for shared library support
 #if defined _WIN32 || defined __CYGWIN__
diff --git a/lite/tools/build_linux.sh b/lite/tools/build_linux.sh
index ace7a8b..d3df143 100755
--- a/lite/tools/build_linux.sh
+++ b/lite/tools/build_linux.sh
@@ -100,7 +100,7 @@ WITH_BENCHMARK=OFF
 # use Arm DNN library instead of built-in math library, defaults to OFF.
 WITH_ARM_DNN_LIBRARY=OFF
 # num of threads used during compiling..
-readonly NUM_PROC=${LITE_BUILD_THREADS:-4}
+readonly NUM_PROC=32
 #####################################################################################################


@@ -344,9 +344,6 @@ function make_publish_so {
         build_dir=${build_dir}.kunlunxin_xpu
     fi

-    if [ -d $build_dir ]; then
-        rm -rf $build_dir
-    fi
     mkdir -p $build_dir
     cd $build_dir


```
</details>

# tensorflow lite

<details>
<summary>Linux</summary>

```bash
# use conda in order to use bazel. By the way, I dislike bazel
conda activate
conda install bazel==6.3.0 --yes
git clone https://github.com/google/flatbuffers.git #--depth=1
git clone https://github.com/tensorflow/tensorflow.git #--depth=1
export BASEDIR=$PWD
cd tensorflow
./configure
# choose clang, and use -O3 option
bazel build --verbose_failures -c opt //tensorflow/lite:tensorflowlite --define tflite_with_xnnpack=true # --jobs 8
bazel build --verbose_failures -c opt --config=monolithic tensorflow/lite/delegates/flex:tensorflowlite_flex --define tflite_with_xnnpack=true # --jobs 8
## optional
bazel build -c opt --config=monolithic tensorflow/lite/tools/benchmark:benchmark_model_plus_flex --jobs 8

mkdir -p install/include/tensorflow
cp -r tensorflow/lite install/include/tensorflow
cp -r tensorflow/core install/include/tensorflow # for armnn
cp -r $BASEDIR/flatbuffers/include/flatbuffers install/include
mkdir -p install/include/armnn
cp -r $BASEDIR/armnn/include  install/include/armnn
cp -r $BASEDIR/armnn/delegate install/include/armnn
find install/include/ ! \( -name '*.h*' \) -type f -exec rm -f {} +

mkdir -p install/lib
cp bazel-bin/tensorflow/lite/libtensorflowlite.so install/lib
cp bazel-bin/tensorflow/lite/delegates/flex/libtensorflowlite_flex.so install/lib
cp -a $BASEDIR/armnn/build/libarmnn.so* install/lib
cp -a $BASEDIR/armnn/build/delegate/libarmnnDelegate.so* install/lib

## optional
mkdir -p install/bin
cp bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model_plus_flex install/bin
```

```diff
diff --git a/third_party/xla/third_party/tsl/tsl/platform/denormal.cc b/third_party/xla/third_party/tsl/tsl/platform/denormal.cc
index 4f071109..02f64c09 100644
--- a/third_party/xla/third_party/tsl/tsl/platform/denormal.cc
+++ b/third_party/xla/third_party/tsl/tsl/platform/denormal.cc
@@ -17,6 +17,7 @@ limitations under the License.

 #include "tsl/platform/cpu_info.h"
 #include "tsl/platform/platform.h"
+#include <cstdint>

 // If we're on gcc 4.8 or older, there's a known bug that prevents the use of
 // intrinsics when the architecture is not defined in the flags. See
```
</details>

## armnn

<details>
<summary>Linux</summary>

```log
vim BUILD

cc_binary(
     name = "libtensorflow_lite_all.so",
     linkshared = 1,
     deps = [
         "//tensorflow/lite:framework",
         "//tensorflow/lite/kernels:builtin_ops",
     ],
)
```

```bash
bazel build --config=opt --config=monolithic --strip=always libtensorflow_lite_all.so
cd $BASEDIR/flatbuffers
mkdir build && cd build
cmake .. -D CMAKE_INSTALL_PREFIX=../install
make install -j32
cd $BASEDIR
git clone https://review.mlplatform.org/ml/ComputeLibrary #--depth=1
cd ComputeLibrary/
# git checkout <tag_name> # e.g. v20.11
# The machine used for this guide only has a Neon CPU which is why I only have "neon=1" but if
# your machine has an arm Gpu you can enable that by adding `opencl=1 embed_kernels=1 to the command below
scons arch=arm64-v8a neon=1 extra_cxx_flags="-fPIC" benchmark_tests=0 validation_tests=0 -j`nproc`
```

```diff
diff --git a/SConstruct b/SConstruct
index 68c518a..05dfe9f 100644
--- a/SConstruct
+++ b/SConstruct
@@ -381,7 +381,7 @@ if 'x86' not in env['arch']:
             auto_toolchain_prefix = "armv7l-tizen-linux-gnueabi-"
     elif env['estate'] == '64' and 'v8' in env['arch']:
         if env['os'] == 'linux':
-            auto_toolchain_prefix = "aarch64-linux-gnu-"
+            auto_toolchain_prefix = ""
         elif env['os'] == 'bare_metal':
             auto_toolchain_prefix = "aarch64-elf-"
         elif env['os'] == 'android':
```

```bash
cd $BASEDIR
git clone "https://review.mlplatform.org/ml/armnn" --depth=1
cd armnn
# git checkout <branch_name> # e.g. branches/armnn_20_11
mkdir build && cd build
# if you've got an arm Gpu add `-DARMCOMPUTECL=1` to the command below
cmake .. -DARMCOMPUTE_ROOT=$BASEDIR/ComputeLibrary \
         -DARMCOMPUTENEON=1 \
         -DBUILD_UNIT_TESTS=0 \
         -DBUILD_ARMNN_TFLITE_DELEGATE=1 \
         -DTENSORFLOW_ROOT=$BASEDIR/tensorflow \
         -DTFLITE_LIB_ROOT=$BASEDIR/tensorflow/bazel-bin \
         -DFLATBUFFERS_ROOT=$BASEDIR/flatbuffers/install \
         -D CMAKE_CXX_FLAGS="-Wno-error=missing-field-initializers -Wno-error=deprecated-declarations"
make -j32
```

```diff
diff --git a/src/armnn/ExecutionFrame.cpp b/src/armnn/ExecutionFrame.cpp
index 92a7990..118fa7e 100644
--- a/src/armnn/ExecutionFrame.cpp
+++ b/src/armnn/ExecutionFrame.cpp
@@ -39,7 +39,7 @@ void ExecutionFrame::RegisterDebugCallback(const DebugCallbackFunction& func)

 void ExecutionFrame::AddWorkloadToQueue(std::unique_ptr<IWorkload> workload)
 {
-    m_WorkloadQueue.push_back(move(workload));
+    m_WorkloadQueue.push_back(std::move(workload));
 }

 void ExecutionFrame::SetNextExecutionFrame(IExecutionFrame* nextExecutionFrame)
```
</details>

# onnxruntime

<details>
<summary>Linux</summary>

```bash
git clone https://github.com/microsoft/onnxruntime.git --depth=1
cd onnxruntime
#git submodule sync
git submodule update --init --recursive
# ./build.sh --config RelWithDebInfo --build_shared_lib --parallel --compile_no_warning_as_error --skip_tests
./build.sh --config Release --build_shared_lib --parallel --compile_no_warning_as_error --skip_tests
```
</details>


# torch

为什么编译pytorch环境编译器之类的这么依赖。。。。

<details>
<summary>Linux</summary>

```bash
git clone https://github.com/google/shaderc --depth=1
cd shaderc
./utils/git-sync-deps
# git clone https://github.com/KhronosGroup/glslang.git third_party/glslang
## https://github.com/KhronosGroup/glslang#2-check-out-external-projects
cd third_party/glslang
git checkout 0c400f67fcf305869c5fb113dd296eca266c9725
cd ../..
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$(pwd)/install" ..
make install -j32
```

https://github.com/conda-forge/pytorch-cpu-feedstock/blob/main/recipe/conda_build_config.yaml

-- MKL_THREADING = OMP
-- Check OMP with lib /lib/aarch64-linux-gnu/libomp.so and flags -fopenmp=libomp -v
-- MKL_THREADING = OMP
-- Check OMP with lib /lib/aarch64-linux-gnu/libomp.so and flags -fopenmp=libomp -v
-- Found OpenMP_C: -fopenmp=libomp
-- Found OpenMP_CXX: -fopenmp=libomp
-- Found OpenMP: TRUE
-- Adding OpenMP CXX_FLAGS: -fopenmp=libomp
-- Will link against OpenMP libraries: /lib/aarch64-linux-gnu/libomp.so

```bash
sudo apt install libomp-14-dev
cd /usr/lib/aarch64-linux-gnu
sudo ln -s ../llvm-14/lib/libomp.so libomp.so
# pytorch not adjust to clang-16 very much!!!
export CC=/usr/bin/clang-14
export CXX=/usr/bin/clang++-14
#if no ubuntu or no root:
#  wget https://github.com/llvm/llvm-project/releases/download/llvmorg-14.0.6/clang+llvm-14.0.6-aarch64-linux-gnu.tar.xz
#  tar xf clang+llvm-14.0.6-aarch64-linux-gnu.tar.xz
#  export CC=$PWD/clang+llvm-14.0.6-aarch64-linux-gnu/bin/clang
#  export CXX=$PWD/clang+llvm-14.0.6-aarch64-linux-gnu/bin/clang++
#  export LIBRARY_PATH=$PWD/clang+llvm-14.0.6-aarch64-linux-gnu/lib
#  export LD_LIBRARY_PATH=$PWD/clang+llvm-14.0.6-aarch64-linux-gnu/lib
conda create -n pytorch python=3.10 pip
conda activate pytorch
pip install pyyaml
pip install numpy # to enable USE_NUMPY by default
##########
# first you should know how to get pytorch easily
# conda install pytorch # will downlowd libopenblasp-r0.3.23.so which we will needed
pip install timm # will install pytorch and its dependency
##########
git clone https://github.com/pytorch/pytorch --depth=1
#git submodule sync
git submodule update --init --recursive
cd pytorch
python setup.py clean
#export PATH=$HOME/work/shaderc/build/install/bin:$PATH
#BUILD_BINARY=ON USE_OPENMP=1 USE_CUDA=0 USE_VULKAN=1 python setup.py bdist_wheel
BUILD_BINARY=ON USE_OPENMP=1 USE_CUDA=0 python setup.py bdist_wheel
```

print(*torch.__config__.show().split("\n"), sep="\n")

note:
1. use USE_OPENMP will get 2x performance
2. but unfortunately, the system openblas which installed by `apt install libopenblas-dev` is buggy!!!
```bash
#sudo rm /lib/aarch64-linux-gnu/libopenblas.so.0
#sudo ln -s $CMAKE_PREFIX_PATH/lib/libopenblasp-r0.3.23.so /lib/aarch64-linux-gnu/libopenblas.so.0
wget http://mirror.archlinuxarm.org/aarch64/extra/openblas-0.3.24-2-aarch64.pkg.tar.xz
tar xf openblas-0.3.24-2-aarch64.pkg.tar.xz
export LD_LIBRARY_PATH=$PWD/usr/lib
```

```bash
#LLVM_VERSION=14.0.6
#conda install clangxx==$LLVM_VERSION llvm-openmp==$LLVM_VERSION libclang==$LLVM_VERSION \
#  clangdev==$LLVM_VERSION llvm==$LLVM_VERSION llvm-dev==$LLVM_VERSION \
#  llvm-tools==$LLVM_VERSION libclang-cpp==$LLVM_VERSION \
#  libstdcxx-devel_linux-aarch64 --yes
```

https://github.com/aws/aws-graviton-getting-started/blob/main/machinelearning/pytorch.md

export ACL_ROOT_DIR=$HOME/work/ComputeLibrary

</details>
