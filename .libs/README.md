```bash
export CC=/usr/bin/clang-16
export CXX=/usr/bin/clang++-16
```

# opencv

just use apt install (Linux) or pkg install (Android)

<details>
<summary>~~Linux~~</summary>

- [OpenCV Basics - Others](https://wykvictor.github.io/2018/08/01/OpenCV-6.html)
- [rebuild your opencv4 from source with "add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)", have fun.](https://github.com/opencv/opencv/issues/13000#issuecomment-452150611)

```bash
git clone https://github.com/opencv/opencv.git --depth=1
cd opencv
mkdir build && cd build
cmake -D CMAKE_INSTALL_PREFIX=../install ..
make install -j`nproc`
cd ../install
vim opencv4.pc
export PKG_CONFIG_PATH=/opencv/install
```

`vim CMakeLists.txt`

```diff
diff --git a/CMakeLists.txt b/CMakeLists.txt
index 40d80e1..c7019c1 100644
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -554,6 +554,7 @@ if(ENABLE_IMPL_COLLECTION)
   add_definitions(-DCV_COLLECT_IMPL_DATA)
 endif()

+add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
 if(OPENCV_DISABLE_FILESYSTEM_SUPPORT)
   add_definitions(-DOPENCV_HAVE_FILESYSTEM_SUPPORT=0)
 endif()

```

```log
# Package Information for pkg-config

Name: OpenCV
Description: Open Source Computer Vision Library
Version: 4.8.0
Libs: -L/opencv/install/lib -lopencv_imgproc -lopencv_imgcodecs -lopencv_core -lopencv_dnn
Libs.private: -ldl -lm -lpthread -lrt
Cflags: -I/opencv/install/include/opencv4
```
</details>

# ncnn

<details>
<summary>Linux</summary>

```bash
# will use openmp lib to enable multi-threads
sudo apt install libomp-16-dev
git clone https://github.com/Tencent/ncnn.git #--depth=1
cd ncnn
#git submodule sync
git submodule update --init --recursive
mkdir -p build && cd build
export LDFLAGS="-L/usr/lib/llvm-16/lib"
export CPPFLAGS="-I/usr/lib/llvm-16/include"
/usr/bin/cmake -D NCNN_SHARED_LIB=ON -D NCNN_VULKAN=ON .. -D CMAKE_BUILD_TYPE=Release \
-D CMAKE_INSTALL_PREFIX=../install -D NCNN_BUILD_BENCHMARK=OFF
make install -j`nproc`
```

```bash
# conda activate # use conda env
cd tools/pnnx
# pip install torch
# remove protobuf & libprotobuf package
mkdir build && cd build
cmake ..
make -j`nproc`
```
</details>

<details>
<summary>Android</summary>

```bash
export ANDROID_NDK=$PWD/android-ndk-r22b
mkdir build && cd build
cmake -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="arm64-v8a" \
    -DANDROID_PLATFORM=android-24 -DNCNN_VULKAN=ON .. \
    -D CMAKE_INSTALL_PREFIX=../install -D NCNN_SHARED_LIB=ON

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
-D CMAKE_INSTALL_PREFIX=../install -D MNN_SEP_BUILD=OFF -D MNN_ARM82=ON #-D MNN_OPENGL=ON
make install -j`nproc`

-D MNN_BUILD_CONVERTER=ON
-D MNN_BUILD_BENCHMARK=ON
-D MNN_BUILD_DEMO=ON
-D MNN_BUILD_QUANTOOLS=ON
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

<details>
<summary>Android</summary>

```bash
## way 1: native build
#pkg install mesa-dev # for opengl
cmake -D CMAKE_BUILD_TYPE=Release -D MNN_USE_LOGCAT=false -D MNN_VULKAN=ON -D MNN_OPENCL=ON .. \
-D CMAKE_INSTALL_PREFIX=../install -DMNN_BUILD_FOR_ANDROID_COMMAND=true -DNATIVE_LIBRARY_OUTPUT=. -DNATIVE_INCLUDE_OUTPUT=.  -D MNN_SEP_BUILD=OFF -D MNN_ARM82=ON #-D MNN_OPENGL=ON
make install -j`nproc`
## way 2: cross build
cd project/android
vim build_64.sh
#######################################################
#!/bin/bash
cmake ../../../ \
-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
-DCMAKE_BUILD_TYPE=Release \
-DANDROID_ABI="arm64-v8a" \
-DMNN_USE_LOGCAT=false \
-DANDROID_PLATFORM=android-24  \
-DMNN_BUILD_FOR_ANDROID_COMMAND=true \
-D MNN_OPENCL=ON -D MNN_VULKAN=ON -D MNN_ARM82=ON \
-D MNN_SEP_BUILD=OFF -D CMAKE_INSTALL_PREFIX=../install \
-DNATIVE_LIBRARY_OUTPUT=. -DNATIVE_INCLUDE_OUTPUT=.

make install -j32
#######################################################
export ANDROID_NDK=<prefix>/android-ndk-r22b
mkdir build && cd build && ../build_64.sh
```
</details>

# TNN

<details>
<summary>Linux</summary>

```bash
git clone https://github.com/Tencent/TNN.git # --depth=1

sudo apt install protobuf-compiler
sudo apt install libprotoc-dev
sudo apt install libomp-16-dev # also for runtime
export LDFLAGS="-L/usr/lib/llvm-16/lib"
export CPPFLAGS="-I/usr/lib/llvm-16/include"
# see https://github.com/YingkunZhou/EdgeTransformerPerf/wiki/tnn for more details
mkdir -p build && cd build
cmake -D CMAKE_BUILD_TYPE=Release \
-D CMAKE_SYSTEM_NAME=Linux \
-D CMAKE_C_COMPILER=clang-16 \
-D CMAKE_CXX_COMPILER=clang++-16 \
-D TNN_ARM_ENABLE=ON \
-D TNN_ARM82_ENABLE=ON \
-D TNN_TEST_ENABLE=ON \
-D TNN_CPU_ENABLE=ON \
-D TNN_RK_NPU_ENABLE=OFF \
-D TNN_OPENMP_ENABLE=ON \
-D TNN_OPENCL_ENABLE=ON \
-D CMAKE_SYSTEM_PROCESSOR=aarch64 \
-D CMAKE_INSTALL_PREFIX=../install \
-D TNN_BUILD_SHARED=ON .. \
# -D TNN_CUDA_ENABLE=ON -D TNN_TENSORRT_ENABLE=ON

make -j`nproc`

mkdir -p ../install/include && mkdir -p ../install/lib
cp -a libTNN.so* ../install/lib
cp -r ../include/tnn ../install/include
---

cd TNN/tools/convert2tnn
./build.sh
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

```diff
diff --git a/third_party/flatbuffers/src/idl_gen_rust.cpp b/third_party/flatbuffers/src/idl_gen_rust.cpp
index 455780cd..6082a02a 100644
--- a/third_party/flatbuffers/src/idl_gen_rust.cpp
+++ b/third_party/flatbuffers/src/idl_gen_rust.cpp
@@ -496,7 +496,6 @@ class RustGenerator : public BaseGenerator {
     // example: f(A, D::E)          -> super::D::E
     // does not include leaf object (typically a struct type).

-    size_t i = 0;
     std::stringstream stream;

     auto s = src->components.begin();
@@ -507,7 +506,6 @@ class RustGenerator : public BaseGenerator {
       if (*s != *d) { break; }
       ++s;
       ++d;
-      ++i;
     }

     for (; s != src->components.end(); ++s) { stream << "super::"; }
diff --git a/tools/converter/source/onnx/onnx_utils.h b/tools/converter/source/onnx/onnx_utils.h
index 27f42bed..403960eb 100644
--- a/tools/converter/source/onnx/onnx_utils.h
+++ b/tools/converter/source/onnx/onnx_utils.h
@@ -17,6 +17,7 @@

 #include <cassert>
 #include <vector>
+#include <cmath>

 #include "onnx.pb.h"
 #include "onnx_proxy_graph.h"
diff --git a/tools/dynamic_range_quantization/utils.h b/tools/dynamic_range_quantization/utils.h
index 3de8d35d..0574b318 100644
--- a/tools/dynamic_range_quantization/utils.h
+++ b/tools/dynamic_range_quantization/utils.h
@@ -13,6 +13,7 @@
 // specific language governing permissions and limitations under the License.
 #include "tnn/core/macro.h"
 #include "tnn/interpreter/raw_buffer.h"
+#include <cmath>

 namespace TNN_NS {

diff --git a/tools/onnx2tnn/src/core/onnx_fuse/onnx2tnn_fuse_gelu.cc b/tools/onnx2tnn/src/core/onnx_fuse/onnx2tnn_fuse_gelu.cc
index 04f888eb..b3716387 100644
--- a/tools/onnx2tnn/src/core/onnx_fuse/onnx2tnn_fuse_gelu.cc
+++ b/tools/onnx2tnn/src/core/onnx_fuse/onnx2tnn_fuse_gelu.cc
@@ -13,6 +13,7 @@
 // specific language governing permissions and limitations under the License.

 #include <algorithm>
+#include <cmath>

 #include "onnx2tnn.h"

```
</details>

<details>
<summary>Android</summary>

```bash
mkdir build && cd build
cmake \
-D CMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
-D ANDROID_ABI="arm64-v8a" \
-D ANDROID_PLATFORM=android-24 \
-D BUILD_FOR_ANDROID_COMMAND=true \
-D TNN_ARM_ENABLE=ON \
-D TNN_ARM82_ENABLE=ON \
-D TNN_TEST_ENABLE=ON \
-D TNN_CPU_ENABLE=ON \
-D TNN_RK_NPU_ENABLE=OFF \
-D TNN_OPENMP_ENABLE=ON \
-D TNN_OPENCL_ENABLE=ON \
-D CMAKE_SYSTEM_PROCESSOR=aarch64 \
-D CMAKE_INSTALL_PREFIX=../install \
-D TNN_BUILD_SHARED=ON ..

make -j`nproc`

mkdir -p ../install/include && mkdir -p ../install/lib
cp -a libTNN.so* ../install/lib
cp -r ../include/tnn ../install/include
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
cd build/Linux/Release
## TO keep the directory structure the same as github release tar pacakge
DESTDIR=../install make install -j`nproc`
cd ../install
mv usr/local/include/onnxruntime/ include
mv usr/local/lib .
```
</details>


# torch

pytorch 实在是太TM复杂了！！！而且还强烈依赖openblas库，对性能的影响非常敏感！！！

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
#BUILD_BINARY=ON BUILD_TEST=0 USE_CUDA=0 USE_VULKAN=1 python setup.py bdist_wheel
BUILD_BINARY=ON BUILD_TEST=0 USE_CUDA=0 python setup.py bdist_wheel
```

```python
import torch
print(*torch.__config__.show().split("\n"), sep="\n")
```

note:
1. use BLAS lib will get 2x performance
2. but unfortunately, the system openblas which installed by `apt install libopenblas-dev` is buggy!!!
```bash
wget http://mirror.archlinuxarm.org/aarch64/extra/openblas-0.3.24-2-aarch64.pkg.tar.xz
tar xf openblas-0.3.24-2-aarch64.pkg.tar.xz
export LD_LIBRARY_PATH=$PWD/usr/lib
```
3. here we use libopenblas.so which contains in [torch-2.1.0.dev20230825-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl](https://github.com/YingkunZhou/EdgeTransformerPerf/releases/download/v0.0/torch-2.1.0.dev20230825-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl)


<details>
<summary>
don't read below
</summary>

~~use conda compiler toolchain~~

```bash
LLVM_VERSION=14.0.6
conda install clangxx==$LLVM_VERSION llvm-openmp==$LLVM_VERSION libclang==$LLVM_VERSION \
  clangdev==$LLVM_VERSION llvm==$LLVM_VERSION llvmdev==$LLVM_VERSION \
  llvm-tools==$LLVM_VERSION libclang-cpp==$LLVM_VERSION \
  gxx==10.3.0 scons --yes # here we use gcc-10.3.0 to build acl

# conda install numactl
export LD_LIBRARY_PATH=$HOME/miniforge3/envs/pytorch/lib
export CPLUS_INCLUDE_PATH=$HOME/miniforge3/envs/pytorch/include
```
</details>

## build with ACL acc

```bash
# https://github.com/aws/aws-graviton-getting-started/blob/main/machinelearning/pytorch.md

export ACL_ROOT_DIR=$HOME/work/ComputeLibrary
export USE_MKLDNN=ON USE_MKLDNN_ACL=ON USE_CUDA=0 BUILD_TEST=0
python setup.py bdist_wheel
```
- https://github.com/aws/aws-graviton-getting-started/blob/main/machinelearning/pytorch.md
- **[the offical methods we choose](https://github.com/pytorch/builder/blob/main/aarch64_linux/build_aarch64_wheel.py)**
- https://github.com/pytorch/pytorch/issues/51039
- https://hub.docker.com/r/armswdev/pytorch-arm-neoverse
- https://github.com/pytorch/xla/blob/master/scripts/build_torch_wheels.sh
- [As for why I want to know this, I want to compile pytorch in the source code to link my self-installed ACLs and find that it is much slower than the torch installed using pip, under the same version of torch.](https://github.com/pytorch/pytorch/issues/97421)
- https://download.pytorch.org/whl/nightly/torch/


- https://github.com/ARM-software/Tool-Solutions/tree/main/docker/pytorch-aarch64
- [Docker必备六大国内镜像](https://segmentfault.com/a/1190000023117518)
https://cr.console.aliyun.com/cn-hangzhou/instances/mirrors


```json
# cat /etc/docker/daemon.json
{
    "bip": "172.18.0.1/16",
    "registry-mirrors": [
        "https://xxx.mirror.aliyuncs.com"
    ]
}
```

export https_proxy=http://xxx:xxx
export http_proxy=http://xxx:xxx

- [Setup the proxy for Dockerfile building](https://dev.to/zyfa/setup-the-proxy-for-dockerfile-building--4jc8)
```diff
diff --git a/docker/pytorch-aarch64/Dockerfile b/docker/pytorch-aarch64/Dockerfile
index 78334c6..5484033 100644
--- a/docker/pytorch-aarch64/Dockerfile
+++ b/docker/pytorch-aarch64/Dockerfile
@@ -25,6 +25,8 @@ ARG default_py_version=3.10
 FROM ubuntu:22.04 AS pytorch-base
 ARG default_py_version
 ENV PY_VERSION="${default_py_version}"
+ENV http_proxy http://xxx:xxx
+ENV https_proxy http://xxx:xxx

 RUN if ! [ "$(arch)" = "aarch64" ] ; then exit 1; fi

```

**we finally use [aarch64_ci_build.sh](https://github.com/pytorch/builder/blob/main/aarch64_linux/aarch64_ci_build.sh) methods to build pytorch**

```dockerfile
ARG default_py_version=3.8

FROM ubuntu:20.04
ARG default_py_version
ENV PY_VERSION="${default_py_version}"

RUN if ! [ "$(arch)" = "aarch64" ] ; then exit 1; fi

ENV TZ=Asia/Shanghai \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update
RUN apt-get -y upgrade
# Install core OS packages
RUN apt-get -y install \
      zsh \
      wget \
      accountsservice \
      apport \
      at \
      autoconf \
      bc \
      build-essential \
      cmake \
      cpufrequtils \
      curl \
      ethtool \
      g++-10 \
      gcc-10 \
      gettext-base \
      gfortran-10 \
      git \
      iproute2 \
      iputils-ping \
      lxd \
      libbz2-dev \
      libc++-dev \
      libcgal-dev \
      libffi-dev \
      libfreetype6-dev \
      libhdf5-dev \
      libjpeg-dev \
      liblzma-dev \
      libncurses5-dev \
      libncursesw5-dev \
      libpng-dev \
      libreadline-dev \
      libsox-fmt-all \
      libsqlite3-dev \
      libssl-dev \
      libxml2-dev \
      libxslt-dev \
      locales \
      lsb-release \
      lvm2 \
      moreutils \
      net-tools \
      open-iscsi \
      openjdk-8-jdk \
      openssl \
      pciutils \
      policykit-1 \
      python${PY_VERSION} \
      python${PY_VERSION}-dev \
      python${PY_VERSION}-distutils \
      python${PY_VERSION}-venv \
      python3-pip \
      python-openssl \
      rsync \
      rsyslog \
      snapd \
      scons \
      sox \
      ssh \
      sudo \
      time \
      udev \
      unzip \
      ufw \
      uuid-runtime \
      vim \
      xz-utils \
      zip \
      zlib1g-dev

# Set default gcc, python and pip versions
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 1 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 1 && \
    update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1
```

```bash
docker build . -f Dockerfile -t xxx
docker run --name pytorch --hostname pytorch -v xxx:/xxx -it xxx bash
docker start pytorch
docker exec -it pytorch zsh
update-alternatives --config gcc
update-alternatives --config g++
# openblas use clang-14, onednn+acl use gcc-10
# export CMAKE_BUILD_PARALLEL_LEVEL=4 # set thread number to build pytorch
# cd /usr/lib/aarch64-linux-gnu
# rm libgomp.so.1; ln -s ../llvm-14/lib/libomp.so.5 libgomp.so.1

# https://github.com/pytorch/pytorch/issues/29327
export USE_QNNPACK=ON
export USE_PYTORCH_QNNPACK=ON

DESIRED_PYTHON="3.8" ./aarch64_ci_build.sh
```

google search: Didn't find engine for operation quantized::conv2d_prepack NoQEngine
- https://github.com/pytorch/pytorch/issues/29327
- https://github.com/pytorch/pytorch/issues/76755

```diff
diff --git a/aten/src/ATen/Context.cpp b/aten/src/ATen/Context.cpp
index 1ec545d..63675a5 100644
--- a/aten/src/ATen/Context.cpp
+++ b/aten/src/ATen/Context.cpp
@@ -286,7 +286,7 @@ bool Context::hasLAPACK() {
 at::QEngine Context::qEngine() const {
   static auto _quantized_engine = []() {
     at::QEngine qengine = at::kNoQEngine;
-#if defined(C10_MOBILE) && defined(USE_PYTORCH_QNNPACK)
+#if defined(USE_PYTORCH_QNNPACK)
     qengine = at::kQNNPACK;
 #endif


```
</details>
