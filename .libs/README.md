```bash
export CC=/usr/bin/clang-16
export CXX=/usr/bin/clang++-16
```

<details>
<summary>Android cross-compile details</summary>

```bash
# 6. (可选) 删除 debug 编译参数，减小二进制体积 参照 https://github.com/android/ndk/issues/243
# 用编辑器打开 $ANDROID_NDK/build/cmake/android.toolchain.cmake 删除 "-g" 这行
# 或者打开 $ANDROID_NDK/build/cmake/android-legacy.toolchain.cmake (Android NDK r23c及以上版本) 执行同样的操作
list(APPEND ANDROID_COMPILER_FLAGS
  -g
  -DANDROID
  ...
```
</details>

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
-D MNN_OPENCL=ON -D MNN_VULKAN=ON -D MNN_ARM82=ON -D MNN_NNAPI=ON\
-D MNN_SEP_BUILD=OFF -D CMAKE_INSTALL_PREFIX=../install \
-DNATIVE_LIBRARY_OUTPUT=. -DNATIVE_INCLUDE_OUTPUT=.

-D MNN_BUILD_BENCHMARK=ON

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

https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/docs/demo_guides/opencl.md

Lite预测库分为**基础预测库**和**全量预测库(with_extra)**：基础预测库只包含基础CV算子（OP），体积较小；全量预测库包含所有Lite算子，体积较大，支持模型较多。

编译时由编译选项 `build_extra`(默认为OFF)控制，`--build_extra=OFF`时编译**基础预测库**，`--build_extra=ON`时编译**全量预测库**。

<details>
<summary>Linux</summary>

build_linux.sh: NUM_PROC=32

```bash
git clone https://github.com/PaddlePaddle/Paddle-Lite.git #--depth=1
cd Paddle-Lite
./lite/tools/build_linux.sh --arch=armv8 --with_extra=ON --toolchain=clang \
--with_exception=ON --with_opencl=ON --with_arm82_fp16=ON
### for cortex-a73 and below
./lite/tools/build_linux.sh --arch=armv8 --with_extra=ON --toolchain=clang \
--with_exception=ON --with_opencl=ON #--with_arm82_fp16=ON
```

```diff
--- a/lite/tools/build_linux.sh
+++ b/lite/tools/build_linux.sh
@@ -344,9 +344,6 @@ function make_publish_so {
         build_dir=${build_dir}.kunlunxin_xpu
     fi

-    if [ -d $build_dir ]; then
-        rm -rf $build_dir
-    fi
     mkdir -p $build_dir
     cd $build_dir
```

[how to convert model](https://github.com/YingkunZhou/EdgeTransformerPerf/wiki/paddlelite#how-to-convert-model)

```bash
./lite/tools/build.sh build_optimize_tool
```

</details>

<details>
<summary>Android</summary>

```bash
export NDK_ROOT=$PWD/android-ndk-r22b
./lite/tools/build_android.sh --arch=armv8 --with_extra=ON --toolchain=clang \
--with_exception=ON --with_opencl=ON --with_java=OFF --android_api_level=24 --with_arm82_fp16=ON
```

build_android.sh: NUM_PROC=32

</details>

# tensorflow lite

python 3.10 conda install bazel

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
bazel build --verbose_failures -c opt //tensorflow/lite:tensorflowlite --define tflite_with_xnnpack=true --define tflite_with_xnnpack_qs8=true # --jobs 8
bazel build --verbose_failures -c opt --config=monolithic tensorflow/lite/delegates/flex:tensorflowlite_flex --define tflite_with_xnnpack=true --define tflite_with_xnnpack_qs8=true # --jobs 8

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
#armnn
cp bazel-bin/libtensorflow_lite_all.so  install/lib
cp -a $BASEDIR/armnn/build/libarmnn.so* install/lib
cp -a $BASEDIR/armnn/build/delegate/libarmnnDelegate.so*  install/lib
#flatbuffer
cp -a $BASEDIR/flatbuffers/install/lib/libflatbuffers.so* install/lib

## gpu support
# sudo apt install libgles2-mesa-dev libegl1-mesa-dev xorg-dev
bazel build -s -c opt --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so
cp bazel-bin/tensorflow/lite/delegates/gpu/libtensorflowlite_gpu_delegate.so install/lib

## optional
bazel build -c opt --config=monolithic tensorflow/lite/tools/benchmark:benchmark_model_plus_flex --jobs 8
mkdir -p install/bin
cp bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model_plus_flex install/bin
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
cmake .. -D CMAKE_INSTALL_PREFIX=../install -D FLATBUFFERS_BUILD_SHAREDLIB=ON
make install -j32
cd $BASEDIR
git clone https://review.mlplatform.org/ml/ComputeLibrary #--depth=1
cd ComputeLibrary/
# git checkout <tag_name> # e.g. v20.11
# The machine used for this guide only has a Neon CPU which is why I only have "neon=1" but if
# your machine has an arm Gpu you can enable that by adding `opencl=1 embed_kernels=1 to the command below
scons arch=arm64-v8a neon=1 extra_cxx_flags="-fPIC" benchmark_tests=0 validation_tests=0 -j 32
scons arch=arm64-v8.2-a neon=1 extra_cxx_flags="-fPIC" benchmark_tests=0 validation_tests=0 -j 32
scons arch=arm64-v8a neon=1 opencl=1 embed_kernels=1 extra_cxx_flags="-fPIC" benchmark_tests=0 validation_tests=0 -j 32
scons arch=arm64-v8.2-a neon=1 opencl=1 embed_kernels=1 extra_cxx_flags="-fPIC" benchmark_tests=0 validation_tests=0 -j 32
```

Here we use arm linux env natively.

```diff
--- a/SConstruct
+++ b/SConstruct
@@ -373,7 +373,7 @@ else: # NONE "multi_isa" builds
-if 'x86' not in env['arch']:
+if 'arm' not in env['arch']:
     if env['estate'] == '32':
         if env['os'] == 'linux':
             auto_toolchain_prefix = "arm-linux-gnueabihf-" if 'v7' in env['arch'] else "armv8l-linux-gnueabihf-"
```

```bash
conda activate # bazel env will give java support!
cd $BASEDIR
git clone "https://review.mlplatform.org/ml/armnn" --depth=1
cd armnn
# git checkout <branch_name> # e.g. branches/armnn_20_11
mkdir build && cd build
# if you've got an arm Gpu add `-DARMCOMPUTECL=1` to the command below
cmake .. -DARMCOMPUTE_ROOT=$BASEDIR/ComputeLibrary \
         -DBUILD_UNIT_TESTS=0 \
         -DBUILD_ARMNN_TFLITE_DELEGATE=1 \
         -DTENSORFLOW_ROOT=$BASEDIR/tensorflow \
         -DTFLITE_LIB_ROOT=$BASEDIR/tensorflow/bazel-bin \
         -DFLATBUFFERS_ROOT=$BASEDIR/flatbuffers/install \
         -DCMAKE_CXX_FLAGS="-Wno-error=missing-field-initializers -Wno-error=deprecated-declarations" \
         -DARMCOMPUTENEON=1 -DARMCOMPUTECL=1
make -j32
```

</details>


<details>
<summary>Android</summary>

For armnn, pre-download all repos latest

```diff
diff --git a/scripts/build_android_ndk_guide.sh b/scripts/build_android_ndk_guide.sh
index a364d4d..d260528 100755
--- a/scripts/build_android_ndk_guide.sh
+++ b/scripts/build_android_ndk_guide.sh
@@ -110,14 +110,14 @@ function GetAndBuildCmake319 {
 function GetAndBuildFlatbuffers {
     cd $WORKING_DIR

-    if [[ ! -d flatbuffers-23.5.26 ]]; then
+    if [[ ! -d flatbuffers ]]; then
         echo "+++ Getting Flatbuffers"
         wget https://github.com/google/flatbuffers/archive/v23.5.26.tar.gz
         tar xf v23.5.26.tar.gz
     fi
     #Build FlatBuffers
     echo "+++ Building x86 Flatbuffers library"
-    cd $WORKING_DIR/flatbuffers-23.5.26
+    cd $WORKING_DIR/flatbuffers

     rm -f CMakeCache.txt

@@ -135,7 +135,7 @@ function GetAndBuildFlatbuffers {
     make all install -j16

     echo "+++ Building Android Flatbuffers library"
-    cd $WORKING_DIR/flatbuffers-23.5.26
+    cd $WORKING_DIR/flatbuffers

     rm -f CMakeCache.txt

@@ -211,7 +211,7 @@ function GetAndBuildComputeLibrary {
     cd $WORKING_DIR/ComputeLibrary

     echo "+++ Building Compute Library"
-    scons toolchain_prefix=llvm- compiler_prefix=aarch64-linux-android$ANDROID_API- arch=arm64-v8a neon=$ACL_NEON opencl=$ACL_CL embed_kernels=$ACL_CL extra_cxx_flags="-fPIC" \
+    scons toolchain_prefix=llvm- compiler_prefix=aarch64-linux-android$ANDROID_API- arch=arm64-v8.2-a neon=$ACL_NEON opencl=$ACL_CL embed_kernels=$ACL_CL extra_cxx_flags="-fPIC" \
     benchmark_tests=0 validation_tests=0 os=android -j16
 }

```

```bash
./armnn/scripts/build_android_ndk_guide.sh
```

For tflite

```bash
# prepare ndk
wget https://dl.google.com/android/repository/android-ndk-r25-linux.zip
unzip android-ndk-r25-linux.zip
wget https://mirrors.cloud.tencent.com/AndroidSDK/commandlinetools-linux-8512546_latest.zip
unzip commandlinetools-linux-8512546_latest.zip
mkdir android-sdk && cd android-sdk
mkdir cmdline-tools
mv ../cmdline-tools/ cmdline-tools/latest
./cmdline-tools/latest/bin/sdkmanager "platform-tools" "platforms;android-33" "build-tools;34.0.0"
# prepare sdk
cd tensorflow
./configure
# compiler use gcc, if use clang, some normal headers cannot find!!!
# api choose 30

## build xnnpack
bazel build --verbose_failures -c opt --config=android_arm64 //tensorflow/lite:tensorflowlite --define tflite_with_xnnpack=true --define tflite_with_xnnpack_qs8=true
bazel build --verbose_failures -c opt --config=android_arm64 --config=monolithic tensorflow/lite/delegates/flex:tensorflowlite_flex --define tflite_with_xnnpack=true --define tflite_with_xnnpack_qs8=true

## build gpu
bazel build -c opt --config=android_arm64 tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so

## build nnapi
bazel build -c opt --config=android_arm64 //tensorflow/lite/nnapi:nnapi_util
bazel build -c opt --config=android_arm64 //tensorflow/lite/nnapi:nnapi_implementation
bazel build -c opt --config=android_arm64 //tensorflow/lite/delegates/nnapi:nnapi_delegate_no_nnapi_implementation

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
cp bazel-bin/tensorflow/lite/delegates/gpu/libtensorflowlite_gpu_delegate.so install/lib
cp bazel-bin/tensorflow/lite/nnapi/libnnapi_implementation.so install/lib
cp bazel-bin/tensorflow/lite/nnapi/libnnapi_util.so install/lib
cp bazel-bin/tensorflow/lite/delegates/nnapi/libnnapi_delegate_no_nnapi_implementation.so install/lib

#armnn
cp -a $BASEDIR/armnn/build/libarmnn.so* install/lib
cp -a $BASEDIR/armnn/build/delegate/libarmnnDelegate.so*  install/lib

```

</details>

# onnxruntime

<details>
<summary>Linux</summary>

```bash
export BASEDIR=$PWD
git clone https://github.com/microsoft/onnxruntime.git --depth=1
cd onnxruntime
#git submodule sync
git submodule update --init --recursive
./build.sh --config Release --use_xnnpack --build_shared_lib --parallel --compile_no_warning_as_error --skip_tests # --use_dnnl
cd build/Linux/Release
## TO keep the directory structure the same as github release tar pacakge
DESTDIR=../onnxruntime make install -j`nproc`
cd ../onnxruntime
mv usr/local/include/onnxruntime/ include
mv usr/local/lib .
rm -rf usr

## onednn
# cp ../../../include/onnxruntime/core/providers/dnnl/dnnl_provider_options.h include
# cp -a ../Release/dnnl/install/* .

## acl
cd $BASEDIR
git clone https://github.com/ARM-software/ComputeLibrary.git -b v23.08 --depth 1 --shallow-submodules
scons arch=arm64-v8.2-a neon=1 extra_cxx_flags="-fPIC" benchmark_tests=0 validation_tests=0 -j 32 debug=0
cd -
./build.sh --config Release --use_xnnpack --use_acl --acl_home $BASEDIR/ComputeLibrary --acl_libs $BASEDIR/ComputeLibrary/build --build_shared_lib --parallel --compile_no_warning_as_error --skip_tests
cp -a ../../../../ComputeLibrary/build/*.so lib
cp ../../../include/onnxruntime/core/providers/acl/acl_provider_factory.h include
```

```diff
diff --git a/cmake/CMakeLists.txt b/cmake/CMakeLists.txt
index 34e7687..282123a 100644
--- a/cmake/CMakeLists.txt
+++ b/cmake/CMakeLists.txt
@@ -1134,18 +1134,13 @@ if (onnxruntime_USE_ACL OR onnxruntime_USE_ACL_1902 OR onnxruntime_USE_ACL_1905
         IMPORTED_NO_SONAME 1
         IMPORTED_LOCATION "${onnxruntime_ACL_LIBS}/libarm_compute.so")

-    add_library(arm_compute_core SHARED IMPORTED)
-    set_target_properties(arm_compute_core PROPERTIES
-        IMPORTED_NO_SONAME 1
-        IMPORTED_LOCATION "${onnxruntime_ACL_LIBS}/libarm_compute_core.so")
-
     add_library(arm_compute_graph SHARED IMPORTED)
     set_target_properties(arm_compute_graph PROPERTIES
         IMPORTED_NO_SONAME 1
         IMPORTED_LOCATION "${onnxruntime_ACL_LIBS}/libarm_compute_graph.so")
   endif()

-  list(APPEND onnxruntime_EXTERNAL_LIBRARIES arm_compute arm_compute_core arm_compute_graph)
+  list(APPEND onnxruntime_EXTERNAL_LIBRARIES arm_compute arm_compute_graph)

 endif()

@@ -1164,11 +1159,6 @@ if (onnxruntime_USE_ARMNN)
         IMPORTED_NO_SONAME 1
         IMPORTED_LOCATION "${onnxruntime_ACL_LIBS}/libarm_compute.so")

-    add_library(arm_compute_core SHARED IMPORTED)
-    set_target_properties(arm_compute_core PROPERTIES
-        IMPORTED_NO_SONAME 1
-        IMPORTED_LOCATION "${onnxruntime_ACL_LIBS}/libarm_compute_core.so")
-
     add_library(arm_compute_graph SHARED IMPORTED)
     set_target_properties(arm_compute_graph PROPERTIES
         IMPORTED_NO_SONAME 1
@@ -1182,7 +1172,7 @@ if (onnxruntime_USE_ARMNN)
         IMPORTED_LOCATION "${onnxruntime_ARMNN_LIBS}/libarmnn.so")
   endif()

-  list(APPEND onnxruntime_EXTERNAL_LIBRARIES armnn arm_compute arm_compute_core arm_compute_graph)
+  list(APPEND onnxruntime_EXTERNAL_LIBRARIES armnn arm_compute arm_compute_graph)
 endif()

 if (onnxruntime_USE_DNNL)
diff --git a/tools/ci_build/build.py b/tools/ci_build/build.py
index b2040b2..691b948 100644
--- a/tools/ci_build/build.py
+++ b/tools/ci_build/build.py
@@ -638,8 +638,8 @@ def parse_arguments():
     parser.add_argument(
         "--use_acl",
         nargs="?",
-        const="ACL_1905",
-        choices=["ACL_1902", "ACL_1905", "ACL_1908", "ACL_2002", "ACL_2308"],
+        const="ACL_2308",
+        choices=["ACL_2308"],
         help="Build with ACL for ARM architectures.",
     )
     parser.add_argument("--acl_home", help="Path to ACL home dir")
```

</details>


<details>
<summary>Android</summary>

```bash
export BASEDIR=$PWD
export ANDROID_NDK=$BASEDIR/android-ndk-r26b
export ANDROID_SDK=$BASEDIR/android-sdk
cd onnxruntime
./build.sh --use_nnapi --use_xnnpack --use_acl --acl_home $BASEDIR/ComputeLibrary --acl_libs $BASEDIR/ComputeLibrary/build --use_qnn --qnn_home /opt/qcom --config Release --android --android_sdk_path $ANDROID_SDK --android_ndk_path $ANDROID_NDK --android_abi arm64-v8a --android_api 30 --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync --skip_tests
cd build/Android/Release
## TO keep the directory structure the same as github release tar pacakge
DESTDIR=../onnxruntime make install -j`nproc`
cd ../onnxruntime
mv usr/local/include/onnxruntime/ include
mv usr/local/lib .
rm -rf usr
cp -a /opt/qcom/aistack/qnn/2.19.0.240124/lib/aarch64-android/* lib
cp -a ../../../../ComputeLibrary/build/*.so lib
cp ../../../include/onnxruntime/core/providers/acl/acl_provider_factory.h include
cp ../../../include/onnxruntime/core/providers/nnapi/nnapi_provider_factory.h include
```

</details>

# torch

<details>
<summary>Linux</summary>

**we finally use [aarch64_ci_build.sh](https://github.com/pytorch/builder/blob/main/aarch64_linux/aarch64_ci_build.sh) methods to build pytorch**

```dockerfile
FROM ubuntu:20.04
ARG default_py_version=3.8
ENV PY_VERSION="${default_py_version}"

RUN if ! [ "$(arch)" = "aarch64" ] ; then exit 1; fi

ENV TZ=Asia/Shanghai \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get -y install \
     accountsservice apport at autoconf bc build-essential cpufrequtils curl ethtool \
     g++-10 gcc-10 gettext-base gfortran-10 git iproute2 iputils-ping lxd libbz2-dev \
     libc++-dev libcgal-dev libffi-dev libfreetype6-dev libhdf5-dev libjpeg-dev \
     liblzma-dev libncurses5-dev libncursesw5-dev libopenblas-dev libopencv-dev libpng-dev \
     libreadline-dev libsox-fmt-all libsqlite3-dev libssl-dev libxml2-dev libxslt-dev locales \
     lsb-release lvm2 moreutils net-tools open-iscsi openjdk-8-jdk openssl pciutils policykit-1 \
     python${PY_VERSION} python${PY_VERSION}-dev python${PY_VERSION}-distutils python${PY_VERSION}-venv \
     python3-pip python-openssl rsync rsyslog snapd scons sox ssh sudo time udev unzip ufw \
     uuid-runtime vim wget xz-utils zip zlib1g-dev zsh

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 1 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 1 && \
    update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /work
```

编译ACL需要gcc>=10，不然会报错：`fatal error: arm_sve.h: No such file or directory`

```bash
docker build . -f Dockerfile -t <image name>
# --name set container name
docker run --name pytorch --hostname pytorch -v <mount local dir>:/work -it <image name> zsh
docker start pytorch
docker exec -it pytorch zsh

export CMAKE_BUILD_PARALLEL_LEVEL=4 # set thread number to build pytorch
# https://github.com/pytorch/pytorch/issues/29327
export USE_QNNPACK=ON
export USE_PYTORCH_QNNPACK=ON
export USE_MPI=0

DESIRED_PYTHON="3.8" ./aarch64_ci_build.sh

cd /pytorch
cp /acl/build/libarm_compute.so       /pytorch/torch/lib
cp /acl/build/libarm_compute_graph.so /pytorch/torch/lib
cp /acl/build/libarm_compute_core.so  /pytorch/torch/lib
# wget http://mirror.archlinuxarm.org/aarch64/extra/ openblas-0.3.26-3-aarch64.pkg.tar.xz
tar openblas-0.3.26-3-aarch64.pkg.tar.xz
cp usr/lib/libopenblas.so.0 /pytorch/torch/lib
rm -rf usr
tar czf torch.tar.gz torch/lib/*.so* torch/include
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


<details>
<summary>Deprecated</summary>

pytorch 实在是太TM复杂了！！！而且还强烈依赖openblas库，对性能的影响非常敏感！！！

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
    "max-concurrent-downloads": 1,
    "registry-mirrors": [
        "https://xxx.mirror.aliyuncs.com"
    ]
}
```

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```


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

</details>
