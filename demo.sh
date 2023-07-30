export OPENCV_LIB=$HOME/miniforge3/envs/py3.8/lib
export OPENCV_INC=$HOME/miniforge3/envs/py3.8/include/opencv4

export NCNN_INC=$PWD/libs/libncnn-07b840087b57a671a1b0f6f7814e05be8a8d71ef/include/ncnn
export TNN_INC=$PWD/libs/libTNN-0afdc3b3ad1f5b3bea205ed3426ed2235481a3a7/include
export MNN_INC=$PWD/libs/libMNN-d8266f9697650d4a90cccea337c2ae3ee070c373/include
export TF_INC=$PWD/libs/libtensorflow-bd472b86e70900037087958b08b7d1339de514f4/include
export ONNX_INC=$PWD/libs/libonnxruntime-9ba5cdbaa481550f86c53826c83bc094eafc2594/include
export PDL_INC=$PWD/libs/paddle_lite_lib.armlinux.armv8-c36a32e4721871499c558b30f3756fe2c954297a/cxx/include

export NCNN_LIB=$PWD/libs/libncnn-07b840087b57a671a1b0f6f7814e05be8a8d71ef/lib
export TNN_LIB=$PWD/libs/libTNN-0afdc3b3ad1f5b3bea205ed3426ed2235481a3a7/lib
export MNN_LIB=$PWD/libs/libMNN-d8266f9697650d4a90cccea337c2ae3ee070c373/lib
export TF_LIB=$PWD/libs/libtensorflow-bd472b86e70900037087958b08b7d1339de514f4/lib
export ONNX_LIB=$PWD/libs/libonnxruntime-9ba5cdbaa481550f86c53826c83bc094eafc2594/lib
export PDL_LIB=$PWD/libs/paddle_lite_lib.armlinux.armv8-c36a32e4721871499c558b30f3756fe2c954297a/cxx/lib

export LD_LIBRARY_PATH=$OPENCV_LIB:$NCNN_LIB:$TNN_LIB:$MNN_LIB:$TF_LIB:$ONNX_LIB:$PDL_LIB

./ncnn_perf        --only-test efficientformerv2_s1
./tnn_perf         --only-test efficientformerv2_s1
./mnn_perf         --only-test efficientformerv2_s1
./tflite_perf      --only-test efficientformerv2_s1
./onnxruntime_perf --only-test efficientformerv2_s1
./paddlelite_perf  --only-test efficientformerv2_s1
