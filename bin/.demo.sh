make run-ncnn-perf 2>/dev/null
make run-mnn-perf  2>/dev/null
make run-tnn-perf   2>/dev/null
make run-pdlite-perf 2>/dev/null
LD_PRELOAD=$PWD/.libs/tensorflow/lib/libtensorflowlite_flex.so make run-tflite-perf 2>/dev/null
make run-onnxruntime-perf 2>/dev/null