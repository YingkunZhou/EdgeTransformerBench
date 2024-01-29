//https://www.tensorflow.org/lite/guide/inference

#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <iomanip>
#include <filesystem>
#include <getopt.h>

#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#ifdef USE_GPU
#include <tensorflow/lite/delegates/gpu/delegate.h>
#endif
#ifdef USE_NNAPI
#include <tensorflow/lite/delegates/nnapi/nnapi_delegate_c_api.h>
#endif
#ifdef USE_ARMNN
#include <armnn_delegate.hpp>
#endif
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/xnnpack/README.md
// XNNPACK engine used by TensorFlow Lite interpreter uses a single thread for inference by default.
// Enable XNNPACK via low-level delegate API (not recommended)
#include <tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h>

#include "utils.h"
using namespace tflite;
#include <chrono>
using namespace std::chrono;

struct {
  std::string model;
  bool validation;
  int input_size;
  int batch_size;
  bool debug;
  std::string data_path;
} args;

#define USE_TFLITE
#include "evaluate.tcc"
#include "benchmark.tcc"

void delete_delegate(char backend, TfLiteDelegate* delegate) {
#ifdef GPU
    if (backend == 'g') {
        TfLiteGpuDelegateV2Delete(delegate);
    }
    else
#endif
#ifdef NNAPI
    if (backend == 'n') {
        TfLiteNnapiDelegateDelete(delegate);
    }
    else
#endif
    if (backend == 'x') {
        TfLiteXNNPackDelegateDelete(delegate);
    }
}

int main(int argc, char* argv[])
{
    args.data_path = "imagenet-div50";
    args.validation = false;
    args.batch_size = 1;
    args.debug = false;
    char backend = 'c';
    char* arg_long = nullptr;
    char* only_test = nullptr;
    int num_threads = 1;
    int fpbits = 32;

    static struct option long_options[] =
    {
        {"validation", no_argument, 0, 'v'},
        {"debug", no_argument, 0, 'g'},
        {"fp", required_argument, 0, 'f'},
        {"backend",  required_argument, 0, 'u'},
        {"batch-size", required_argument, 0, 'b'},
        {"data-path",  required_argument, 0, 'd'},
        {"only-test",  required_argument, 0, 'o'},
        {"threads",  required_argument, 0, 't'},
        {"append",  required_argument, 0, 0},
        {0, 0, 0, 0}
    };
    int option_index;
    int c;
    while ((c = getopt_long(argc, argv, "vgubdot", long_options, &option_index)) != -1)
    {
        switch (c)
        {
            case 0:
            {
                std::cout << "Got long option " << long_options[option_index].name << "." << std::endl;
                arg_long = optarg;
                if (arg_long)
                {
                    std::cout << arg_long << std::endl;
                }
                break;
            }
            case 'v':
                args.validation = true;
                break;
            case 'b':
                args.batch_size = atoi(optarg);
                break;
            case 'd':
                args.data_path = optarg;
                break;
            case 'o':
                only_test = optarg;
                break;
            case 'g':
                args.debug = true;
                break;
            case 'u':
                backend = optarg[0];
                break;
            case 't':
                num_threads = atoi(optarg);
                break;
            case 'f':
                fpbits = atoi(optarg);
                break;
            case '?':
                std::cout << "Got unknown option." << std::endl;
                break;
            default:
                std::cout << "Got unknown parse returns: " << c << std::endl;
        }
    }
    std::cout << "INFO: Using num_threads == " << num_threads << std::endl;

    for (const auto & model: test_models) {
        args.model = model.first;
        if (only_test && strcmp(only_test, "ALL") && args.model.find(only_test) == std::string::npos) {
            continue;
        }

        args.input_size = model.second;
        std::string model_file = ".tflite/" + args.model + ".tflite";
        if (model_exists(model_file) == 0) {
            std::cerr << args.model << " model doesn't exist!!!" << std::endl;
            continue;
        }
        // create a interpreter
        std::cout << "Creating tflite runtime interpreter: " << args.model << std::endl;
        std::unique_ptr<FlatBufferModel> tflite_model = FlatBufferModel::BuildFromFile(model_file.c_str());
        ops::builtin::BuiltinOpResolver resolver;
        // std::unique_ptr<Interpreter> interpreter;
        auto interpreter = std::make_unique<Interpreter>();
        InterpreterBuilder interpreter_builder(*tflite_model, resolver);
        interpreter_builder.SetNumThreads(num_threads);
        if (interpreter_builder(&interpreter) != kTfLiteOk) return EXIT_FAILURE;
        TfLiteDelegate* delegate;

#ifdef USE_GPU
        if (backend == 'g') {
            std::cout << "INFO: Using GPU backend" << std::endl;
            // https://www.tensorflow.org/lite/performance/gpu_advanced?hl=zh-cn
            // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/gpu/README.md
            // NEW: Prepare GPU delegate.
            // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/gpu/delegate_options.cc
            TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
            if (fpbits == 16) options.is_precision_loss_allowed = 1; // GPU performs FP16 calculation internally
            /* GPU 委托序列化
            options.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_SERIALIZATION;
            options.serialization_dir = kTmpDir;
            options.model_token = kModelToken;
            */
            delegate = TfLiteGpuDelegateV2Create(&options);
            // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/gpu/cl/testing/delegate_testing.cc
            if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) {
                std::cout << "Failed to ModifyGraphWithDelegate to gpu!" << std::endl;
                continue;
                return EXIT_FAILURE;
            }
        }
        else
#endif
#ifdef USE_NNAPI
        if (backend == 'n') {
            std::cout << "INFO: Using NNAPI backend" << std::endl;
            //TODO: https://discuss.tensorflow.org/t/neural-network-fallback-to-cpu-using-nnapi-on-android/7703
            // https://community.nxp.com/t5/i-MX-Processors/how-to-know-the-imx8m-plus-NPU-acceleration-is-enable-already/m-p/1305328
            // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/evaluation/utils.cc#L106C42-L106C42
            // https://zenn.dev/iwatake2222/scraps/05e60dd7338294
            // StatefulNnApiDelegate::Options options;
            // options.execution_preference = tflite::StatefulNnApiDelegate::Options::kSustainedSpeed;
            // options.disallow_nnapi_cpu = true;
            // options.allow_fp16 = true;
            // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/nnapi/nnapi_delegate_c_api.cc
            TfLiteNnapiDelegateOptions options = TfLiteNnapiDelegateOptionsDefault();
            delegate = TfLiteNnapiDelegateCreate(&options);
            if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) {
                std::cout << "Failed to ModifyGraphWithDelegate to nnapi!" << std::endl;
                continue;
                return EXIT_FAILURE;
            }
        }
        else
#endif
#ifdef USE_ARMNN
        if (backend == 'a' || backend == 'm') { // a: cpu; m: mali gpu
            std::cout << "INFO: Using ARMNN backend" << std::endl;
            //https://review.mlplatform.org/plugins/gitiles/ml/armnn/+/3b38eedb3cc8f1c95a9ce62ddfbe926708666e72/delegate/BuildGuideNative.md#delegate-build-guide-introduction
            //https://github.com/search?q=repo%3AARM-software%2Farmnn+TfLiteArmnnDelegateDelete&type=code
            //https://github.com/nxp-imx/armnn-imx/blob/lf-5.15.5_1.0.0/delegate/samples/armnn_delegate_example.cpp
            //https://github.com/ARM-software/armnn/blob/branches/armnn_23_11/samples/ObjectDetection/Readme.md
            // Create the Arm NN Delegate
            armnn::OptimizerOptionsOpaque optimizerOptions;
            std::vector<armnn::BackendId> backends; // armnn::Compute::CpuRef
            if (backend == 'a') {
                    backends = {armnn::Compute::CpuAcc};
            }
            else {
                std::vector<armnn::BackendId> backends =
                    backends = {armnn::Compute::GpuAcc, armnn::Compute::CpuAcc};
            }
            // the leagal name to pass thread number parameter
            unsigned int numberOfThreads = num_threads;

            /* enable fast math optimization */
            if (backend == 'm') {
                armnn::BackendOptions modelOptionGpu("GpuAcc", {{"FastMathEnabled", true}});
                optimizerOptions.AddModelOption(modelOptionGpu);
            }

            armnn::BackendOptions modelOptionCpu("CpuAcc",
                                        {
                                            { "FastMathEnabled", true },
                                            { "NumberOfThreads", numberOfThreads }
                                        });
            optimizerOptions.AddModelOption(modelOptionCpu);
            /* enable reduce float32 to float16 optimization */
            // https://community.arm.com/arm-community-blogs/b/ai-and-ml-blog/posts/making-the-most-of-arm-nn-for-gpu-inference
            if (fpbits == 16) optimizerOptions.SetReduceFp32ToFp16(true);
            armnnDelegate::DelegateOptions delegateOptions(backends, optimizerOptions);
            /* create delegate object */
            std::unique_ptr<TfLiteDelegate, decltype(&armnnDelegate::TfLiteArmnnDelegateDelete)>
                theArmnnDelegate(armnnDelegate::TfLiteArmnnDelegateCreate(delegateOptions),
                                 armnnDelegate::TfLiteArmnnDelegateDelete);

            // Instruct the Interpreter to use the armnnDelegate
            if (interpreter->ModifyGraphWithDelegate(std::move(theArmnnDelegate)) != kTfLiteOk) {
                std::cout << "Failed to ModifyGraphWithDelegate to Armnn!" << std::endl;
                continue;
                return EXIT_FAILURE;
            }
        }
        else
#endif
        if (backend == 'x') {
            std::cout << "INFO: Using XNNPACK backend" << std::endl;
            // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/xnnpack/README.md
            // IMPORTANT: initialize options with TfLiteXNNPackDelegateOptionsDefault() for
            // API-compatibility with future extensions of the TfLiteXNNPackDelegateOptions
            // structure.
            TfLiteXNNPackDelegateOptions options = TfLiteXNNPackDelegateOptionsDefault();
            options.num_threads = num_threads;
            if (fpbits == 16) options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16;
            delegate = TfLiteXNNPackDelegateCreate(&options);
            if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) {
                std::cout << "Failed to ModifyGraphWithDelegate to XNN!" << std::endl;
                continue;
                return EXIT_FAILURE;
            }
        }
        else if (interpreter->AllocateTensors() != kTfLiteOk) {
            std::cout << "Failed to allocate tensors!" << std::endl;
            return EXIT_FAILURE;
        }

        if (args.validation) {
            evaluate(interpreter);
        }
        else {
            benchmark(interpreter);
        }

        // IMPORTANT: release the interpreter before destroying the delegate
        interpreter.reset();
        delete_delegate(backend, delegate);
    }
    return EXIT_SUCCESS;
}
