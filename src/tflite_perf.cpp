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
#define USE_XNNPACK // always enable
#ifdef USE_XNNPACK
#include <tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h>
#endif

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

TfLiteDelegate* delegate;
void delete_delegate(char backend) {
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
#ifdef XNNPACK
    if (backend == 'x') {
        TfLiteXNNPackDelegateDelete(delegate);
    }
#endif
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

    static struct option long_options[] =
    {
        {"validation", no_argument, 0, 'v'},
        {"debug", no_argument, 0, 'g'},
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
        if (args.model.find("EMO") != std::string::npos) {
            //std::cout << "tflite didn't suppot EMO model!" << std::endl;
            continue;
        }
        if (only_test && args.model.find(only_test) == std::string::npos) {
            continue;
        }

        args.input_size = model.second;
        std::string model_file = ".tflite/" + args.model + ".tflite";
        // create a interpreter
        std::cout << "Creating tflite runtime interpreter: " << args.model << std::endl;
        std::unique_ptr<FlatBufferModel> tflite_model = FlatBufferModel::BuildFromFile(model_file.c_str());
        ops::builtin::BuiltinOpResolver resolver;
        // std::unique_ptr<Interpreter> interpreter;
        auto interpreter = std::make_unique<Interpreter>();
        InterpreterBuilder interpreter_builder(*tflite_model, resolver);
        interpreter_builder.SetNumThreads(num_threads);
        if (interpreter_builder(&interpreter) != kTfLiteOk) return EXIT_FAILURE;

#ifdef USE_GPU
        if (backend == 'g') {
            std::cout << "INFO: Using GPU backend" << std::endl;
            // https://www.tensorflow.org/lite/performance/gpu_advanced?hl=zh-cn
            // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/gpu/README.md
            // NEW: Prepare GPU delegate.
            TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
            // GPU 委托序列化
            /*options.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_SERIALIZATION;
            options.serialization_dir = kTmpDir;
            options.model_token = kModelToken;*/
            delegate = TfLiteGpuDelegateV2Create(&options);
            // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/gpu/cl/testing/delegate_testing.cc
            if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) {
                std::cout << "Failed to ModifyGraphWithDelegate to gpu!" << std::endl;
                return EXIT_FAILURE;
            }
            // interpreter_builder.AddDelegate(delegate);
            // if (interpreter_builder(&interpreter) != kTfLiteOk) return EXIT_FAILURE;
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
                return EXIT_FAILURE;
            }
        }
        else
#endif
#ifdef USE_ARMNN
        if (backend == 'a') {
            std::cout << "INFO: Using ARMNN backend" << std::endl;
            //https://review.mlplatform.org/plugins/gitiles/ml/armnn/+/3b38eedb3cc8f1c95a9ce62ddfbe926708666e72/delegate/BuildGuideNative.md#delegate-build-guide-introduction
            // Create the Arm NN Delegate
            //armnnDelegate::DelegateOptions options = armnnDelegate::TfLiteArmnnDelegateOptionsDefault();
            //std::unique_ptr<TfLiteDelegate, decltype(&armnnDelegate::TfLiteArmnnDelegateDelete)>
            //    theArmnnDelegate(armnnDelegate::TfLiteArmnnDelegateCreate(options), armnnDelegate::TfLiteArmnnDelegateDelete);
            //https://github.com/nxp-imx/armnn-imx/blob/lf-5.15.5_1.0.0/delegate/samples/armnn_delegate_example.cpp
            std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
            armnnDelegate::DelegateOptions delegateOptions(backends);
            std::unique_ptr<TfLiteDelegate, decltype(&armnnDelegate::TfLiteArmnnDelegateDelete)>
                        theArmnnDelegate(armnnDelegate::TfLiteArmnnDelegateCreate(delegateOptions),
                                         armnnDelegate::TfLiteArmnnDelegateDelete);

            // Instruct the Interpreter to use the armnnDelegate
            if (interpreter->ModifyGraphWithDelegate(theArmnnDelegate.get()) != kTfLiteOk) {
                std::cout << "Failed to ModifyGraphWithDelegate to Armnn!" << std::endl;
                return EXIT_FAILURE;
            }
        }
        else
#endif
#ifdef USE_XNNPACK
        if (backend == 'x') {
            std::cout << "INFO: Using XNNPACK backend" << std::endl;
            // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/xnnpack/README.md
            // IMPORTANT: initialize options with TfLiteXNNPackDelegateOptionsDefault() for
            // API-compatibility with future extensions of the TfLiteXNNPackDelegateOptions
            // structure.
            TfLiteXNNPackDelegateOptions options = TfLiteXNNPackDelegateOptionsDefault();
            options.num_threads = num_threads;
            delegate = TfLiteXNNPackDelegateCreate(&options);
            if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) {
                std::cout << "Failed to ModifyGraphWithDelegate to XNN!" << std::endl;
                return EXIT_FAILURE;
            }
        }
#endif

        if (interpreter->AllocateTensors() != kTfLiteOk) {
            std::cout << "Failed to allocate tensors!" << std::endl;
            return EXIT_FAILURE;
        }

        if (args.validation) {
            evaluate(interpreter);
        }
        else {
            benchmark(interpreter);
        }

        interpreter.reset();
        delete_delegate(backend);
    }
    return EXIT_SUCCESS;
}
