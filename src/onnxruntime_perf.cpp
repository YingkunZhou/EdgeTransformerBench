// https://leimao.github.io/blog/ONNX-Runtime-CPP-Inference/
// https://github.com/leimao/ONNX-Runtime-Inference/blob/main/src/inference.cpp
// https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/test/perftest/ort_test_session.cc

#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <iomanip>
#include <filesystem>
#include <getopt.h>
#include <algorithm>

#include <onnxruntime_cxx_api.h>

#if USE_DNNL
#include <dnnl_provider_options.h>
#endif

#ifdef USE_TENSORRT
#include <tensorrt_provider_options.h>
#endif

#ifdef USE_NNAPI
#include <nnapi_provider_factory.h>
#endif

#ifdef USE_ACL
#include <acl_provider_factory.h>
#endif

#ifdef USE_COREML
#include <coreml_provider_factory.h>
#endif

#include "utils.h"

#include <chrono>
using namespace std::chrono;

struct {
  std::string model;
  bool validation;
  int input_size;
  int batch_size;
  std::string data_path;

  std::vector<const char*> input_name;
  std::vector<const char*> output_name;
  std::vector<Ort::Value> input;
  std::vector<Ort::Value> output;
} args;

#define USE_ONNXRUNTIME
#include "evaluate.tcc"
#include "benchmark.tcc"

int main(int argc, char* argv[])
{
    args.data_path = "imagenet-div50";
    args.validation = false;
    args.batch_size = 1;
    char backend = 'a';
    bool use_gpu = false;
    bool use_int8 = false;
    char* arg_long = nullptr;
    char* only_test = nullptr;
    int num_threads = 1;

    static struct option long_options[] =
    {
        {"validation", no_argument, 0, 'v'},
        {"use-gpu", no_argument, 0, 'g'},
        {"use-int8", no_argument, 0, 'i'},
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
    while ((c = getopt_long(argc, argv, "vgiubdot", long_options, &option_index)) != -1)
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
                use_gpu = true;
                break;
            case 'i':
                use_int8 = true;
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

    std::string instanceName{"image-classification-inference"};
    //{ORT_LOGGING_LEVEL_ERROR, ORT_LOGGING_LEVEL_WARNING, ORT_LOGGING_LEVEL_VERBOSE}
    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                 instanceName.c_str());

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(num_threads);
    //https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/test/perftest/ort_test_session.cc#L676
    session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
    session_options.SetInterOpNumThreads(num_threads);
    // Sets graph optimization level
    // Available levels are
    // ORT_DISABLE_ALL -> To disable all optimizations
    // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node removals)
    // ORT_ENABLE_EXTENDED -> To enable extended optimizations
    // (Includes level 1 + more complex optimizations like node fusions)
    // ORT_ENABLE_ALL -> To Enable All possible optimizations
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef USE_NNAPI
    if (backend == 'n') {
        std::cout << "INFO: Using NNAPI backend" << std::endl;
        //https://onnxruntime.ai/docs/execution-providers/NNAPI-ExecutionProvider
        //https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/test/perftest/ort_test_session.cc#L449
        uint32_t nnapi_flags = 0;
        nnapi_flags |= NNAPI_FLAG_USE_FP16;
        // nnapi_flags |= NNAPI_FLAG_USE_NCHW;
        // nnapi_flags |= NNAPI_FLAG_CPU_DISABLED;
        // nnapi_flags |= NNAPI_FLAG_CPU_ONLY;
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nnapi(session_options, nnapi_flags));
    }
    else
#endif
#ifdef USE_DNNL
    if (backend == 'd') {
        std::cout << "INFO: Using ONEDNN backend" << std::endl;
        //https://onnxruntime.ai/docs/execution-providers/oneDNN-ExecutionProvider.html
        //https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/test/perftest/ort_test_session.cc#L50
        // Generate provider options
        OrtDnnlProviderOptions dnnl_options;
        dnnl_options.use_arena = 1;
        dnnl_options.threadpool_args = static_cast<void*>(&num_threads);;
        session_options.AppendExecutionProvider_Dnnl(dnnl_options);
    }
    else
#endif
#ifdef USE_TENSORRT
    if (backend == 't') {
        std::cout << "INFO: Using TENSORRT backend" << std::endl;
        // https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html
        const auto& api = Ort::GetApi();
        OrtTensorRTProviderOptionsV2* tensorrt_options;
        Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&tensorrt_options));
        std::vector<const char*> option_keys = {
            "device_id",
            "trt_max_workspace_size",
            "trt_dump_subgraphs",
            "trt_engine_cache_enable",
            "trt_timing_cache_enable",
            "trt_engine_cache_path",
            "trt_timing_cache_path",
        };
        std::vector<const char*> option_values = {
            "0",
            "2147483648",
            "1",
            "1", //"trt_engine_cache_enable",
            "1", //"trt_timing_cache_enable",
            ".trt-cache", //"trt_engine_cache_path",
            ".trt-cache", //"trt_timing_cache_path",
        };

        if (!use_gpu) {
            option_keys.push_back("trt_dla_enable");
            option_keys.push_back("trt_dla_core");
            option_values.push_back("1");
            option_values.push_back("0");
        }

        if (use_int8) {
            option_keys.push_back("trt_int8_enable");
            option_keys.push_back("trt_int8_use_native_calibration_table");
            option_keys.push_back("trt_int8_calibration_table_name");
            option_values.push_back("1");
            option_values.push_back("1");
            option_values.push_back(only_test);
        }
        else {
            option_keys.push_back("trt_fp16_enable");
            option_values.push_back("1");
        }

        Ort::ThrowOnError(api.UpdateTensorRTProviderOptions(
            tensorrt_options,
            option_keys.data(), option_values.data(), option_keys.size()));

        session_options.AppendExecutionProvider_TensorRT_V2(*tensorrt_options);
        // api.ReleaseTensorRTProviderOptions(tensorrt_options);

        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = tensorrt_options->device_id;
        // 0:OrtCudnnConvAlgoSearchExhaustive,  // expensive exhaustive benchmarking using cudnnFindConvolutionForwardAlgorithmEx
        // 1:OrtCudnnConvAlgoSearchHeuristic,   // lightweight heuristic based search using cudnnGetConvolutionForwardAlgorithm_v7
        // 2:OrtCudnnConvAlgoSearchDefault,     // default algorithm using CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
        cuda_options.cudnn_conv_algo_search = static_cast<OrtCudnnConvAlgoSearch>(0);
        cuda_options.do_copy_in_default_stream = true; // TODO
        // TODO: Support arena configuration for users of perf test
        // cuda_options.arena_extend_strategy = 0; // 0: kNextPowerOfTwo, 1: kSameAsRequested
        session_options.AppendExecutionProvider_CUDA(cuda_options);
    }
    else
#endif
#ifdef USE_ACL
    if (backend == 'a') {
        std::cout << "INFO: Using ACL backend" << std::endl;
        bool enable_cpu_mem_arena = true;
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_ACL(session_options, enable_cpu_mem_arena));
    }
    else
#endif
#ifdef USE_COREML
    if (backend == 'c') {
        std::cout << "INFO: Using COREML backend" << std::endl;
        // https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html
        uint32_t coreml_flags = 0;
        // coreml_flags |= COREML_FLAG_CREATE_MLPROGRAM;
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(session_options, coreml_flags));
    }
    else
#endif
    if (backend == 'q') {
        std::cout << "INFO: Using QNN backend" << std::endl;
        //https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html
        //https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/test/perftest/ort_test_session.cc#L323
        std::unordered_map<std::string, std::string> qnn_options;
        qnn_options["backend_path"] = ".libs/onnxruntime/lib/libQnnCpu.so";
        // qnn_options["backend_path"] = ".libs/onnxruntime/lib/libQnnHtp.so";
        // qnn_options["profiling_level"] = "off"; //{"off", "basic", "detailed"};
        //"burst", "balanced", "default", "high_performance",
        //"high_power_saver", "low_balanced", "extreme_power_saver",
        //"low_power_saver", "power_saver", "sustained_high_performance"
        // qnn_options["htp_performance_mode"] = "high_performance";
        // qnn_options["htp_graph_finalization_optimization_mode"] = "3"; //{"0", "1", "2", "3"};
        // qnn_options["qnn_context_priority"] = "high"; //{"low", "normal", "normal_high", "high"};
        // qnn_options["htp_arch"] = "0"; //{"0", "68", "69", "73", "75"};
        //key == "rpc_control_latency" || key == "vtcm_mb" || key == "soc_model" || key == "device_id"
        session_options.AppendExecutionProvider("QNN", qnn_options);
    }
    else if (backend == 'x') {
        std::cout << "INFO: Using XNNPACK backend" << std::endl;
        //https://onnxruntime.ai/docs/execution-providers/Xnnpack-ExecutionProvider.html
        //https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/test/perftest/ort_test_session.cc#L586
        session_options.AddConfigEntry("session.intra_op.allow_spinning", "0");
        session_options.AppendExecutionProvider(
        "XNNPACK", {{"intra_op_num_threads", std::to_string(num_threads)}});
    }
    else {
        std::cout << "INFO: Using CPU backend" << std::endl;
    }

    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    for (const auto & model: test_models) {
        args.model = model.first;
        if (only_test && strcmp(only_test, "ALL") && args.model.find(only_test) == std::string::npos) {
            continue;
        }

        args.input_size = model.second;
        std::string model_file = ".onnx/" + args.model + ".onnx";
        if (model_exists(model_file) == 0) {
            std::cerr << args.model << " model doesn't exist!!!" << std::endl;
            continue;
        }
        // create a session
        std::cout << "Creating onnx runtime session: " << args.model << std::endl;
        Ort::Session session(env, model_file.c_str(), session_options);
        //// input
        Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
        auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
        if (inputDims.at(0) == -1)
        {
            std::cout << "Got dynamic batch size. Setting input batch size to "
                    << args.batch_size << "." << std::endl;
            inputDims.at(0) = args.batch_size;
        }

        size_t inputTensorSize  = vectorProduct(inputDims);
        std::vector<float> input_tensor(inputTensorSize);
        std::vector<Ort::Value> inputTensors;
        args.input.clear();
        args.input.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo,
            input_tensor.data(), inputTensorSize,
            inputDims.data(), inputDims.size()));

        const auto inputName = session.GetInputNameAllocated(0, allocator);
        args.input_name.clear();
        args.input_name.push_back(&*inputName);

        //// output
        Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
        auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> outputDims = outputTensorInfo.GetShape();
        if (outputDims.at(0) == -1)
        {
            std::cout << "Got dynamic batch size. Setting output batch size to "
                    << args.batch_size << "." << std::endl;
            outputDims.at(0) = args.batch_size;
        }

        size_t outputTensorSize = vectorProduct(outputDims);
        std::vector<float> output_tensor(outputTensorSize);
        std::vector<Ort::Value> outputTensors;
        args.output.clear();
        args.output.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo,
            output_tensor.data(), outputTensorSize,
            outputDims.data(), outputDims.size()));

        const auto outputName = session.GetOutputNameAllocated(0, allocator);
        args.output_name.clear();
        args.output_name.push_back(&*outputName);

#if 0
        //print basic model I/O info
        std::cout << "Number of Input Nodes: " << session.GetInputCount() << std::endl;
        std::cout << "Number of Output Nodes: " << session.GetOutputCount() << std::endl;

        std::cout << "Input Name: " << args.input_name << std::endl;
        std::cout << "Input Type: " << inputTensorInfo.GetElementType() << std::endl;
        std::cout << "Input Dimensions: " << inputDims << std::endl;

        std::cout << "Output Name: " << args.output_name << std::endl;
        std::cout << "Output Type: " << outputTensorInfo.GetElementType() << std::endl;
        std::cout << "Output Dimensions: " << outputDims << std::endl;
#endif

        if (args.validation) {
            evaluate(session, input_tensor, output_tensor);
        }
        else {
            benchmark(session, input_tensor, output_tensor);
        }
    }
}
