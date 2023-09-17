// https://github.com/microsoft/onnxruntime/blob/v1.8.2/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/CXX_Api_Sample.cpp
// https://github.com/microsoft/onnxruntime/blob/v1.8.2/include/onnxruntime/core/session/onnxruntime_cxx_api.h

#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <iomanip>
#include <filesystem>
#include <getopt.h>
#include <algorithm>

#include <onnxruntime_cxx_api.h>
#ifdef NNAPI
#include <nnapi_provider_factory.h>
#endif
#include "utils.h"

#include <chrono>
using namespace std::chrono;

/**
 * @brief Operator overloading for printing vectors
 * @tparam T
 * @param os
 * @param v
 * @return std::ostream&
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i)
    {
        os << v[i];
        if (i != v.size() - 1)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

/**
 * @brief Print ONNX tensor data type
 * https://github.com/microsoft/onnxruntime/blob/rel-1.6.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L93
 * @param os
 * @param type
 * @return std::ostream&
 */
std::ostream& operator<<(std::ostream& os,
                         const ONNXTensorElementDataType& type)
{
    switch (type)
    {
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
            os << "undefined";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            os << "float";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            os << "uint8_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            os << "int8_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            os << "uint16_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            os << "int16_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            os << "int32_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            os << "int64_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
            os << "std::string";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            os << "bool";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            os << "float16";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            os << "double";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            os << "uint32_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            os << "uint64_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
            os << "float real + float imaginary";
            break;
        case ONNXTensorElementDataType::
            ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
            os << "double real + float imaginary";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
            os << "bfloat16";
            break;
        default:
            break;
    }

    return os;
}

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
    bool debug = false;
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
                debug = true;
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
    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                 instanceName.c_str());
    Ort::SessionOptions sessionOptions;
#ifdef USE_NNAPI
    if (backend == 'n') {
        std::cout << "INFO: Using NNAPI backend" << std::endl;
        // https://onnxruntime.ai/docs/execution-providers/NNAPI-ExecutionProvider
        uint32_t nnapi_flags = 0;
        //nnapi_flags |= NNAPI_FLAG_USE_FP16;
        //nnapi_flags |= NNAPI_FLAG_USE_NCHW;
        //nnapi_flags |= NNAPI_FLAG_CPU_DISABLED;
        //nnapi_flags |= NNAPI_FLAG_CPU_ONLY;
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nnapi(sessionOptions, nnapi_flags));
    }
    else
#endif
    if (backend == 'q') {
        std::cout << "INFO: Using QNN backend" << std::endl;
        // https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html
        std::unordered_map<std::string, std::string> qnn_options;
        // qnn_options["backend_path"] = "libQnnCpu.so";
        qnn_options["backend_path"] = "libQnnHtp.so";
        Ort::SessionOptions session_options;
        session_options.AppendExecutionProvider("QNN", qnn_options);
    }
    else
    {
        std::cout << "INFO: Using CPU backend" << std::endl;
        sessionOptions.SetIntraOpNumThreads(num_threads);
        // Sets graph optimization level
        // Available levels are
        // ORT_DISABLE_ALL -> To disable all optimizations
        // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node
        // removals) ORT_ENABLE_EXTENDED -> To enable extended optimizations
        // (Includes level 1 + more complex optimizations like node fusions)
        // ORT_ENABLE_ALL -> To Enable All possible optimizations
        sessionOptions.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);

        sessionOptions.SetExecutionMode(
            ExecutionMode::ORT_PARALLEL);

        int inter_threads = num_threads; // TODO
        sessionOptions.SetInterOpNumThreads(inter_threads);
    }

    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu( //TODO: cpu?
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    for (const auto & model: test_models) {
        args.model = model.first;
        if (only_test && args.model.find(only_test) == std::string::npos) {
            continue;
        }

        args.input_size = model.second;
        std::string model_file = ".onnx/" + args.model + ".onnx";
        // create a session
        std::cout << "Creating onnx runtime session: " << args.model << std::endl;
        Ort::Session session(env, model_file.c_str(), sessionOptions);
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
        args.input.push_back(Ort::Value::CreateTensor<float>(memoryInfo,
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
        args.output.push_back(Ort::Value::CreateTensor<float>(memoryInfo,
            output_tensor.data(), outputTensorSize,
            outputDims.data(), outputDims.size()));

        const auto outputName = session.GetOutputNameAllocated(0, allocator);
        args.output_name.clear();
        args.output_name.push_back(&*outputName);

        if (debug) {
            //print basic model I/O info
            std::cout << "Number of Input Nodes: " << session.GetInputCount() << std::endl;
            std::cout << "Number of Output Nodes: " << session.GetOutputCount() << std::endl;

            std::cout << "Input Name: " << args.input_name << std::endl;
            std::cout << "Input Type: " << inputTensorInfo.GetElementType() << std::endl;
            std::cout << "Input Dimensions: " << inputDims << std::endl;

            std::cout << "Output Name: " << args.output_name << std::endl;
            std::cout << "Output Type: " << outputTensorInfo.GetElementType() << std::endl;
            std::cout << "Output Dimensions: " << outputDims << std::endl;
        }

        if (args.validation) {
            evaluate(session, input_tensor, output_tensor);
        }
        else {
            benchmark(session, input_tensor, output_tensor);
        }
    }
}
