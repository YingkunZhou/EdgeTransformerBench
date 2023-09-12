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
#include "onnxruntime_perf.h"
#include "utils.h"

const int WARMUP_SEC = 5;
const int TEST_SEC = 20;

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

void evaluate(
    Ort::Session &session,
    std::vector<float> &input_tensor,
    std::vector<float> &output_tensor)
{
    int class_index = 0;
    int num_predict = 0;
    int num_acc1 = 0;
    int num_acc5 = 0;
    std::cout << std::fixed << std::setprecision(4);

    int scale = 1;
    int offset = 0;
    if (args.data_path.find("20") != std::string::npos) {
        scale = 20;
    }
    else if (args.data_path.find("50") != std::string::npos) {
        scale = 50;
        offset = 15;
    }

    std::vector<std::filesystem::path> classes = traverse_class(args.data_path);
    for (const std::string& class_path : classes) {
        for (const auto & image: std::filesystem::directory_iterator(class_path)) {
            load_image(image.path(), input_tensor.data(), args.model, args.input_size, args.batch_size);
            session.Run(Ort::RunOptions{nullptr},
                        args.input_name.data(), args.input.data() , 1 /*Number of inputs*/,
                        args.output_name.data(),args.output.data(), 1 /*Number of outputs*/
                        );
            num_predict++;
            bool acc1 = false;
            num_acc5 += acck(output_tensor.data(), 5, class_index*scale+offset, acc1);
            num_acc1 += acc1;
        }
        class_index++;
        std::cout << "Done [" << class_index << "/" << classes.size() << "]";
        std::cout << "\tacc1: " << num_acc1*1.f/num_predict;
        std::cout << "\tacc5: " << num_acc5*1.f/num_predict << std::endl;
    }
}

void benchmark(
    Ort::Session &session,
    std::vector<float> &input_tensor,
    std::vector<float> &output_tensor)
{
    // Measure latency
    load_image("daisy.jpg", input_tensor.data(), args.model, args.input_size, args.batch_size);
    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &end);
    clock_gettime(CLOCK_REALTIME, &start);

    while (end.tv_sec - start.tv_sec < WARMUP_SEC) {
        session.Run(Ort::RunOptions{nullptr},
                    args.input_name.data(), args.input.data() , 1 /*Number of inputs*/,
                    args.output_name.data(),args.output.data(), 1 /*Number of outputs*/
                    );
        clock_gettime(CLOCK_REALTIME, &end);
    }

    print_topk(output_tensor.data(), 3);

    std::vector<double> time_list = {};
    double time_tot = 0;
    while (time_tot < TEST_SEC) {
        clock_gettime(CLOCK_REALTIME, &start);
        session.Run(Ort::RunOptions{nullptr},
                    args.input_name.data(), args.input.data() , 1 /*Number of inputs*/,
                    args.output_name.data(),args.output.data(), 1 /*Number of outputs*/
                    );
        clock_gettime(CLOCK_REALTIME, &end);
        long long seconds = end.tv_sec - start.tv_sec;
        long long nanoseconds = end.tv_nsec - start.tv_nsec;
        double elapse = seconds + nanoseconds * 1e-9;
        time_list.push_back(elapse);
        time_tot += elapse;
    }

    double time_max = *std::max_element(time_list.begin(), time_list.end()) * 1000;
    double time_min = *std::min_element(time_list.begin(), time_list.end()) * 1000;
    double time_mean = time_tot * 1000 / time_list.size();
    std::sort(time_list.begin(), time_list.end());
    double time_median = time_list[time_list.size() / 2] * 1000;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "[" << time_list.size() << " iters]";
    std::cout << " min ="   << std::setw(7) << time_min  << "ms";
    std::cout << " max ="   << std::setw(7) << time_max  << "ms";
    std::cout << " median ="<< std::setw(7) << time_median<<"ms";
    std::cout << " mean ="  << std::setw(7) << time_mean << "ms" << std::endl;
}

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

    std::string instanceName{"image-classification-inference"};
    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                 instanceName.c_str());
    Ort::SessionOptions sessionOptions;
#ifdef NNAPI
    if (backend == 'n') {
        // https://onnxruntime.ai/docs/execution-providers/NNAPI-ExecutionProvider
        uint32_t nnapi_flags = 0;
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nnapi(sessionOptions, nnapi_flags));
    }
    else
#endif
#ifdef QNN
    if (backend == 'q') {
        // https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html
        std::unordered_map<std::string, std::string> qnn_options;
        // qnn_options["backend_path"] = "libQnnCpu.so";
        qnn_options["backend_path"] = "libQnnHtp.so";
        Ort::SessionOptions session_options;
        session_options.AppendExecutionProvider("QNN", qnn_options);
    }
    else
#endif
    {
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
        std::string model_file = "onnx/" + args.model + ".onnx";
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
