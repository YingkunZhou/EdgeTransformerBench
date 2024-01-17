/* Reference code:
   https://github.com/alibaba/MNN/blob/master/benchmark/benchmark.cpp
   https://github.com/alibaba/MNN/blob/master/demo/exec/pictureRecognition.cpp
   https://www.yuque.com/mnn/en/create_session
*/

#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <iomanip>
#include <filesystem>
#include <getopt.h>
#include <cstring>

#include <MNN/Interpreter.hpp>
#include "utils.h"
#include <chrono>
using namespace std::chrono;

struct {
  std::string model;
  bool validation;
  int input_size;
  int batch_size;
  bool debug;
  std::string data_path;
  MNN::Tensor* input;
  MNN::Tensor* output;
} args;

#define USE_MNN
#include "evaluate.tcc"
#include "benchmark.tcc"

int main(int argc, char* argv[])
{
    args.data_path = "imagenet-div50";
    args.validation = false;
    args.batch_size = 1;
    args.debug = false;
    int forward = MNN_FORWARD_CPU;
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
    while ((c = getopt_long(argc, argv, "vgubdot", // TODO
            long_options, &option_index)) != -1)
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
                if (optarg[0] == 'v') {
                    forward = MNN_FORWARD_VULKAN;
                    std::cout << "INFO: Using Vulkan backend" << std::endl;
                }
                else if (optarg[0] == 'o') {
                    forward = MNN_FORWARD_OPENCL;
                    std::cout << "INFO: Using OpenCL backend" << std::endl;
                }
                else if (optarg[0] == 'g') {
                    forward = MNN_FORWARD_OPENGL;
                    std::cout << "INFO: Using OpenGL backend" << std::endl;
                }
                else if (optarg[0] == 'c') {
                    forward = MNN_FORWARD_CUDA;
                    std::cout << "INFO: Using CUDA backend" << std::endl;
                }
                else {
                    std::cout << "INFO: Using CPU backend" << std::endl;
                }
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

    // https://www.yuque.com/mnn/cn/create_session#Wi4on
    // TODO: for better performance (lower latency)
    int precision = MNN::BackendConfig::Precision_High;
    MNN::ScheduleConfig config;
    config.type = static_cast<MNNForwardType>(forward);
    config.numThread = num_threads;
    MNN::BackendConfig backendConfig;
    backendConfig.precision = (MNN::BackendConfig::PrecisionMode) precision;
    backendConfig.power = MNN::BackendConfig::Power_High;
    config.backendConfig = &backendConfig;
    // https://www.yuque.com/mnn/cn/create_session#xtQLb
    auto runtimeInfo = MNN::Interpreter::createRuntime({config});

    for (const auto & model: test_models) {
        args.model = model.first;
        if (only_test && strcmp(only_test, "ALL") && args.model.find(only_test) == std::string::npos) {
            continue;
        }

        args.input_size = model.second;

        std::cout << "Creating MNN Interpreter: " << args.model << std::endl;
        std::string model_file = ".mnn/" + args.model + ".mnn";
        if (model_exists(model_file) == 0) {
            std::cerr << args.model << " model doesn't exist!!!" << std::endl;
            continue;
        }
        std::shared_ptr<MNN::Interpreter> net(MNN::Interpreter::createFromFile(model_file.c_str()), MNN::Interpreter::destroy);
#if 0
        // https://www.yuque.com/mnn/cn/create_session#KtfMk
        //net->setCacheFile(".cachefile");
        net->setSessionMode(MNN::Interpreter::Session_Backend_Auto);
        net->setSessionHint(MNN::Interpreter::MAX_TUNING_NUMBER, 10);
#else
        // net->setSessionMode(MNN::Interpreter::Session_Debug);
        net->setSessionMode(MNN::Interpreter::Session_Release);
#endif

        auto session = net->createSession(config, runtimeInfo);
        if (args.debug) {
            float memoryUsage = 0.0f;
            float flops = 0.0f;
            int backendType[2]; // TODO: 2?
            net->getSessionInfo(session, MNN::Interpreter::MEMORY, &memoryUsage);
            net->getSessionInfo(session, MNN::Interpreter::FLOPS, &flops);
            net->getSessionInfo(session, MNN::Interpreter::BACKENDS, backendType);
            MNN_PRINT("Session Info: memory use %f MB, flops is %f M, backendType is %d, batch size = %d\n", memoryUsage, flops, backendType[0], args.batch_size);
        }

        args.input = net->getSessionInput(session, NULL);
        args.output = net->getSessionOutput(session, NULL);
        //auto shape = input->shape();
        //shape[0] = args.batch_size; //e.g. Set Batch Size
        //std::vector<int> shape{1, 3, 224, 224}; //or
        //net->resizeTensor(input, shape);
        //net->resizeSession(session);
        net->releaseModel(); //TODO: ?

        auto input_tensor = new MNN::Tensor(args.input, args.input->getDimensionType());
        if (args.validation) {
            evaluate(net, session, input_tensor);
        }
        else {
            benchmark(net, session, input_tensor);
        }
    }
}
