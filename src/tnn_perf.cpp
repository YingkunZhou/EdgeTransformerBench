/* Reference code:
   https://github.com/Tencent/TNN/blob/master/doc/cn/user/api.md
   https://github.com/Tencent/TNN/blob/master/examples/base/tnn_sdk_sample.cc
   https://github.com/Tencent/TNN/blob/master/test/test.cc
*/

#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <iomanip>
#include <filesystem>
#include <getopt.h>
#include <fstream>

#include <tnn/core/common.h>
#include <tnn/core/instance.h>
#include <tnn/core/macro.h>
#include <tnn/core/tnn.h>
#include "utils.h"

#include <chrono>
using namespace std::chrono;

// Helper functions
std::string fdLoadFile(std::string path) {
    std::ifstream file(path);
    if (file.is_open()) {
        file.seekg(0, file.end);
        int size      = file.tellg();
        char* content = new char[size];
        file.seekg(0, file.beg);
        file.read(content, size);
        std::string fileContent;
        fileContent.assign(content, size);
        delete[] content;
        file.close();
        return fileContent;
    }

    return "";
}

struct {
  std::string model;
  bool validation;
  int input_size;
  int batch_size;
  bool debug;
  std::string data_path;
  std::vector<int> input_dims;
  std::shared_ptr<tnn::Mat> output_tensor;
  tnn::Status status;
} args;

#define USE_TNN
#include "evaluate.tcc"
#include "benchmark.tcc"

int main(int argc, char* argv[])
{
    args.data_path = "imagenet-div50";
    args.validation = false;
    args.batch_size = 1;
    args.debug = false;
    tnn::DeviceType backend = tnn::DEVICE_ARM;
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
                if (optarg[0] == 'o') {
                    backend = tnn::DEVICE_OPENCL;
                    std::cout << "INFO: Using OpenCL backend" << std::endl;
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

    // TODO: Set the cpu affinity.
    // usually, -dl 0-3 for little core, -dl 4-7 for big core
    // only works when -dl flags were set. benchmark script not set -dl flags
    // SetCpuAffinity();

    for (const auto & model: test_models) {
        args.model = model.first;
        if (only_test && args.model.find(only_test) == std::string::npos) {
            continue;
        }

        args.input_size = model.second;

        std::cout << "Creating TNN net: " << args.model << std::endl;
        tnn::ModelConfig model_config;
        // model_config.model_type = tnn::MODEL_TYPE_NCNN;
        model_config.model_type = tnn::MODEL_TYPE_TNN;
        model_config.params.clear();
        // TODO: has opt suffix?
        std::string tnnproto = ".tnn/" + args.model + ".opt.tnnproto";
        std::string tnnmodel = ".tnn/" + args.model + ".opt.tnnmodel";

        model_config.params.push_back(fdLoadFile(tnnproto.c_str()));
        model_config.params.push_back(fdLoadFile(tnnmodel.c_str()));
        // model_config.params.push_back(model_path_str_) ??
        tnn::TNN net;
        auto status = net.Init(model_config);

        tnn::NetworkConfig network_config;
        network_config.device_type = backend;

        // TODO: network_config.{library_path, precision, cache_path, network_type}
        args.input_dims = {1, 3/*image_channel*/, args.input_size, args.input_size};
        tnn::InputShapesMap input_shapes = {{"input", args.input_dims}};
        std::shared_ptr<tnn::Instance> instance = net.CreateInst(network_config, status, input_shapes);
        // TODO: num_threads <= 4 in orin doesn't work?!
        instance->SetCpuNumThreads(num_threads);

        size_t inputTensorSize  = vectorProduct(args.input_dims);
        std::vector<float> input_tensor(inputTensorSize);
        if (args.validation) {
            evaluate(net, instance, input_tensor);
        }
        else {
            benchmark(net, instance, input_tensor);
        }
    }
}
