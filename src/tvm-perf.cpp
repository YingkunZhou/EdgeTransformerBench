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

#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
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
} args;

#define USE_TVM
#include "evaluate.tcc"
#include "benchmark.tcc"

int main(int argc, char* argv[])
{
    args.data_path = "imagenet-div50";
    args.validation = false;
    args.batch_size = 1;
    args.debug = false;
    char* arg_long = nullptr;
    char* only_test = nullptr;
    char* extern_model = nullptr;
    int num_threads = 1;
    int fpbits = 32;
    DLDevice dev{kDLCPU, 0};

    static struct option long_options[] =
    {
        {"validation", no_argument, 0, 'v'},
        {"debug", no_argument, 0, 'g'},
        {"model", required_argument, 0, 'm'},
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
    while ((c = getopt_long(argc, argv, "vgmfubdot", // TODO
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
            case 'm':
                extern_model = optarg;
                break;
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
                if (optarg[0] == 'c') {
                    std::cout << "INFO: Using OpenCL backend" << std::endl;
                    dev.device_type = kDLOpenCL;
                }
                else {
                    std::cout << "INFO: Using CPU backend" << std::endl;
                }
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
        if (extern_model) {
            args.model = extern_model;
        }
        else {
            args.model = model.first;
        }
        if (only_test && strcmp(only_test, "ALL") && args.model.find(only_test) == std::string::npos) {
            continue;
        }

        args.input_size = model.second;

        std::cout << "Running TVM graph executor: " << args.model << std::endl;
        std::string model_file = ".tvm/" + args.model + ".tar.so";
        if (model_exists(model_file) == 0) {
            std::cerr << args.model << " model doesn't exist!!!" << std::endl;
            continue;
        }

        tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile(model_file);
        // create the graph executor module
        tvm::runtime::Module gmod = mod_factory.GetFunction("default")(dev);

        tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
        tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");
        tvm::runtime::PackedFunc run = gmod.GetFunction("run");
        // Use the C++ API
        tvm::runtime::NDArray input_tensor  =
        tvm::runtime::NDArray::Empty({1, 3, args.input_size, args.input_size}, DLDataType{kDLFloat, 32, 1}, dev);
        tvm::runtime::NDArray output_tensor =
        tvm::runtime::NDArray::Empty({1, 1000}, DLDataType{kDLFloat, 32, 1}, dev);

        if (args.validation) {
            evaluate(set_input, get_output, run, input_tensor, output_tensor);
        }
        else {
            benchmark(set_input, get_output, run, input_tensor, output_tensor);
        }

        if (extern_model) {
            break;
        }
    }
}
