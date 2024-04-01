/* Reference code:
   https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/demo/cxx/mobile_classify/mobile_classify.cc
   https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/demo/cxx/mobile_light/mobilenetv1_light_api.cc
*/

#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <iomanip>
#include <filesystem>
#include <getopt.h>
#include <fstream>
#include <cstring>

#include <paddle_api.h>
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

#define USE_PDLITE
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
    char backend = 'c';

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
                backend = optarg[0];
                if (optarg[0] == 'o') {
                    std::cout << "INFO: Using OpenCL backend" << std::endl;
                }
                else if (optarg[0] == 'n') {
                    std::cout << "INFO: Using NNAPI backend" << std::endl;
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
    // TODO:
    int power_mode = paddle::lite_api::LITE_POWER_NO_BIND;
    std::string nnadapter_context_properties;
    std::vector<std::string> nnadapter_device_names;
    nnadapter_device_names.emplace_back("android_nnapi");

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

        std::string model_file = ".pdlite/" + args.model + ".nb";
        if (model_exists(model_file) == 0) {
            std::cerr << args.model << " model doesn't exist!!!" << std::endl;
            continue;
        }
        std::cout << "Creating PaddlePredictor: " << args.model << std::endl;
        paddle::lite_api::MobileConfig config;
        // 1. Set MobileConfig
        config.set_model_from_file(model_file);

        // NOTE: Use android gpu with opencl, you should ensure:
        //  first, [compile **cpu+opencl** paddlelite
        //    lib](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/docs/demo_guides/opencl.md);
        //  second, [convert and use opencl nb
        //    model](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/docs/user_guides/opt/opt_bin.md).

        bool is_opencl_backend_valid =
            ::paddle::lite_api::IsOpenCLBackendValid(/*check_fp16_valid = false*/);
        std::cout << "is_opencl_backend_valid:"
                    << (is_opencl_backend_valid ? "true" : "false") << std::endl;
        if (is_opencl_backend_valid) {
            // Set opencl kernel binary.
            // Large addtitional prepare time is cost due to algorithm selecting and
            // building kernel from source code.
            // Prepare time can be reduced dramitically after building algorithm file
            // and OpenCL kernel binary on the first running.
            // The 1st running time will be a bit longer due to the compiling time if
            // you don't call `set_opencl_binary_path_name` explicitly.
            // So call `set_opencl_binary_path_name` explicitly is strongly
            // recommended.

            // Make sure you have write permission of the binary path.
            // We strongly recommend each model has a unique binary name.
            const std::string bin_path = ".pdlite/";
            const std::string bin_name = args.model + "_kernel.bin";
            config.set_opencl_binary_path_name(bin_path, bin_name);

            // opencl tune option
            // CL_TUNE_NONE: 0
            // CL_TUNE_RAPID: 1
            // CL_TUNE_NORMAL: 2
            // CL_TUNE_EXHAUSTIVE: 3
            const std::string tuned_path = ".pdlite/";
            const std::string tuned_name = args.model + "_tuned.bin";
            config.set_opencl_tune(paddle::lite_api::CL_TUNE_NORMAL, tuned_path, tuned_name);

            // opencl precision option
            // CL_PRECISION_AUTO: 0, first fp16 if valid, default
            // CL_PRECISION_FP32: 1, force fp32
            // CL_PRECISION_FP16: 2, force fp16
            if (fpbits == 32) {
                config.set_opencl_precision(paddle::lite_api::CL_PRECISION_FP32);
            } else {
                config.set_opencl_precision(paddle::lite_api::CL_PRECISION_FP16);
            }
        } else {
            std::cout << "*** nb model will be running on cpu. ***" << std::endl;
            // you can give backup cpu nb model instead
            // config.set_model_from_file(cpu_nb_model_dir);
        }

        config.set_threads(num_threads);
        config.set_power_mode(static_cast<paddle::lite_api::PowerMode>(power_mode));

        // https://www.paddlepaddle.org.cn/lite/develop/demo_guides/android_nnapi.html
        // https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/tests/api/test_resnet50_fp32_v1_8_nnadapter.cc
        if (backend == 'n') {
            config.set_nnadapter_device_names(nnadapter_device_names);
            config.set_nnadapter_context_properties(nnadapter_context_properties);
        }
        // 2. Create PaddlePredictor by MobileConfig
        std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor =
            paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::MobileConfig>(config);

        // 3. Prepare input data from image
        std::unique_ptr<paddle::lite_api::Tensor> input_tensor(std::move(predictor->GetInput(0)));
        input_tensor->Resize({1, 3, args.input_size, args.input_size});

        if (args.validation) {
            evaluate(predictor, input_tensor);
        }
        else {
            benchmark(predictor, input_tensor);
        }

        if (extern_model) {
            break;
        }
    }
}
