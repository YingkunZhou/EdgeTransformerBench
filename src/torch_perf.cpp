#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <iomanip>
#include <filesystem>
#include <getopt.h>

#include <torch/script.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/api/include/torch/utils.h>

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
  at::Tensor output;
} args;

// Some common guards for inference-only custom mobile LibTorch.
struct MobileCallGuard {
#if !defined(USE_TORCH_MOBILE) // TODO: mobile model without gaurd?
  // AutoGrad is disabled for mobile by default.
  torch::autograd::AutoGradMode no_autograd_guard{false};
  // VariableType dispatch is not included in default mobile build. We need set
  // this guard globally to avoid dispatch error (only for dynamic dispatch).
  // Thanks to the unification of Variable class and Tensor class it's no longer
  // required to toggle the NonVariableTypeMode per op - so it doesn't hurt to
  // always set NonVariableTypeMode for inference only use case.
  // torch::AutoNonVariableTypeMode non_var_guard{true};
  c10::InferenceMode guard{true};
  // Disable graph optimizer to ensure list of unused ops are not changed for
  // custom mobile build.
  torch::jit::GraphOptimizerEnabledGuard no_optimizer_guard{false};
#endif
};

#define USE_TORCH
#include "evaluate.tcc"
#include "benchmark.tcc"

int main(int argc, char* argv[])
{
    args.data_path = "imagenet-div50";
    args.validation = false;
    args.batch_size = 1;
    args.debug = false;
    char backend = ' ';
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
    // https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html
    at::init_num_threads();
    at::set_num_threads(num_threads);
    //at::set_num_interop_threads(num_threads); //TODO
    if (backend == 'n') {
        std::cout << "INFO: Using NNAPI backend" << std::endl;
    }
    else if (backend == 'c') {
        std::cout << "INFO: Using mobile CPU backend" << std::endl;
    }
    else if (backend == 'v') {
        std::cout << "INFO: Using Vulkan backend" << std::endl;
    }
    else {
        std::cout << "INFO: Using trace CPU backend" << std::endl;
    }

    for (const auto & model: test_models) {
        args.model = model.first;
        if (only_test && strcmp(only_test, "ALL") && args.model.find(only_test) == std::string::npos) {
            continue;
        }

        args.input_size = model.second;
        std::string model_file;
        if (backend == 'n') {
          model_file = ".pt/" + args.model + ".n.ptl";
        }
        else if (backend == 'c') {
          model_file = ".pt/" + args.model + ".c.ptl";
        }
        else if (backend == 'v') {
          model_file = ".pt/" + args.model + ".v.ptl";
        }
        else {
          model_file = ".pt/" + args.model + ".pt";
        }

        if (model_exists(model_file) == 0) {
            std::cerr << args.model << " model doesn't exist!!!" << std::endl;
            continue;
        }
        std::cout << "Creating pytorch module: " << args.model << std::endl;
#if defined(USE_TORCH_MOBILE)
        auto module = torch::jit::_load_for_mobile(model_file);
#else
        MobileCallGuard guard;
        torch::jit::script::Module module = torch::jit::load(model_file);
        module.eval();
#endif
        torch::Tensor input = torch::rand({args.batch_size, 3, args.input_size, args.input_size});
        if (args.validation) {
            evaluate(module, input);
        }
        else {
            benchmark(module, input);
        }
    }
}
