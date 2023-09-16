#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <iomanip>
#include <filesystem>
#include <getopt.h>

#include <torch/script.h>
#include <torch/csrc/jit/mobile/import.h>

#include "utils.h"

const int WARMUP_SEC = 5;
const int TEST_SEC = 20;

struct {
  std::string model;
  bool validation;
  int input_size;
  int batch_size;
  bool debug;
  std::string data_path;
} args;

// Some common guards for inference-only custom mobile LibTorch.
struct MobileCallGuard {
  // AutoGrad is disabled for mobile by default.
  torch::autograd::AutoGradMode no_autograd_guard{false};
  // VariableType dispatch is not included in default mobile build. We need set
  // this guard globally to avoid dispatch error (only for dynamic dispatch).
  // Thanks to the unification of Variable class and Tensor class it's no longer
  // required to toggle the NonVariableTypeMode per op - so it doesn't hurt to
  // always set NonVariableTypeMode for inference only use case.
  torch::AutoNonVariableTypeMode non_var_guard{true};
  // Disable graph optimizer to ensure list of unused ops are not changed for
  // custom mobile build.
  torch::jit::GraphOptimizerEnabledGuard no_optimizer_guard{false};
};

void evaluate(
  torch::jit::script::Module &module,
  torch::Tensor *input)
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
    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);
    for (const std::string& class_path : classes) {
        for (const auto & image: std::filesystem::directory_iterator(class_path)) {
            load_image(image.path(), input->host<float>(), args.model, args.input_size, args.batch_size);
            /////////////////////////////////
            MobileCallGuard guard;
            auto output = module.forward({input}).toTensor();
            /////////////////////////////////
            num_predict++;
            bool acc1 = false;
            num_acc5 += acck(/*here*/output.data_ptr<float>(), 5, class_index*scale+offset, acc1);
            num_acc1 += acc1;
            delete output;
        }
        class_index++;
        std::cout << "Done [" << class_index << "/" << classes.size() << "]";
        std::cout << "\tacc1: " << num_acc1*1.f/num_predict;
        std::cout << "\tacc5: " << num_acc5*1.f/num_predict << std::endl;
    }
    clock_gettime(CLOCK_REALTIME, &end);
    long long seconds = end.tv_sec - start.tv_sec;
    long long nanoseconds = end.tv_nsec - start.tv_nsec;
    double elapse = seconds + nanoseconds * 1e-9;
    std::cout << "elapse time: " << elapse << std::endl;
}

void benchmark(
    torch::jit::script::Module &module,
    torch::Tensor *input)
{
    // Measure latency
    load_image("daisy.jpg", input.data_ptr<float>(), args.model, args.input_size, args.batch_size);
    at::Tensor output;
    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &end);
    clock_gettime(CLOCK_REALTIME, &start);
    /// warmup
#if !defined(DEBUG) && !defined(TEST)
    while (end.tv_sec - start.tv_sec < WARMUP_SEC) {
#endif
        /////////////////////////////////
        MobileCallGuard guard;
        output = module.forward({input}).toTensor();
        /////////////////////////////////
#if !defined(DEBUG) && !defined(TEST)
        clock_gettime(CLOCK_REALTIME, &end);
    }
#endif

    print_topk(output.data_ptr<float>(), 3);
#if defined(TEST)
    return;
#endif

    /// testup
    std::vector<double> time_list = {};
    double time_tot = 0;
    while (time_tot < TEST_SEC) {
        clock_gettime(CLOCK_REALTIME, &start);
        /////////////////////////////////
        MobileCallGuard guard;
        output = module.forward({input}).toTensor();
        /////////////////////////////////
        clock_gettime(CLOCK_REALTIME, &end);
        double elapse = end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) * 1e-9;
        time_tot += elapse;
        time_list.push_back(elapse);
    }

    double forward_max = *std::max_element(time_list.begin(), time_list.end()) * 1000;
    double forward_min = *std::min_element(time_list.begin(), time_list.end()) * 1000;
    double forward_median = time_list[time_list.size() / 2] * 1000;
    double forward_mean = forward_tot * 1000 / time_list.size();
    double time_mean = time_tot * 1000 / time_list.size();
    std::sort(time_list.begin(), time_list.end());

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "[" << time_list.size() << " iters]";
    std::cout << " min ="   << std::setw(7) << forward_min  << "ms";
    std::cout << " max ="   << std::setw(7) << forward_max  << "ms";
    std::cout << " median ="<< std::setw(7) << forward_median<<"ms";
    std::cout << " mean ="  << std::setw(7) << forward_mean << "ms";
    std::cout << " mean ="  << std::setw(7) << time_mean << "ms" << std::endl;
}

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
        if (only_test && args.model.find(only_test) == std::string::npos) {
            continue;
        }

        args.input_size = model.second;
        //////////////////////////////////////////////////
        std::cout << "Creating PyTorch Interpreter: " << args.model << std::endl;
        std::string model_file = ".pt/" + args.model + ".pt";
        MobileCallGuard guard;
        torch::jit::script::Module module = torch::jit::load(model_file);
        module.eval();
        torch::Tensor input = torch::rand({args.batch_size, 3, args.input_size, args.input_size});
        //////////////////////////////////////////////////
        if (args.validation) {
            evaluate(module, input);
        }
        else {
            benchmark(module, input);
        }
    }
}
