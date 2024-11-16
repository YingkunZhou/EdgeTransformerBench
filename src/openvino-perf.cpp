/* Reference code:
   https://github.com/openvinotoolkit/openvino/blob/master/samples/cpp/benchmark/sync_benchmark/main.cpp
   https://github.com/openvinotoolkit/openvino/tree/master/samples/cpp/classification_sample_async
   https://github.com/openvinotoolkit/openvino/blob/master/samples/cpp/hello_query_device/README.md
   https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes.html
   https://docs.openvino.ai/2024/openvino-workflow/model-optimization.html
*/

#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <iomanip>
#include <filesystem>
#include <getopt.h>
#include <cstring>

#include <openvino/openvino.hpp>
#include <openvino/runtime/intel_npu/properties.hpp>
#include "utils.h"
#include <chrono>
using namespace std::chrono;

struct {
  std::string model;
  bool validation;
  int input_size;
  int batch_size;
  std::string data_path;
} args;

#define USE_OPENVINO
#include "evaluate.tcc"
#include "benchmark.tcc"

int main(int argc, char* argv[])
{
    args.data_path = "imagenet-div50";
    args.validation = false;
    args.batch_size = 1;
    std::string device_name;
    char* arg_long = nullptr;
    char* only_test = nullptr;
    char* extern_model = nullptr;
    int extern_size = 224;
    int num_threads = 1;
    int fpbits = 32;

    static struct option long_options[] =
    {
        {"validation", no_argument, 0, 'v'},
        {"model", required_argument, 0, 'm'},
        {"size", required_argument, 0, 's'},
        {"fp", required_argument, 0, 'f'},
        {"backend",  required_argument, 0, 'u'},
        {"batch-size", required_argument, 0, 'b'},
        {"data-path",  required_argument, 0, 'd'},
        {"only-test",  required_argument, 0, 'o'},
        {"threads",  required_argument, 0, 't'},
        {0, 0, 0, 0}
    };
    int option_index;
    int c;
    while ((c = getopt_long(argc, argv, "vgmsfubot", // TODO
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
            case 's':
                extern_size = atoi(optarg);
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
            case 'u':
                if (optarg[0] == 'g') {
                    device_name = "GPU";
                    std::cout << "INFO: Using GPU backend" << std::endl;
                }
                else if (optarg[0] == 'n') {
                    device_name = "NPU";
                    std::cout << "INFO: Using NPU backend" << std::endl;
                }
                else {
                    device_name = "CPU";
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

    ov::AnyMap device_config = {};
    // Mutable: PERFORMANCE_HINT : LATENCY
    // Mutable: EXECUTION_MODE_HINT : PERFORMANCE
    // Optimize for latency. Most of the devices are configured for latency by default,
    // but there are exceptions like GNA
    device_config.insert(ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
    if (device_name == "CPU") {
        // device_config.insert(ov::hint::enable_hyper_threading(true)); // Damn, worse performace!
        device_config.insert(ov::inference_num_threads(num_threads));
        // device_config.insert(ov::hint::scheduling_core_type(ov::hint::SchedulingCoreType::PCORE_ONLY)); // ECORE_ONLY
    }
    else if (device_name == "NPU") {
        // https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html
        // TODO: has no difference?!
        device_config.insert(ov::intel_npu::compilation_mode_params(ov::Any("optimization-level=2 performance-hint-override=latency")));
        device_config.insert(ov::intel_npu::turbo(true));
    }
    device_config.insert(ov::hint::inference_precision(fpbits == 16? ov::element::f16 : ov::element::f32));

    for (const auto & model: test_models) {
        if (extern_model) {
            args.model = extern_model;
            args.input_size = extern_size;
        }
        else {
            args.model = model.first;
            args.input_size = model.second;
        }
        if (only_test && strcmp(only_test, "ALL") && args.model.find(only_test) == std::string::npos) {
            continue;
        }

        std::cout << "Creating OpenVino infer request: " << args.model << std::endl;
        std::string model_file = ".xml/" + args.model + ".xml";
        if (model_exists(model_file) == 0) {
            std::cerr << args.model << " model doesn't exist!!!" << std::endl;
            continue;
        }
        // Create ov::Core and use it to compile a model.
        // Select the device by providing the name as the second parameter to CLI.
        // Using MULTI device is pointless in sync scenario
        // because only one instance of ov::InferRequest is used
        ov::Core core;
        ov::CompiledModel compiled_model = core.compile_model(model_file, device_name, device_config);
        ov::InferRequest ireq = compiled_model.create_infer_request();
        ov::Tensor input_tensor = ireq.get_input_tensor();

        if (args.validation) {
            evaluate(ireq, input_tensor);
        }
        else {
            benchmark(ireq, input_tensor);
        }

        if (extern_model) {
            break;
        }
    }
}
