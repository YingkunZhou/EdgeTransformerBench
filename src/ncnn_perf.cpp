/* Reference code:
   https://github.com/Tencent/ncnn/blob/master/examples/shufflenetv2.cpp
   https://github.com/Tencent/ncnn/blob/master/benchmark/benchncnn.cpp
*/

#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <iomanip>
#include <filesystem>
#include <getopt.h>
#include <net.h>
#include <cpu.h>
#include <gpu.h>
#include "utils.h"

#include <chrono>
using namespace std::chrono;

#if !defined(DEBUG_C)
#define DEBUG_C 3
#endif

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

#if NCNN_VULKAN
// for -D NCNN_VULKAN=ON built ncnn
static ncnn::VulkanDevice* g_vkdev = 0;
static ncnn::VkAllocator* g_blob_vkallocator = 0;
static ncnn::VkAllocator* g_staging_vkallocator = 0;
#endif // NCNN_VULKAN

struct {
  std::string model;
  bool validation;
  int input_size;
  int batch_size;
  std::string data_path;
  const char* input_name;
  const char* output_name;
  ncnn::Mat output_tensor;
} args;

#define USE_NCNN
#include "evaluate.tcc"
#include "benchmark.tcc"

int main(int argc, char* argv[])
{
    args.data_path = "imagenet-div50";
    args.validation = false;
    args.batch_size = 1;
    bool debug = false;
    bool use_vulkan = false;
    char* arg_long = nullptr;
    char* only_test = nullptr;
    int num_threads = 1;
    int fpbits = 32;

    static struct option long_options[] =
    {
        {"validation", no_argument, 0, 'v'},
        {"debug", no_argument, 0, 'g'},
        {"fp", required_argument, 0, 'f'},
        {"backend", required_argument, 0, 'u'},
        {"batch-size", required_argument, 0, 'b'},
        {"data-path",  required_argument, 0, 'd'},
        {"only-test",  required_argument, 0, 'o'},
        {"threads",  required_argument, 0, 't'},
        {"append",  required_argument, 0, 0},
        {0, 0, 0, 0}
    };
    int option_index;
    int c;
    while ((c = getopt_long(argc, argv, "vgfubdot", // TODO
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
                debug = true;
                break;
            case 'u':
                if (optarg[0] == 'v') {
                    use_vulkan = true;
                    std::cout << "INFO: Using Vulkan backend" << std::endl;
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

    g_blob_pool_allocator.set_size_compare_ratio(0.f);
    g_workspace_pool_allocator.set_size_compare_ratio(0.f);
#if NCNN_VULKAN
    if (use_vulkan)
    {
        int gpu_device = 0; //TODO
        g_vkdev = ncnn::get_gpu_device(gpu_device);
        g_blob_vkallocator = new ncnn::VkBlobAllocator(g_vkdev);
        g_staging_vkallocator = new ncnn::VkStagingAllocator(g_vkdev);
    }
#endif // NCNN_VULKAN

    ncnn::set_omp_dynamic(0);
    // TODO: doesn't work?
    // ncnn::set_omp_num_threads(num_threads);

    for (const auto & model: test_models) {
        args.model = model.first;
        if (only_test && strcmp(only_test, "ALL") && args.model.find(only_test) == std::string::npos) {
            continue;
        }
        // TODO
        args.input_size = model.second;

        g_blob_pool_allocator.clear();
        g_workspace_pool_allocator.clear();
#if NCNN_VULKAN
        if (use_vulkan)
        {
            g_blob_vkallocator->clear();
            g_staging_vkallocator->clear();
        }
#endif // NCNN_VULKAN

        char param_file[256];
        char model_file[256];
        sprintf(param_file, ".ncnn/" "%s.ncnn.param", args.model.c_str());
        sprintf(model_file, ".ncnn/" "%s.ncnn.bin", args.model.c_str());
        if (model_exists(model_file) == 0) {
            std::cerr << args.model << " model doesn't exist!!!" << std::endl;
            continue;
        }
        // create a net
        std::cout << "Creating ncnn net: " << args.model << std::endl;
        ncnn::Net net;
        net.opt.num_threads = num_threads;
    if (fpbits == 32) {
#if 0
        // https://github.com/Tencent/ncnn/blob/master/src/option.cpp#L46C18-L46C18
        std::cout << "opt status: " <<
        net.opt.use_fp16_packed     <<
        net.opt.use_fp16_storage    <<
        net.opt.use_fp16_arithmetic <<
        net.opt.use_bf16_storage    <<
        net.opt.use_int8_inference  <<
        net.opt.use_int8_packed     <<
        net.opt.use_int8_storage    <<
        net.opt.use_int8_arithmetic <<
        net.opt.use_packing_layout;
#endif
        net.opt.use_fp16_packed     = false;
        net.opt.use_fp16_storage    = false;
        net.opt.use_fp16_arithmetic = false;
        net.opt.use_bf16_storage    = false;
        net.opt.use_int8_inference  = false;
        net.opt.use_int8_packed     = false;
        net.opt.use_int8_storage    = false;
        net.opt.use_int8_arithmetic = false;
        //net.opt.use_packing_layout  = false;
#if 0
        std::cout << " ==> "        <<
        net.opt.use_fp16_packed     <<
        net.opt.use_fp16_storage    <<
        net.opt.use_fp16_arithmetic <<
        net.opt.use_bf16_storage    <<
        net.opt.use_int8_inference  <<
        net.opt.use_int8_packed     <<
        net.opt.use_int8_storage    <<
        net.opt.use_int8_arithmetic <<
        net.opt.use_packing_layout  << std::endl;
#endif
    }
#if NCNN_VULKAN
        if (use_vulkan)
        {
            net.opt.use_vulkan_compute = true; //TODO
            net.set_vulkan_device(g_vkdev);
        }
#endif // NCNN_VULKAN
        net.load_param(param_file);
        net.load_model(model_file);

#if defined(DEBUG)
        std::cout << "input_size: " << args.input_size << " channels: " << DEBUG_C << std::endl;
        ncnn::Mat input_tensor = ncnn::Mat(args.input_size, args.input_size, DEBUG_C);
#else
        ncnn::Mat input_tensor = ncnn::Mat(args.input_size, args.input_size, 3);
#endif
        args.input_name = net.input_names()[0];
        args.output_name = net.output_names()[0];
        if (args.validation) {
            evaluate(net, input_tensor);
        }
        else {
            benchmark(net, input_tensor);
        }
    }

#if NCNN_VULKAN
    if (use_vulkan)
    {
        delete g_blob_vkallocator;
        delete g_staging_vkallocator;
        ncnn::destroy_gpu_instance();
    }
#endif // NCNN_VULKAN
}
