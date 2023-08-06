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

const int WARMUP_SEC = 5;
const int TEST_SEC = 20;

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
} args;

void evaluate(ncnn::Net &net, ncnn::Mat &input_tensor)
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
    const std::vector<const char*>& input_names = net.input_names();
    const std::vector<const char*>& output_names = net.output_names();
    ncnn::Mat output_tensor;

    std::vector<std::filesystem::path> classes = traverse_class(args.data_path);
    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);
    for (const std::string& class_path : classes) {
        for (const auto & image: std::filesystem::directory_iterator(class_path)) {
            load_image(image.path(), (float *)input_tensor.data, args.model, args.input_size, args.batch_size);
            ncnn::Extractor ex = net.create_extractor();
            ex.input(input_names[0], input_tensor);
            ex.extract(output_names[0], output_tensor);
            num_predict++;
            bool acc1 = false;
            num_acc5 += acck((float *)output_tensor.data, 5, class_index*scale+offset, acc1);
            num_acc1 += acc1;
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

void benchmark(ncnn::Net &net, ncnn::Mat &input_tensor)
{
    // Measure latency
    const std::vector<const char*>& input_names = net.input_names();
    const std::vector<const char*>& output_names = net.output_names();
    // https://zhuanlan.zhihu.com/p/578501922
#if !defined(DEBUG)
    load_image("daisy.jpg", (float *)input_tensor.data, args.model, args.input_size, args.batch_size);
#else
    for (int i = 0; i < args.batch_size*3*args.input_size*args.input_size; i++)
        ((float *)input_tensor.data)[i] = 1;
#endif

    ncnn::Mat output_tensor;

    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &end);
    clock_gettime(CLOCK_REALTIME, &start);
    /// warmup
#if !defined(DEBUG) || !defined(TEST)
    while (end.tv_sec - start.tv_sec < WARMUP_SEC) {
#endif
        ncnn::Extractor ex = net.create_extractor();
        ex.input(input_names[0], input_tensor);
        ex.extract(output_names[0], output_tensor);
#if !defined(DEBUG) || !defined(TEST)
        clock_gettime(CLOCK_REALTIME, &end);
    }
#endif

#if defined(DEBUG)
    std::cout << ((float *)output_tensor.data)[0] << " " << ((float *)output_tensor.data)[1] << std::endl;
    return;
#endif
    print_topk((float *)output_tensor.data, 3);
#if defined(TEST)
    return;
#endif

    /// testup
    std::vector<double> time_list = {};
    double time_tot = 0;
    while (time_tot < TEST_SEC) {
        clock_gettime(CLOCK_REALTIME, &start);
        ncnn::Extractor ex = net.create_extractor();
        ex.input(input_names[0], input_tensor);
        ex.extract(output_names[0], output_tensor);
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
    std::cout << "min =\t" << time_min << "ms\tmax =\t" << time_max << "ms\tmean =\t";
    std::cout << time_mean << "ms\tmedian =\t" << time_median << "ms" << std::endl;}

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

    static struct option long_options[] =
    {
        {"validation", no_argument, 0, 'v'},
        {"debug", no_argument, 0, 'g'},
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
    while ((c = getopt_long(argc, argv, "vbdot", // TODO
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
                if (optarg[0] == 'v')
                    use_vulkan = true;
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

    int powersave = 2; //TODO
    ncnn::set_cpu_powersave(powersave);
    ncnn::set_omp_dynamic(0);
    // TODO: doesn't work?
    // ncnn::set_omp_num_threads(num_threads);

    for (const auto & model: test_models) {
        args.model = model.first;
        if (only_test && args.model.find(only_test) == std::string::npos) {
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
        sprintf(param_file, "ncnn/" "%s.ncnn.param", args.model.c_str());
        sprintf(model_file, "ncnn/" "%s.ncnn.bin", args.model.c_str());
        // create a net
        std::cout << "Creating ncnn net: " << args.model << std::endl;
        ncnn::Net net;
        net.opt.num_threads = num_threads;
#if NCNN_VULKAN
        if (use_vulkan)
        {
            net.opt.use_vulkan_compute = true; //TODO
            net.set_vulkan_device(g_vkdev);
        }
#endif // NCNN_VULKAN
        net.load_param(param_file);
        net.load_model(model_file);

        ncnn::Mat input_tensor = ncnn::Mat(args.input_size, args.input_size, 3);

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
