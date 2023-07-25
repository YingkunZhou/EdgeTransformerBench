//https://www.tensorflow.org/lite/guide/inference

#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <iomanip>
#include <filesystem>
#include <getopt.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include "utils.h"

using namespace tflite;

const int WARMUP_SEC = 5;
const int TEST_SEC = 20;

struct {
  std::string model;
  bool validation;
  int input_size;
  int batch_size;
  std::string data_path;
} args;

void evaluate(
    std::unique_ptr<Interpreter> &interpreter)
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
            float *input_tensor = interpreter->typed_input_tensor<float>(0);
            load_image(image.path(), input_tensor, args.model, args.input_size, args.batch_size);
            interpreter->Invoke();
            float *output_tensor = interpreter->typed_output_tensor<float>(0);
            num_predict++;
            bool acc1 = false;
            num_acc5 += acck(output_tensor, 5, class_index*scale+offset, acc1);
            num_acc1 += acc1;
        }
        class_index++;
        std::cout << "Done [" << class_index << "/" << classes.size() << "]";
        std::cout << "\tacc1: " << num_acc1*1.f/num_predict;
        std::cout << "\tacc5: " << num_acc5*1.f/num_predict << std::endl;
    }
}

void benchmark(
    std::unique_ptr<Interpreter> &interpreter)
{
    // Measure latency
    float *input_tensor = interpreter->typed_input_tensor<float>(0);
    load_image("daisy.jpg", input_tensor, args.model, args.input_size, args.batch_size);

    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &end);
    clock_gettime(CLOCK_REALTIME, &start);

    while (end.tv_sec - start.tv_sec < WARMUP_SEC) {
        interpreter->Invoke();
        clock_gettime(CLOCK_REALTIME, &end);
    }

    float *output_tensor = interpreter->typed_output_tensor<float>(0);
    print_topk(output_tensor, 3);

    std::vector<double> time_list = {};
    while (std::reduce(time_list.begin(), time_list.end()) < TEST_SEC) {
        clock_gettime(CLOCK_REALTIME, &start);
        interpreter->Invoke();
        clock_gettime(CLOCK_REALTIME, &end);
        long long seconds = end.tv_sec - start.tv_sec;
        long long nanoseconds = end.tv_nsec - start.tv_nsec;
        double elapse = seconds + nanoseconds * 1e-9;
        time_list.push_back(elapse);
    }

    double time_max = *std::max_element(time_list.begin(), time_list.end()) * 1000;
    double time_min = *std::min_element(time_list.begin(), time_list.end()) * 1000;
    double time_mean = std::reduce(time_list.begin(), time_list.end()) * 1000 / time_list.size();
    std::sort(time_list.begin(), time_list.end());
    double time_median = time_list[time_list.size() / 2] * 1000;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "min =\t" << time_min << "ms\tmax =\t" << time_max << "ms\tmean =\t" << time_mean << "ms\tmedian =\t" << time_median << "ms" << std::endl;
}

int main(int argc, char* argv[])
{
    args.data_path = "imagenet-div50";
    args.validation = false;
    args.batch_size = 1;
    bool debug = false;
    char* arg_long = nullptr;
    char* only_test = nullptr;

    static struct option long_options[] =
    {
        {"validation", no_argument, 0, 'v'},
        {"debug", no_argument, 0, 'g'},
        {"batch-size", required_argument, 0, 'b'},
        {"data-path",  required_argument, 0, 'd'},
        {"only-test",  required_argument, 0, 'o'},
        {"append",  required_argument, 0, 0},
        {0, 0, 0, 0}
    };
    int option_index;
    int c;
    while ((c = getopt_long(argc, argv, "vbdo", long_options, &option_index)) != -1)
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
            case '?':
                std::cout << "Got unknown option." << std::endl;
                break;
            default:
                std::cout << "Got unknown parse returns: " << c << std::endl;
        }
    }

    int num_threads = 1; // TODO
    ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<Interpreter> interpreter;

    for (const auto & model: test_models) {
        args.model = model.first;
        if (args.model.find("EMO") != std::string::npos) {
            continue;
        }
        if (only_test && args.model.find(only_test) == std::string::npos) {
            continue;
        }

        args.input_size = model.second;
        std::string model_file = "tflite/";
        model_file.append(args.model);
        model_file.append(".tflite");

        // create a interpreter
        std::unique_ptr<FlatBufferModel> tflite_model = FlatBufferModel::BuildFromFile(model_file.c_str());
        InterpreterBuilder builder(*tflite_model, resolver);
        builder.SetNumThreads(num_threads);
        builder(&interpreter);
        interpreter->AllocateTensors();

        if (args.validation) {
            evaluate(interpreter);
        }
        else {
            benchmark(interpreter);
        }
    }
}
