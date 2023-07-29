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
#include <fstream>

#include <MNN/Interpreter.hpp>
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

void evaluate(
    std::shared_ptr<MNN::Interpreter> &net,
    MNN::Session *session,
    MNN::Tensor *input_tensor)
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

    MNN::Tensor* input = net->getSessionInput(session, NULL);

    std::vector<std::filesystem::path> classes = traverse_class(args.data_path);
    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);
    for (const std::string& class_path : classes) {
        for (const auto & image: std::filesystem::directory_iterator(class_path)) {
            load_image(image.path(), input_tensor->host<float>(), args.model, args.input_size, args.batch_size);
            input->copyFromHostTensor(input_tensor);
            net->runSession(session);
            MNN::Tensor* output = net->getSessionOutput(session, NULL);
            auto dimType = output->getDimensionType();
            if (output->getType().code != halide_type_float) {
                dimType = MNN::Tensor::TENSORFLOW;
            }
            auto output_tensor = new MNN::Tensor(output, dimType);
            output->copyToHostTensor(output_tensor);
            num_predict++;
            bool acc1 = false;
            num_acc5 += acck(output_tensor->host<float>(), 5, class_index*scale+offset, acc1);
            num_acc1 += acc1;
            delete output_tensor;
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
    std::shared_ptr<MNN::Interpreter> &net,
    MNN::Session *session,
    MNN::Tensor *input_tensor)
{
    auto input = net->getSessionInput(session, NULL);
    auto output = net->getSessionOutput(session, NULL);

    // Measure latency
    load_image("daisy.jpg", input_tensor->host<float>(), args.model, args.input_size, args.batch_size);

    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &end);
    clock_gettime(CLOCK_REALTIME, &start);
    /// warmup
    while (end.tv_sec - start.tv_sec < WARMUP_SEC) {
        // TODO: runSession will overwirte the value in input_tensor!!!
        input->copyFromHostTensor(input_tensor);
        net->runSession(session);
        clock_gettime(CLOCK_REALTIME, &end);
    }

    auto dimType = output->getDimensionType();
    if (output->getType().code != halide_type_float) {
        dimType = MNN::Tensor::TENSORFLOW;
    }

    auto output_tensor = new MNN::Tensor(output, dimType);
    output->copyToHostTensor(output_tensor);
    print_topk(output_tensor->host<float>(), 3);
    delete output_tensor;

    /// testup
    std::vector<double> time_list = {};
    double time_tot = 0;
    while (time_tot < TEST_SEC) {
        clock_gettime(CLOCK_REALTIME, &start);
        input->copyFromHostTensor(input_tensor);
        net->runSession(session);
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
    std::cout << time_mean << "ms\tmedian =\t" << time_median << "ms" << std::endl;
}

int main(int argc, char* argv[])
{
    args.data_path = "imagenet-div50";
    args.validation = false;
    args.batch_size = 1;
    args.debug = false;
    char* arg_long = nullptr;
    char* only_test = nullptr;
    int num_threads = 1;

    static struct option long_options[] =
    {
        {"validation", no_argument, 0, 'v'},
        {"debug", no_argument, 0, 'g'},
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
                args.debug = true;
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


    int forward = MNN_FORWARD_CPU;
    int precision = 2;

    for (const auto & model: test_models) {
        args.model = model.first;
        if (only_test && args.model.find(only_test) == std::string::npos) {
            continue;
        }

        args.input_size = model.second;

        std::cout << "Creating MNN Interpreter: " << args.model << std::endl;
        std::string model_file = "mnn/" + args.model + ".mnn";
        //std::shared_ptr<MNN::Interpreter> net(MNN::Interpreter::createFromFile(model_file.c_str()), MNN::Interpreter::destroy);
        std::shared_ptr<MNN::Interpreter> net(MNN::Interpreter::createFromFile(model_file.c_str()));
#if 1
        net->setCacheFile(".cachefile");
        net->setSessionMode(MNN::Interpreter::Session_Backend_Auto);
        net->setSessionHint(MNN::Interpreter::MAX_TUNING_NUMBER, 5);
#else
        net->setSessionMode(MNN::Interpreter::Session_Release);
#endif

        MNN::ScheduleConfig config;
#if 1
        config.type  = MNN_FORWARD_AUTO;
#else
        config.type = static_cast<MNNForwardType>(forward);
        config.numThread = num_threads;
        MNN::BackendConfig backendConfig;
        backendConfig.precision = (MNN::BackendConfig::PrecisionMode) precision;
        backendConfig.power = MNN::BackendConfig::Power_High;
        config.backendConfig = &backendConfig;
#endif
        auto session = net->createSession(config);

        if (args.debug) {
            float memoryUsage = 0.0f;
            float flops = 0.0f;
            int backendType[2]; // TODO: 2?
            net->getSessionInfo(session, MNN::Interpreter::MEMORY, &memoryUsage);
            net->getSessionInfo(session, MNN::Interpreter::FLOPS, &flops);
            net->getSessionInfo(session, MNN::Interpreter::BACKENDS, backendType);
            MNN_PRINT("Session Info: memory use %f MB, flops is %f M, backendType is %d, batch size = %d\n", memoryUsage, flops, backendType[0], args.batch_size);
        }

        auto input = net->getSessionInput(session, NULL);
        //auto shape = input->shape();
        //shape[0] = args.batch_size; //e.g. Set Batch Size
        //std::vector<int> shape{1, 3, 224, 224}; //or
        //net->resizeTensor(input, shape);
        //net->resizeSession(session);
        net->releaseModel(); //TODO: ?

        auto input_tensor = new MNN::Tensor(input, MNN::Tensor::CAFFE);
        if (args.validation) {
            evaluate(net, session, input_tensor);
        }
        else {
            benchmark(net, session, input_tensor);
        }
    }
}
