const int WARMUP_SEC = 5;
const int TEST_SEC = 20;

#if defined(USE_PERF)
// https://learn.arm.com/learning-paths/servers-and-cloud-computing/arm_pmu/perf_event_open/
#include <linux/perf_event.h> /* Definition of PERF_* constants */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h> /* Definition of SYS_* constants */
#include <unistd.h>
#include <inttypes.h>
#include "arm_pmuv3.h"

// Executes perf_event_open syscall and makes sure it is succesful or exit
static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid, int cpu, int group_fd, unsigned long flags){
  int fd;
  fd = syscall(SYS_perf_event_open, hw_event, pid, cpu, group_fd, flags);
  if (fd == -1) {
    fprintf(stderr, "Error creating event");
    exit(EXIT_FAILURE);
  }

  return fd;
}
#endif

#if defined(USE_NCNN)
void benchmark(ncnn::Net &net, ncnn::Mat &input_tensor)
#endif
#if defined(USE_MNN)
void benchmark(
    std::shared_ptr<MNN::Interpreter> &net,
    MNN::Session *session,
    MNN::Tensor *input_tensor)
#endif
#if defined(USE_TNN)
void benchmark(
    tnn::TNN &net,
    std::shared_ptr<tnn::Instance> &instance,
    std::vector<float> &input)
#endif
#if defined(USE_PDLITE)
void benchmark(
    std::shared_ptr<paddle::lite_api::PaddlePredictor> &predictor,
    std::unique_ptr<paddle::lite_api::Tensor> &input_tensor)
#endif
#if defined(USE_TFLITE)
void benchmark(
    std::unique_ptr<Interpreter> &interpreter)
#endif
#if defined(USE_ONNXRUNTIME)
void benchmark(
    Ort::Session &session,
    std::vector<float> &input_tensor,
    std::vector<float> &output_tensor)
#endif
#if defined(USE_TORCH)
void benchmark(
#if defined(USE_TORCH_MOBILE)
    torch::jit::mobile::Module
#else
    torch::jit::script::Module
#endif
    &module,
    torch::Tensor &input)
#endif
#if defined(USE_TVM)
void benchmark(
    tvm::runtime::PackedFunc &set_input,
    tvm::runtime::PackedFunc &get_output,
    tvm::runtime::PackedFunc &run,
    tvm::runtime::NDArray &input_tensor,
    tvm::runtime::NDArray &output_tensor)
#endif
{
    // Measure latency
#if defined(USE_NCNN)
    // https://zhuanlan.zhihu.com/p/578501922
#if !defined(DEBUG)
    load_image("daisy.jpg", (float *)input_tensor.data, args.model, args.input_size, args.batch_size);
#else
    for (int i = 0; i < input_tensor.total(); i++)
        ((float *)input_tensor.data)[i] = 1;
#endif
#endif
#if defined(USE_MNN)
#if !defined(DEBUG)
    load_image("daisy.jpg", input_tensor->host<float>(), args.model, args.input_size, args.batch_size);
#else
    for (int i = 0; i < input_tensor->size() / 4; i++)
        input_tensor->host<float>()[i] = 1;
#endif
#endif
#if defined(USE_TNN)
    load_image("daisy.jpg", input.data(), args.model, args.input_size, args.batch_size);
    auto input_tensor = std::make_shared<tnn::Mat>(tnn::DEVICE_NAIVE, tnn::NCHW_FLOAT, args.input_dims, input.data());
    auto status = instance->SetInputMat(input_tensor, tnn::MatConvertParam());
#endif
#if defined(USE_PDLITE)
    load_image("daisy.jpg", input_tensor->mutable_data<float>(), args.model, args.input_size, args.batch_size);
#endif
#if defined(USE_TFLITE)
    float *input_tensor = interpreter->typed_input_tensor<float>(0);
    load_image("daisy.jpg", input_tensor, args.model, args.input_size, args.batch_size);
#endif
#if defined(USE_ONNXRUNTIME)
    load_image("daisy.jpg", input_tensor.data(), args.model, args.input_size, args.batch_size);
#endif
#if defined(USE_TORCH)
    load_image("daisy.jpg", input.data_ptr<float>(), args.model, args.input_size, args.batch_size);
#endif
#if defined(USE_TVM)
    load_image("daisy.jpg", static_cast<float*>(input_tensor->data), args.model, args.input_size, args.batch_size);
    set_input("input", input_tensor);
#endif

    auto start = high_resolution_clock::now();
    auto stop = high_resolution_clock::now();
    /// warmup
    while (duration_cast<seconds>(stop - start).count() < WARMUP_SEC) {
#if defined(USE_NCNN)
        ncnn::Extractor ex = net.create_extractor();
        ex.input(args.input_name, input_tensor);
        ex.extract(args.output_name, args.output_tensor);
#endif
#if defined(USE_MNN)
        // runSession will overwirte the value in input_tensor!!!
        // https://www.yuque.com/mnn/cn/create_session#KtfMk
        args.input->copyFromHostTensor(input_tensor);
        // https://www.yuque.com/mnn/cn/run_session#cy08Z ?
        net->runSession(session);
#endif
#if defined(USE_TNN)
        args.status = instance->Forward();
#endif
#if defined(USE_PDLITE)
        predictor->Run();
#endif
#if defined(USE_TFLITE)
        interpreter->Invoke();
#endif
#if defined(USE_ONNXRUNTIME)
        session.Run(Ort::RunOptions{nullptr},
                    args.input_name.data(), args.input.data() , 1 /*Number of inputs*/,
                    args.output_name.data(),args.output.data(), 1 /*Number of outputs*/
                    );
#endif
#if defined(USE_TORCH)
        args.output = module.forward({input}).toTensor();
#endif
#if defined(USE_TVM)
        run();
#endif
#if defined(DEBUG) || defined(TEST)
        break;
#endif
        stop = high_resolution_clock::now();
    }

#if defined(USE_NCNN)
#if defined(DEBUG)
    size_t len = args.output_tensor.total();
    std::cout << "[len: " << len << "] ";
    std::cout << "(0: " << ((float *)args.output_tensor.data)[0] << ") (1: " << ((float *)args.output_tensor.data)[1] << ") ";
    std::cout << "(-2:" << ((float *)args.output_tensor.data)[len-2] << ") (-1:" << ((float *)args.output_tensor.data)[len-1] << ")" << std::endl;
    return;
#endif
    print_topk((float *)args.output_tensor.data, 3);
#endif
#if defined(USE_MNN)
    auto output_tensor = new MNN::Tensor(args.output, args.output->getDimensionType());
    args.output->copyToHostTensor(output_tensor);
#if defined(DEBUG)
    size_t len = output_tensor->size() / 4;
    std::cout << "[len: " << len << "] ";
    std::cout << "(0: " << output_tensor->host<float>()[0] << ") (1: " << output_tensor->host<float>()[1] << ") ";
    std::cout << "(-2:" << output_tensor->host<float>()[len-2] << ") (-1:" << output_tensor->host<float>()[len-1] << ")" << std::endl;
    return;
#endif
    print_topk(output_tensor->host<float>(), 3);
    delete output_tensor;
#endif
#if defined(USE_TNN)
    args.status = instance->GetOutputMat(args.output_tensor);
    print_topk((float *)args.output_tensor->GetData(), 3);
#endif
#if defined(USE_PDLITE)
    std::unique_ptr<const paddle::lite_api::Tensor> output_tensor(std::move(predictor->GetOutput(0)));
    print_topk(output_tensor->data<float>(), 3);
#endif
#if defined(USE_TFLITE)
    float *output_tensor = interpreter->typed_output_tensor<float>(0);
    print_topk(output_tensor, 3);
#endif
#if defined(USE_ONNXRUNTIME)
    print_topk(output_tensor.data(), 3);
#endif
#if defined(USE_TORCH)
    print_topk(args.output.data_ptr<float>(), 3);
#endif
#if defined(USE_TVM)
    get_output(0, output_tensor);
    print_topk(static_cast<float*>(output_tensor->data), 3);
#endif
#if defined(TEST)
    return;
#endif

#if defined(USE_PERF)
    struct perf_event_attr pe0, pe1, pe2, pe3;
    int fd0, fd1, fd2, fd3;

    memset(&pe0, 0, sizeof(struct perf_event_attr));
    pe0.type = PERF_TYPE_RAW;
    pe0.size = sizeof(struct perf_event_attr);
    pe0.config = ARMV8_PMUV3_PERFCTR_INST_RETIRED;
    pe0.disabled = 1; pe0.pinned = 1; pe0.exclude_kernel = 1; pe0.exclude_hv = 1;

    memset(&pe1, 0, sizeof(struct perf_event_attr));
    pe1.type = PERF_TYPE_RAW;
    pe1.size = sizeof(struct perf_event_attr);
    pe1.config = ARMV8_PMUV3_PERFCTR_LD_RETIRED;
    pe1.disabled = 1; pe1.pinned = 1; pe1.exclude_kernel = 1; pe1.exclude_hv = 1;

    memset(&pe2, 0, sizeof(struct perf_event_attr));
    pe2.type = PERF_TYPE_RAW;
    pe2.size = sizeof(struct perf_event_attr);
    pe2.config = ARMV8_PMUV3_PERFCTR_L1D_CACHE_REFILL;
    pe2.disabled = 1; pe2.pinned = 1; pe2.exclude_kernel = 1; pe2.exclude_hv = 1;

    memset(&pe3, 0, sizeof(struct perf_event_attr));
    pe3.type = PERF_TYPE_RAW;
    pe3.size = sizeof(struct perf_event_attr);
    pe3.config = ARMV8_PMUV3_PERFCTR_BUS_ACCESS;
    pe3.disabled = 1; pe3.pinned = 1; pe3.exclude_kernel = 1; pe3.exclude_hv = 1;

    // Create the events
    fd0 = perf_event_open(&pe0, 0, -1, -1, 0);
    fd1 = perf_event_open(&pe1, 0, -1, -1, 0);
    fd2 = perf_event_open(&pe2, 0, -1, -1, 0);
    fd3 = perf_event_open(&pe3, 0, -1, -1, 0);
    //Reset counters and start counting
    ioctl(fd0, PERF_EVENT_IOC_RESET, 0); ioctl(fd0, PERF_EVENT_IOC_ENABLE, 0);
    ioctl(fd1, PERF_EVENT_IOC_RESET, 0); ioctl(fd1, PERF_EVENT_IOC_ENABLE, 0);
    ioctl(fd2, PERF_EVENT_IOC_RESET, 0); ioctl(fd2, PERF_EVENT_IOC_ENABLE, 0);
    ioctl(fd3, PERF_EVENT_IOC_RESET, 0); ioctl(fd3, PERF_EVENT_IOC_ENABLE, 0);
#endif
    /// testup
    std::vector<float> time_list = {};
    float time_tot = 0;
#if defined(USE_PERF)
    for (int i=0; i < 10; i++) {
#else
    while (time_tot < TEST_SEC) {
#endif
        start = high_resolution_clock::now();
#if defined(USE_NCNN)
        ncnn::Extractor ex = net.create_extractor();
        ex.input(args.input_name, input_tensor);
        ex.extract(args.output_name, args.output_tensor);
#endif
#if defined(USE_MNN)
#if 0
        // https://github.com/alibaba/MNN/blob/master/benchmark/benchmark.cpp style
        // will get better performance,
        // BUT will give meaningless output result!!!
        void* host = args.input->map(MNN::Tensor::MAP_TENSOR_WRITE,  args.input->getDimensionType());
        args.input->unmap(MNN::Tensor::MAP_TENSOR_WRITE,  args.input->getDimensionType(), host);
#else
        args.input->copyFromHostTensor(input_tensor);
#endif
        // net->runSessionWithCallBack(session, NULL, NULL, true);
        net->runSession(session);
#if 0
        host = args.output->map(MNN::Tensor::MAP_TENSOR_READ,  args.output->getDimensionType());
        args.output->unmap(MNN::Tensor::MAP_TENSOR_READ,  args.output->getDimensionType(), host);
#endif
#endif
#if defined(USE_TNN)
        status = instance->Forward();
#endif
#if defined(USE_PDLITE)
        predictor->Run();
#endif
#if defined(USE_TFLITE)
        interpreter->Invoke();
#endif
#if defined(USE_ONNXRUNTIME)
        session.Run(Ort::RunOptions{nullptr},
                    args.input_name.data(), args.input.data() , 1 /*Number of inputs*/,
                    args.output_name.data(),args.output.data(), 1 /*Number of outputs*/
                    );
#endif
#if defined(USE_TORCH)
        args.output = module.forward({input}).toTensor();
#endif
#if defined(USE_TVM)
        run();
#endif
        stop = high_resolution_clock::now();
        auto elapse = duration_cast<microseconds>(stop - start);
        time_list.push_back(elapse.count() / 1000.0);
        time_tot += elapse.count() / 1000000.0;
    }

#if defined(USE_PERF)
    // Stop counting
    ioctl(fd0, PERF_EVENT_IOC_DISABLE, 0);
    ioctl(fd1, PERF_EVENT_IOC_DISABLE, 0);
    ioctl(fd2, PERF_EVENT_IOC_DISABLE, 0);
    ioctl(fd3, PERF_EVENT_IOC_DISABLE, 0);
    // Read and print result
    uint64_t count[4];
    read(fd0, &count[0], sizeof(count[0]));
    read(fd1, &count[1], sizeof(count[1]));
    read(fd2, &count[2], sizeof(count[2]));
    read(fd3, &count[3], sizeof(count[3]));
    // Clean up file descriptor
    close(fd0); close(fd1); close(fd2); close(fd3);
    std::cout << "insn= " << count[0] << "/stall=  "<< count[1] << " == " << count[0]*1.0/count[1] << std::endl;
    std::cout << "insn= " << count[0] << "/miss=   "<< count[2] << " == " << count[0]*1.0/count[2] << std::endl;
    std::cout << "insn= " << count[0] << "/memory= "<< count[3] << " == " << count[0]*1.0/count[3] << std::endl;
#else
    float time_max = *std::max_element(time_list.begin(), time_list.end());
    float time_min = *std::min_element(time_list.begin(), time_list.end());
    float time_mean = time_tot * 1e3 / time_list.size();
    std::sort(time_list.begin(), time_list.end());
    float time_median = time_list[time_list.size() / 2];

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "[" << time_list.size() << " iters]";
    std::cout << " min ="   << std::setw(7) << time_min  << "ms";
    std::cout << " max ="   << std::setw(7) << time_max  << "ms";
    std::cout << " median ="<< std::setw(7) << time_median<<"ms";
    std::cout << " mean ="  << std::setw(7) << time_mean << "ms" << std::endl;
#endif
}