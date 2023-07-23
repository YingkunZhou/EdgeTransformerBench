// https://github.com/microsoft/onnxruntime/blob/v1.8.2/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/CXX_Api_Sample.cpp
// https://github.com/microsoft/onnxruntime/blob/v1.8.2/include/onnxruntime/core/session/onnxruntime_cxx_api.h
#include <onnxruntime_cxx_api.h>

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <iomanip>
#include <filesystem>


template <typename T>
T vectorProduct(const std::vector<T>& v)
{
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

/**
 * @brief Operator overloading for printing vectors
 * @tparam T
 * @param os
 * @param v
 * @return std::ostream&
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i)
    {
        os << v[i];
        if (i != v.size() - 1)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

/**
 * @brief Print ONNX tensor data type
 * https://github.com/microsoft/onnxruntime/blob/rel-1.6.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L93
 * @param os
 * @param type
 * @return std::ostream&
 */
std::ostream& operator<<(std::ostream& os,
                         const ONNXTensorElementDataType& type)
{
    switch (type)
    {
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
            os << "undefined";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            os << "float";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            os << "uint8_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            os << "int8_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            os << "uint16_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            os << "int16_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            os << "int32_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            os << "int64_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
            os << "std::string";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            os << "bool";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            os << "float16";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            os << "double";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            os << "uint32_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            os << "uint64_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
            os << "float real + float imaginary";
            break;
        case ONNXTensorElementDataType::
            ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
            os << "double real + float imaginary";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
            os << "bfloat16";
            break;
        default:
            break;
    }

    return os;
}

const float IMAGENET_DEFAULT_MEAN[3] = {0.485f, 0.456f, 0.406f};
const float IMAGENET_DEFAULT_STD[3]  = {0.229f, 0.224f, 0.225f};
const int NUM_CLASSES = 1000;
const int WARMUP_SEC = 5;
const int TEST_SEC = 20;

void pre_process(cv::Mat& img, int input_size) {
    int size = input_size + 32;

    int width  = img.cols;
    int height = img.rows;
    if (width > height) {
        width = size * width / height;
        height = size;
    }
    else {
        height = size * height / width;
        width = size;
    }

    // transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC)
    /* https://github.com/python-pillow/Pillow/issues/2718
       but has difference with python pillow
    */
    cv::resize(img, img, cv::Size(width, height), cv::InterpolationFlags::INTER_CUBIC);

    // std::cout << img.at<cv::Vec3b>(112,112) << std::endl;
    // transforms.CenterCrop(args.input_size)
    // https://gist.github.com/1duo/c868f0bccf0d1f6cd8b36086ba295e04
    int offW = (img.cols - input_size) / 2;
    int offH = (img.rows - input_size) / 2;
    const cv::Rect roi(offW, offH, input_size, input_size);
    img = img(roi);

    // transforms.ToTensor()
    img.convertTo(img, CV_32F, 1.f / 255.f);

    // transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    cv::Mat channels[3];
    cv::split(img, channels);
    /* Normalization per channel
       Normalization parameters obtained from
       https://github.com/onnx/models/tree/master/vision/classification/squeezenet
    */
    channels[0] = (channels[0] - IMAGENET_DEFAULT_MEAN[0]) / IMAGENET_DEFAULT_STD[0];
    channels[1] = (channels[1] - IMAGENET_DEFAULT_MEAN[1]) / IMAGENET_DEFAULT_STD[1];
    channels[2] = (channels[2] - IMAGENET_DEFAULT_MEAN[2]) / IMAGENET_DEFAULT_STD[2];
    cv::merge(channels, 3, img);

    // HWC to CHW
    cv::dnn::blobFromImage(img, img);
}

void load_image(
    std::string img_path,
    std::vector<float> &input_tensor,
    int input_size,
    int batch_size)
{
    // read img and pre-process
    cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    pre_process(img, input_size);
    // Make copies of the same image input.
    for (int64_t i = 0; i < batch_size; ++i)
    {
        std::copy(img.begin<float>(), img.end<float>(),
        input_tensor.begin() + i * 3 * input_size * input_size);
    }
}

void scores_topk(
    std::vector<std::pair<float, int>> &vec,
    const std::vector<float> &scores, const int topk)
{
    vec.resize(NUM_CLASSES);
    for (int i = 0; i < NUM_CLASSES; i++) {
        vec[i] = std::make_pair(scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater<std::pair<float, int>>());
}

bool acck(
    const std::vector<float> &scores,
    const int topk,
    const int index,
    bool &acc1)
{
    std::vector<std::pair<float, int>> vec;
    scores_topk(vec, scores, topk);
    if (index == vec[0].second) {
        acc1 = true;
        return true;
    }
    for (int i = 1; i < topk; i++) {
        if (index == vec[i].second) {
            return true;
        }
    }
    return false;
}

void print_topk(const std::vector<float> &scores, const int topk) {
    std::vector<std::pair<float, int>> vec;
    scores_topk(vec, scores, topk);
    // print topk and score
    for (int i = 0; i < topk; i++) {
        printf("(index: %d,  score: %f), ", vec[i].second, vec[i].first);
    }
    printf("\n");
}

std::vector<std::filesystem::path>
traverse_class(std::string imagenet_path) {
    std::vector<std::filesystem::path> classes;
    std::copy(std::filesystem::directory_iterator(imagenet_path),
    std::filesystem::directory_iterator(), std::back_inserter(classes));
    std::sort(classes.begin(), classes.end());
    return classes;
}

struct {
  std::string model;
  bool validation;
  int input_size;
  bool usi_eval;
  int batch_size;
  std::string data_path;
  std::vector<const char*> input_name;
  std::vector<const char*> output_name;
  std::vector<Ort::Value> input;
  std::vector<Ort::Value> output;
} args;

void evaluate(
    Ort::Session &session,
    std::vector<float> &input_tensor,
    std::vector<float> &output_tensor)
{
    int class_index = 0;
    int num_predict = 0;
    int num_acc1 = 0;
    int num_acc5 = 0;
    std::cout << std::fixed << std::setprecision(4);

    std::vector<std::filesystem::path> classes = traverse_class(args.data_path);
    for (const std::string& class_path : classes) {
        for (const auto & image: std::filesystem::directory_iterator(class_path)) {
            load_image(image.path(), input_tensor, args.input_size, args.batch_size);
            session.Run(Ort::RunOptions{nullptr},
                        args.input_name.data(), args.input.data() , 1 /*Number of inputs*/,
                        args.output_name.data(),args.output.data(), 1 /*Number of outputs*/
                        );
            num_predict++;
            bool acc1 = false;
            num_acc5 += acck(output_tensor, 5, class_index*50+15, acc1);
            num_acc1 += acc1;
        }
        class_index++;
        std::cout << "Done [" << class_index << "/" << classes.size() << "]";
        std::cout << "\tacc1: " << num_acc1*1.f/num_predict;
        std::cout << "\tacc5: " << num_acc5*1.f/num_predict << std::endl;
    }
}

void benchmark(
    Ort::Session &session,
    std::vector<float> &input_tensor,
    std::vector<float> &output_tensor)
{
    // Measure latency
    load_image("daisy.jpg", input_tensor, args.input_size, args.batch_size);
    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &end);
    clock_gettime(CLOCK_REALTIME, &start);

    while (end.tv_sec - start.tv_sec < WARMUP_SEC) {
        session.Run(Ort::RunOptions{nullptr},
                    args.input_name.data(), args.input.data() , 1 /*Number of inputs*/,
                    args.output_name.data(),args.output.data(), 1 /*Number of outputs*/
                    );
        clock_gettime(CLOCK_REALTIME, &end);
    }

    print_topk(output_tensor, 3);

    std::vector<double> time_list = {};
    while (std::reduce(time_list.begin(), time_list.end()) < TEST_SEC) {
        clock_gettime(CLOCK_REALTIME, &start);
        session.Run(Ort::RunOptions{nullptr},
                    args.input_name.data(), args.input.data() , 1 /*Number of inputs*/,
                    args.output_name.data(),args.output.data(), 1 /*Number of outputs*/
                    );
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
    std::cout << "min =\t" << time_min << "ms\tmax =\t" << time_min << "ms\tmean =\t" << time_mean << "ms\tmedian =\t" << time_median << "ms" << std::endl;
}

int main(int argc, char* argv[])
{
    args.data_path = "imagenet-div50";
    args.validation = true;

    std::string model_file = argv[1];
    args.model = model_file;
    args.input_size = 224;

    int num_threads = argc > 4? atoi(argv[4]) : 1;
    int inter_threads = argc > 5? atoi(argv[6]) : 1;
    args.batch_size = 1;

    std::string instanceName{"image-classification-inference"};
    std::string modelFilepath{model_file};

    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                 instanceName.c_str());
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(num_threads);

    // Sets graph optimization level
    // Available levels are
    // ORT_DISABLE_ALL -> To disable all optimizations
    // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node
    // removals) ORT_ENABLE_EXTENDED -> To enable extended optimizations
    // (Includes level 1 + more complex optimizations like node fusions)
    // ORT_ENABLE_ALL -> To Enable All possible optimizations
    sessionOptions.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);

    sessionOptions.SetExecutionMode(
        ExecutionMode::ORT_PARALLEL);

    sessionOptions.SetInterOpNumThreads(inter_threads);

    Ort::AllocatorWithDefaultOptions allocator;

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu( //TODO: cpu?
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    // create a session
    Ort::Session session(env, modelFilepath.c_str(), sessionOptions);

    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
    if (inputDims.at(0) == -1)
    {
        std::cout << "Got dynamic batch size. Setting input batch size to "
                  << args.batch_size << "." << std::endl;
        inputDims.at(0) = args.batch_size;
    }

    Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> outputDims = outputTensorInfo.GetShape();
    if (outputDims.at(0) == -1)
    {
        std::cout << "Got dynamic batch size. Setting output batch size to "
                  << args.batch_size << "." << std::endl;
        outputDims.at(0) = args.batch_size;
    }

    size_t inputTensorSize  = vectorProduct(inputDims);
    size_t outputTensorSize = vectorProduct(outputDims);

    std::vector<float> input_tensor(inputTensorSize);
    std::vector<Ort::Value> inputTensors;
    args.input.push_back(Ort::Value::CreateTensor<float>(memoryInfo,
        input_tensor.data(), inputTensorSize,
        inputDims.data(), inputDims.size()));

    std::vector<float> output_tensor(outputTensorSize);
    std::vector<Ort::Value> outputTensors;
    args.output.push_back(Ort::Value::CreateTensor<float>(memoryInfo,
        output_tensor.data(), outputTensorSize,
        outputDims.data(), outputDims.size()));

    const auto inputName = session.GetInputNameAllocated(0, allocator);
    const auto outputName = session.GetOutputNameAllocated(0, allocator);

    args.input_name.push_back(&*inputName);
    args.output_name.push_back(&*outputName);

#ifdef DEBUG
    //print basic model I/O info
    std::cout << "Number of Input Nodes: " << session.GetInputCount() << std::endl;
    std::cout << "Number of Output Nodes: " << session.GetOutputCount() << std::endl;

    std::cout << "Input Name: " << args.input_name << std::endl;
    std::cout << "Input Type: " << inputTensorInfo.GetElementType() << std::endl;
    std::cout << "Input Dimensions: " << inputDims << std::endl;

    std::cout << "Output Name: " << args.output_name << std::endl;
    std::cout << "Output Type: " << outputTensorInfo.GetElementType() << std::endl;
    std::cout << "Output Dimensions: " << outputDims << std::endl;
#endif

    if (args.validation) {
        evaluate(session, input_tensor, output_tensor);
    }
    else {
        benchmark(session, input_tensor, output_tensor);
    }
}
