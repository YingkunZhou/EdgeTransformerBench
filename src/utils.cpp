
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "PillowResize.hpp"

#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <iomanip>
#include <filesystem>
#include "utils.h"


std::vector<std::pair<std::string, int>> test_models = {
#if defined(DEBUG)
    {"debug", 1},
#endif
    {"efficientformerv2_s0", 224},
    {"efficientformerv2_s1", 224},
    {"efficientformerv2_s2", 224},

    {"SwiftFormer_XS", 224},
    {"SwiftFormer_S", 224},
    {"SwiftFormer_L1", 224},

    {"EMO_1M", 224},
    {"EMO_2M", 224},
    {"EMO_5M", 224},
    {"EMO_6M", 224},

    {"edgenext_xx_small", 256},
    {"edgenext_x_small", 256},
    {"edgenext_small", 256},

    {"mobilevitv2_050", 256},
    {"mobilevitv2_075", 256},
    {"mobilevitv2_100", 256},
    {"mobilevitv2_125", 256},
    {"mobilevitv2_150", 256},
    {"mobilevitv2_175", 256},
    {"mobilevitv2_200", 256},

    {"mobilevit_xx_small", 256},
    {"mobilevit_x_small", 256},
    {"mobilevit_small", 256},

    {"LeViT_128S", 224},
    {"LeViT_128" , 224},
    {"LeViT_192" , 224},
    {"LeViT_256" , 224},

    {"resnet50", 224},
    {"mobilenetv3_large_100", 224},
    {"tf_efficientnetv2_b0", 224},
    {"tf_efficientnetv2_b1", 240},
    {"tf_efficientnetv2_b2", 260},
    {"tf_efficientnetv2_b3", 300},
};

void pre_process(cv::Mat& img, std::string model, int input_size) {
    bool is_resnet50  = model.find("resnet50") != std::string::npos;
    bool is_edgenext  = model.find("edgenext") != std::string::npos;
    bool is_edgenext_small = model.find("edgenext_small") != std::string::npos;
    bool is_mobilevit = model.find("mobilevit") != std::string::npos;
    bool is_efficientnetv2_b3 = model.find("efficientnetv2_b3") != std::string::npos;

    int size;
    double crop_pct;

    if (is_resnet50 ||  is_edgenext_small) { // for EdgeNeXt
        crop_pct = 0.95;
        size = int(input_size / crop_pct);
    }
    else if (is_edgenext) {
        crop_pct = 224.f / 256.f;
        size = int(input_size / crop_pct);
    }
    else {
        size = input_size + 32;
    }

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
       https://github.com/zurutech/pillow-resize
    */
    // cv::resize(img, img, cv::Size(width, height), cv::InterpolationFlags::INTER_CUBIC);
    img = PillowResize::resize(img, cv::Size(width, height), PillowResize::INTERPOLATION_BICUBIC);

    // std::cout << img.at<cv::Vec3b>(112,112) << std::endl;
    // transforms.CenterCrop(args.input_size)
    // https://gist.github.com/1duo/c868f0bccf0d1f6cd8b36086ba295e04
    int offW = (img.cols - input_size) / 2;
    int offH = (img.rows - input_size) / 2;
    const cv::Rect roi(offW, offH, input_size, input_size);
    img = img(roi);

    // transforms.ToTensor()
    img.convertTo(img, CV_32F, 1.f / 255.f);

    if (is_mobilevit) {
        cv::dnn::blobFromImage(img, img);
        return;
    }
    // transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    cv::Mat channels[3];
    cv::split(img, channels);
    /* Normalization per channel
       Normalization parameters obtained from
       https://github.com/onnx/models/tree/master/vision/classification/squeezenet
    */
    if (is_efficientnetv2_b3) {
        channels[0] = (channels[0] - 0.5) / 0.5;
        channels[1] = (channels[1] - 0.5) / 0.5;
        channels[2] = (channels[2] - 0.5) / 0.5;
    }
    else {
        channels[0] = (channels[0] - IMAGENET_DEFAULT_MEAN[0]) / IMAGENET_DEFAULT_STD[0];
        channels[1] = (channels[1] - IMAGENET_DEFAULT_MEAN[1]) / IMAGENET_DEFAULT_STD[1];
        channels[2] = (channels[2] - IMAGENET_DEFAULT_MEAN[2]) / IMAGENET_DEFAULT_STD[2];
    }
    cv::merge(channels, 3, img);

    // HWC to CHW
    cv::dnn::blobFromImage(img, img);
}

void load_image(
    std::string img_path,
    float *input_tensor,
    std::string model,
    int input_size,
    int batch_size)
{
    // read img and pre-process
    cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
    if (model.find("mobilevit_") == std::string::npos) {
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    }
    pre_process(img, model, input_size);
    // Make copies of the same image input.
    for (int64_t i = 0; i < batch_size; ++i)
    {
        std::copy(img.begin<float>(), img.end<float>(),
        input_tensor + i * 3 * input_size * input_size);
    }
}

void scores_topk(
    std::vector<std::pair<float, int>> &vec,
    const float *scores, const int topk)
{
    vec.resize(NUM_CLASSES);
    for (int i = 0; i < NUM_CLASSES; i++) {
        vec[i] = std::make_pair(scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater<std::pair<float, int>>());
}

bool acck(
    const float *scores,
    const int topk,
    const int index,
    int &num_acc1)
{
    std::vector<std::pair<float, int>> vec;
    scores_topk(vec, scores, topk);
    if (index == vec[0].second) {
        num_acc1 += 1;
        return true;
    }
    for (int i = 1; i < topk; i++) {
        if (index == vec[i].second) {
            return true;
        }
    }
    return false;
}

void print_topk(const float *scores, const int topk) {
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