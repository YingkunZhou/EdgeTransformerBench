#ifndef UTILS_H
#define UTILS_H

const float IMAGENET_DEFAULT_MEAN[3] = {0.485f, 0.456f, 0.406f};
const float IMAGENET_DEFAULT_STD[3]  = {0.229f, 0.224f, 0.225f};
const int NUM_CLASSES = 1000;

template <typename T>
T vectorProduct(const std::vector<T>& v)
{
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

void load_image(
    std::string img_path,
    float *input_tensor,
    std::string model,
    int input_size,
    int batch_size
);

void scores_topk(
    std::vector<std::pair<float, int>> &vec,
    const float *scores, const int topk);

bool acck(
    const float *scores,
    const int topk,
    const int index,
    int &acc1
);

void print_topk(const float *scores, const int topk);

std::vector<std::filesystem::path>
traverse_class(std::string imagenet_path);

extern std::vector<std::pair<std::string, int>> test_models;

#endif // UTILS_H
