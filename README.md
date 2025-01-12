# EdgeTransformerPerf (ETBench)

Please check [tutorials.md](https://github.com/YingkunZhou/EdgeTransformerBench/blob/main/tutorials.md) to figure out how to use ETBench for latest models assessment on Linux, Android, Macos and Windows across various CPUs/GPUs/NPUS.

Edge/mobile CNN+Transformer hybrid DNN backbone inference benchmark (currently only for computer vision task)

we filter out the model which satisfy one of the condition below:
- #params > 15M (for FP32 model > 60M) which is too large for edge/mobile devices.
- GMACs > 2G for low computational power edge devices, it is beyond their capacity to sustain, especially in scenarios that require real-time processing.

| Model | Top-1 |  [Top-1 <br />//20 est.](https://github.com/YingkunZhou/EdgeTransformerBench/releases/download/v0.0/imagenet-div20.tar) | [Top-1 <br />//50 est.](https://github.com/YingkunZhou/EdgeTransformerBench/releases/download/v0.0/imagenet-div50.tar) | #params | GMACs | wight
|:---------------|:----:|:---:|:--:|:--:|:--:|:--:|
| [efficientformerv2_s0](https://arxiv.org/abs/2212.08059) |   76.2   |  76.3  | 76.0 |  3.5M    |   0.40G   | [eformer_s0_450.pth](https://drive.google.com/file/d/1PXb7b9pv9ZB4cfkRkYEdwgWuVwvEiazq/view?usp=share_link) |
| efficientformerv2_s1 |   79.7   |  78.8  | 79.6 |  6.1M    |   0.65G   | [eformer_s1_450.pth](https://drive.google.com/file/d/1EKe1vt-3mG7iceVIMaET_DyISzVTJMn8/view?usp=share_link) |
| efficientformerv2_s2 |   82.0   |  82.0  | 81.9 | 12.6M    |   1.25G   | [eformer_s2_450.pth](https://drive.google.com/file/d/1gjbFyB5T_yAkmzHNuXEljqScYVQZafMQ/view?usp=share_link) |
||
| [SwiftFormer_XS](https://arxiv.org/abs/2303.15446) |   75.7   |  76.1  | 75.3 | 3.5M   |   0.4G   | [SwiftFormer_XS_ckpt.pth](https://drive.google.com/file/d/15Ils-U96pQePXQXx2MpmaI-yAceFAr2x/view?usp=sharing) |
| SwiftFormer_S  |   78.5   |  78.3  | 78.3 | 6.1M   |   1.0G   | [SwiftFormer_S_ckpt.pth](https://drive.google.com/file/d/1_0eWwgsejtS0bWGBQS3gwAtYjXdPRGlu/view?usp=sharing) |
| SwiftFormer_L1 |   80.9   |  80.7  | 81.8 |12.1M   |   1.6G   | [SwiftFormer_L1_ckpt.pth](https://drive.google.com/file/d/1jlwrwWQ0SQzDRc5adtWIwIut5d1g9EsM/view?usp=sharing) |
||
| [EMO_1M](https://arxiv.org/abs/2301.01146)  |   71.5   |  70.7  | 68.3 | 1.3M   |   0.26G   | [EMO_1M.pth](https://github.com/zhangzjn/EMO/blob/main/resources/EMO_1M/net.pth) |
| EMO_2M  |   75.1   |  74.8  | 73.6 | 2.3M   |   0.44G   | [EMO_2M.pth](https://github.com/zhangzjn/EMO/blob/main/resources/EMO_2M/net.pth) |
| EMO_5M  |   78.4   |  78.2  | 77.6 | 5.1M   |   0.90G   | [EMO_5M.pth](https://github.com/zhangzjn/EMO/blob/main/resources/EMO_5M/net.pth) |
| EMO_6M  |   79.0   |  79.2  | 77.9 | 6.1M   |   0.96G   | [EMO_6M.pth](https://github.com/zhangzjn/EMO/blob/main/resources/EMO_6M/net.pth) |
||
| [edgenext_xx_small](https://arxiv.org/abs/2206.10589)  |   71.2   |  70.8  | 70.4 | 1.3M   |   0.26G   | [edgenext_xx_small.pth](https://github.com/mmaaz60/EdgeNeXt/releases/download/v1.0/edgenext_xx_small.pth) |
| edgenext_x_small   |   74.9   |  74.9  | 74.9 | 2.3M   |   0.54G   | [edgenext_x_small.pth](https://github.com/mmaaz60/EdgeNeXt/releases/download/v1.0/edgenext_x_small.pth) |
| edgenext_small/usi |   81.1   |  80.8  | 80.0 | 5.6M   |   1.26G   | [edgenext_small_usi.pth](https://github.com/mmaaz60/EdgeNeXt/releases/download/v1.1/edgenext_small_usi.pth) |
||
| [mobilevitv2_050](https://arxiv.org/abs/2206.02680)  |   70.2   |  69.9  | 66.7 | 1.4M   |   0.5G   | [mobilevitv2-0.5.pt](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet1k/256x256/mobilevitv2-0.5.pt) |
| mobilevitv2_075  |   75.6   |  75.0  | 74.4 | 2.9M   |   1.0G   | [mobilevitv2-0.75.pt](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet1k/256x256/mobilevitv2-0.75.pt) |
| mobilevitv2_100  |   78.1   |  77.9  | 76.9 | 4.9M   |   1.8G   | [mobilevitv2-1.0.pt](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet1k/256x256/mobilevitv2-1.0.pt) |
| [x] mobilevitv2_125  |   79.7   |  79.1  | 80.7 | 7.5M   |   2.8G   | [mobilevitv2-1.25.pt](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet1k/256x256/mobilevitv2-1.25.pt) |
| [x] mobilevitv2_150  |   81.5   |  80.8  | 81.8 |10.6M   |   4.0G   | [mobilevitv2-1.5.pt](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet21k_to_1k/256x256/mobilevitv2-1.5.pt) |
| [x] mobilevitv2_175  |   81.9   |  80.8  | 81.1 |14.3M   |   5.5G   | [mobilevitv2-1.75.pt](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet21k_to_1k/256x256/mobilevitv2-1.75.pt) |
| [x] mobilevitv2_200  |   82.3   |  82.0  | 83.1 |18.4M   |   7.2G   | [mobilevitv2-2.0.pt](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet21k_to_1k/256x256/mobilevitv2-2.0.pt) |
||
| [mobilevit_xx_small](https://arxiv.org/abs/2110.02178)  |   68.9   |  68.9  | 66.6 | 1.3M   |   0.36G   | [mobilevit_xxs.pt](https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_xxs.pt) |
| mobilevit_x_small   |   74.7   |  74.3  | 73.9 | 2.3M   |   0.89G   | [mobilevit_xs.pt](https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_xs.pt) |
| mobilevit_small     |   78.2   |  77.7  | 78.1 | 5.6M   |   2.0 G   | [mobilevit_s.pt](https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_s.pt) |
||
| [LeViT_128S](https://arxiv.org/abs/2104.01136)     |   76.5   |  75.9  | 76.2 | 7.8M   |   0.30G   | [LeViT-128S.pth](https://dl.fbaipublicfiles.com/LeViT/LeViT-128S-96703c44.pth) |
| LeViT_128      |   78.6   |  79.3  | 78.2 | 9.2M   |   0.41G   | [LeViT-128.pth](https://dl.fbaipublicfiles.com/LeViT/LeViT-128-b88c2750.pth) |
| LeViT_192      |   79.9   |  79.8  | 79.3 | 11 M   |   0.66G   | [LeViT-192.pth](https://dl.fbaipublicfiles.com/LeViT/LeViT-192-92712e41.pth) |
| [x] LeViT_256      |   81.6   |  81.2  | 81.4 | 19 M   |   1.12G   | [LeViT-256.pth](https://dl.fbaipublicfiles.com/LeViT/LeViT-256-13b5763e.pth) |

## Traditional CNN

| Model | Top-1 |  Top-1 <br />//20 est. | Top-1 <br />//50 est. | #params | [GMACs](https://github.com/da2so/efficientnetv2) | wight
|:---------------|:----:|:---:|:--:|:--:|:--:|:--:|
|resnet50 | 80.4 | 80.3 | 81.1 | 25.6M | 4.1G |
||
|mobilenetv3_large_100 | 75.8 | 75.7 | 75.3 |  5.5M | 0.29G |
|tf_efficientnetv2_b0  | 78.4 | 78.1 | 76.7 |  7.1M | 0.72G |
|tf_efficientnetv2_b1  | 79.5 | 79.3 | 79.4 |  8.1M | 1.2G |
|tf_efficientnetv2_b2  | 80.2 | 81.7 | 80.4 | 10.1M | 1.7G |
|tf_efficientnetv2_b3  | 81.6 | 81.9 | 82.0 | 14.4M | 3.0G |
