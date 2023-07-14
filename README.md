# EdgeTransformerPerf
edge/mobile transformer based Vision DNN inference benchmark

| Model | Top-1 |  Top-1 quick est. | #params | GMACs |
|:---------------|:----:|:---:|:--:|:--:|
| efficientformerv2_s0 |   76.2   |  76.3  |  3.5M    |   0.40G   |
| efficientformerv2_s1 |   79.7   |  78.8  |  6.1M    |   0.65G   |
| efficientformerv2_s2 |   82.0   |  82.0  | 12.6M    |   1.25G   |
|
| SwiftFormer-XS |   75.7   |  76.1  |  3.5M   |   0.4G   |
| SwiftFormer-S  |   78.5   |  78.3  |  6.1M   |   1.0G   |
| SwiftFormer-L1 |   80.9   |  80.7  | 12.1M   |   1.6G   |
|
| edgenext_xx_small  |   71.2   |  70.8  | 1.33M   |   0.26G   |
| edgenext_x_small   |   74.9   |  74.9  | 2.34M   |   0.54G   |
| edgenext_small/usi |   81.1   |  80.8  | 5.59M   |   1.26G   |
|
| EMO_1M  |   71.5   |  70.7  | 1.3M   |   0.26G   |
| EMO_2M  |   75.1   |  74.8  | 2.3M   |   0.44G   |
| EMO_5M  |   78.4   |  78.2  | 5.1M   |   0.90G   |
| EMO_6M  |   79.0   |  79.2  | 6.1M   |   0.96G   |
|
| mobilevitv2_050  |   70.2   |  69.9  |  1.4M   |   0.5G   |
| mobilevitv2_075  |   75.6   |  75.0  |  2.9M   |   1.0G   |
| mobilevitv2_100  |   78.1   |  77.9  |  4.9M   |   1.8G   |
| mobilevitv2_125  |   79.7   |  79.1  |  7.5M   |   2.8G   |
| mobilevitv2_150  |   81.5   |  80.8  | 10.6M   |   4.0G   |
| mobilevitv2_175  |   81.9   |  80.8  | 14.3M   |   5.5G   |
| mobilevitv2_200  |   82.3   |  82.0  | 18.4M   |   7.2G   |
|
| mobilevit_xx_small  |   68.9   |  68.9  | 1.3M   |   0.36G   |
| mobilevit_x_small   |   74.7   |  74.3  | 2.3M   |   0.89G   |
| mobilevit_small     |   78.2   |  77.7  | 5.6M   |   2.0 G   |
|
| LeViT-128S     |   76.5   |  75.9  | 7.8M   |   0.30G   |
| LeViT-128      |   78.6   |  79.3  | 9.2M   |   0.41G   |
| LeViT-192      |   79.9   |  79.8  | 11 M   |   0.66G   |
| LeViT-256      |   81.6   |  81.2  | 19 M   |   1.12G   |

