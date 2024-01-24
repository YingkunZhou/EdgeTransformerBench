# https://linux.cn/article-9718-1.html
# parallel --jobs 16 < convert-tools/convert.sh
# Now, onnx2paddle support convert onnx model opset_verison [7, 8, 9, 10, 11, 12, 13, 14, 15]
OMP_NUM_THREADS=1 python python/convert.py --format onnx --only-convert efficientformerv2_s0 --opset-version 14
OMP_NUM_THREADS=1 python python/convert.py --format onnx --only-convert efficientformerv2_s1 --opset-version 14
OMP_NUM_THREADS=1 python python/convert.py --format onnx --only-convert efficientformerv2_s2 --opset-version 14
OMP_NUM_THREADS=1 python python/convert.py --format onnx --only-convert SwiftFormer_XS --opset-version 14
OMP_NUM_THREADS=1 python python/convert.py --format onnx --only-convert SwiftFormer_S  --opset-version 14
OMP_NUM_THREADS=1 python python/convert.py --format onnx --only-convert SwiftFormer_L1 --opset-version 14
OMP_NUM_THREADS=1 python python/convert.py --format onnx --only-convert EMO_1M --opset-version 14
OMP_NUM_THREADS=1 python python/convert.py --format onnx --only-convert EMO_2M --opset-version 14
OMP_NUM_THREADS=1 python python/convert.py --format onnx --only-convert EMO_5M --opset-version 14
OMP_NUM_THREADS=1 python python/convert.py --format onnx --only-convert EMO_6M --opset-version 14
OMP_NUM_THREADS=1 python python/convert.py --format onnx --only-convert edgenext_xx_small --opset-version 14
OMP_NUM_THREADS=1 python python/convert.py --format onnx --only-convert edgenext_x_small --opset-version 14
OMP_NUM_THREADS=1 python python/convert.py --format onnx --only-convert edgenext_small --opset-version 14
OMP_NUM_THREADS=1 python python/convert.py --format onnx --only-convert mobilevitv2_050 --opset-version 14
OMP_NUM_THREADS=1 python python/convert.py --format onnx --only-convert mobilevitv2_075 --opset-version 14
OMP_NUM_THREADS=1 python python/convert.py --format onnx --only-convert mobilevitv2_100 --opset-version 14
OMP_NUM_THREADS=1 python python/convert.py --format onnx --only-convert mobilevitv2_125 --opset-version 14
OMP_NUM_THREADS=1 python python/convert.py --format onnx --only-convert mobilevitv2_150 --opset-version 14
OMP_NUM_THREADS=1 python python/convert.py --format onnx --only-convert mobilevitv2_175 --opset-version 14
OMP_NUM_THREADS=1 python python/convert.py --format onnx --only-convert mobilevitv2_200 --opset-version 14
OMP_NUM_THREADS=1 python python/convert.py --format onnx --only-convert mobilevit_xx_small --opset-version 9
OMP_NUM_THREADS=1 python python/convert.py --format onnx --only-convert mobilevit_x_small --opset-version 9
OMP_NUM_THREADS=1 python python/convert.py --format onnx --only-convert mobilevit_small --opset-version 9
OMP_NUM_THREADS=1 python python/convert.py --format onnx --only-convert LeViT_128 --fuse --opset-version 14
OMP_NUM_THREADS=1 python python/convert.py --format onnx --only-convert LeViT_192 --fuse --opset-version 14
OMP_NUM_THREADS=1 python python/convert.py --format onnx --only-convert LeViT_256 --fuse --opset-version 14
OMP_NUM_THREADS=1 python python/convert.py --format onnx --only-convert resnet50 --opset-version 14
OMP_NUM_THREADS=1 python python/convert.py --format onnx --only-convert mobilenetv3_large_100 --opset-version 14