# https://linux.cn/article-9718-1.html
# parallel --jobs 16 < convert-tools/convert.sh
OMP_NUM_THREADS=1 python python/convert.py --format onnx --only-convert efficientformerv2_s0
OMP_NUM_THREADS=1 python python/convert.py --format onnx --only-convert efficientformerv2_s1
OMP_NUM_THREADS=1 python python/convert.py --format onnx --only-convert efficientformerv2_s2
OMP_NUM_THREADS=1 python python/convert.py --format onnx --only-convert SwiftFormer_XS
OMP_NUM_THREADS=1 python python/convert.py --format onnx --only-convert SwiftFormer_S
OMP_NUM_THREADS=1 python python/convert.py --format onnx --only-convert SwiftFormer_L1
OMP_NUM_THREADS=1 python python/convert.py --format onnx --only-convert edgenext_xx_small
OMP_NUM_THREADS=1 python python/convert.py --format onnx --only-convert edgenext_x_small
OMP_NUM_THREADS=1 python python/convert.py --format onnx --only-convert edgenext_small
OMP_NUM_THREADS=1 python python/convert.py --format onnx --only-convert mobilevit_xx_small --opset-version 9
OMP_NUM_THREADS=1 python python/convert.py --format onnx --only-convert mobilevit_x_small --opset-version 9
OMP_NUM_THREADS=1 python python/convert.py --format onnx --only-convert mobilevit_small --opset-version 9
OMP_NUM_THREADS=1 python python/convert.py --format onnx --only-convert LeViT_128 --fuse
OMP_NUM_THREADS=1 python python/convert.py --format onnx --only-convert LeViT_192 --fuse
OMP_NUM_THREADS=1 python python/convert.py --format onnx --only-convert LeViT_256 --fuse
OMP_NUM_THREADS=1 python python/convert.py --format onnx --only-convert resnet50
OMP_NUM_THREADS=1 python python/convert.py --format onnx --only-convert mobilenetv3_large_100
