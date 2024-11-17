@echo off
powercfg -setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMIN 100
powercfg -setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 100

python python/openvino-perf.py --format int8 --device NPU --threads 1 --only-test efficientformerv2 2>nul
python python/openvino-perf.py --format int8 --device NPU --threads 1 --only-test SwiftFormer       2>nul
python python/openvino-perf.py --format int8 --device NPU --threads 1 --only-test EMO               2>nul
python python/openvino-perf.py --format int8 --device NPU --threads 1 --only-test edgenext          2>nul
python python/openvino-perf.py --format int8 --device NPU --threads 1 --only-test mobilevitv2       2>nul
python python/openvino-perf.py --format int8 --device NPU --threads 1 --only-test mobilevit         2>nul
python python/openvino-perf.py --format int8 --device NPU --threads 1 --only-test LeViT             2>nul
python python/openvino-perf.py --format int8 --device NPU --threads 1 --only-test net               2>nul