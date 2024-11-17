@echo off
powercfg -setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMIN 100
powercfg -setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 100

python python/openvino-perf.py --only-test s0 --format int8 --device GPU --threads 1 2>$null
python python/openvino-perf.py --only-test s1 --format int8 --device GPU --threads 1 2>$null
python python/openvino-perf.py --only-test s2 --format int8 --device GPU --threads 1 2>$null

python python/openvino-perf.py --only-test _XS --format int8 --device GPU --threads 1 2>$null
python python/openvino-perf.py --only-test r_S --format int8 --device GPU --threads 1 2>$null
python python/openvino-perf.py --only-test _L1 --format int8 --device GPU --threads 1 2>$null

python python/openvino-perf.py --only-test 1M --format int8 --device GPU --threads 1 2>$null
python python/openvino-perf.py --only-test 2M --format int8 --device GPU --threads 1 2>$null
python python/openvino-perf.py --only-test 6M --format int8 --device GPU --threads 1 2>$null

python python/openvino-perf.py --only-test xt_xx --format int8 --device GPU --threads 1 2>$null
python python/openvino-perf.py --only-test xt_x_ --format int8 --device GPU --threads 1 2>$null
python python/openvino-perf.py --only-test xt_sm --format int8 --device GPU --threads 1 2>$null

python python/openvino-perf.py --only-test 2_05 --format int8 --device GPU --threads 1 2>$null
python python/openvino-perf.py --only-test 2_07 --format int8 --device GPU --threads 1 2>$null
python python/openvino-perf.py --only-test 2_10 --format int8 --device GPU --threads 1 2>$null

python python/openvino-perf.py --only-test it_xx --format int8 --device GPU --threads 1 2>$null
python python/openvino-perf.py --only-test it_x_ --format int8 --device GPU --threads 1 2>$null
python python/openvino-perf.py --only-test it_sm --format int8 --device GPU --threads 1 2>$null

python python/openvino-perf.py --only-test 128 --format int8 --device GPU --threads 1 2>$null
python python/openvino-perf.py --only-test 192 --format int8 --device GPU --threads 1 2>$null

python python/openvino-perf.py --only-test net --format int8 --device GPU --threads 1 2>$null
