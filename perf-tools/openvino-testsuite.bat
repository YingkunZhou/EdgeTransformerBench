@echo off
powercfg -setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMIN 100
powercfg -setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 100

SET _sleep=0
echo >>>>>>>>>>>PCORE1: fp32<<<<<<<<<
CALL :testsuite_series PCORE fp32 1 %_sleep%
echo >>>>>>>>>>>PCORE1: int8<<<<<<<<<
CALL :testsuite_series PCORE int8 1 %_sleep%

echo >>>>>>>>>>>ECORE1: fp32<<<<<<<<<
CALL :testsuite_series ECORE fp32 1 %_sleep%
echo >>>>>>>>>>>ECORE1: int8<<<<<<<<<
CALL :testsuite_series ECORE int8 1 %_sleep%

echo >>>>>>>>>>>ECORE4: fp32<<<<<<<<<
CALL :testsuite_series ECORE fp32 4 %_sleep%
echo >>>>>>>>>>>ECORE4: int8<<<<<<<<<
CALL :testsuite_series ECORE int8 4 %_sleep%

echo >>>>>>>>>>>NPU: fp16<<<<<<<<<
CALL :testsuite_series NPU fp16 1 %_sleep%
echo >>>>>>>>>>>NPU: int8<<<<<<<<<
CALL :testsuite_series NPU int8 1 %_sleep%

echo >>>>>>>>>>>GPU: fp32<<<<<<<<<
CALL :testsuite_series GPU fp32 1 %_sleep%
echo >>>>>>>>>>>GPU: fp16<<<<<<<<<
CALL :testsuite_series GPU fp16 1 %_sleep%
echo >>>>>>>>>>>GPU: int8<<<<<<<<<
CALL :testsuite_series GPU int8 1 %_sleep%

:testsuite_series
python python/openvino-perf.py --device %1 --format %2 --threads %3 --sleep %4 --only-test SwiftFormer       2>nul
python python/openvino-perf.py --device %1 --format %2 --threads %3 --sleep %4 --only-test EMO               2>nul
python python/openvino-perf.py --device %1 --format %2 --threads %3 --sleep %4 --only-test edgenext          2>nul
python python/openvino-perf.py --device %1 --format %2 --threads %3 --sleep %4 --only-test mobilevitv2       2>nul
python python/openvino-perf.py --device %1 --format %2 --threads %3 --sleep %4 --only-test mobilevit_        2>nul
python python/openvino-perf.py --device %1 --format %2 --threads %3 --sleep %4 --only-test LeViT             2>nul
python python/openvino-perf.py --device %1 --format %2 --threads %3 --sleep %4 --only-test net               2>nul
exit /b 0