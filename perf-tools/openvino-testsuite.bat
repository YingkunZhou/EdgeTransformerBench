@echo off
powercfg -setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMIN 100
powercfg -setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 100

:testsuite_series
SETLOCAL
SET _device=%1
SET _format=%2
SET _threads=%3
SET _sleep=%4
ENDLOCAL
python python/openvino-perf.py --device %_device% --format %_format% --threads %_threads% --sleep %_sleep% --only-test SwiftFormer       2>nul
python python/openvino-perf.py --device %_device% --format %_format% --threads %_threads% --sleep %_sleep% --only-test EMO               2>nul
python python/openvino-perf.py --device %_device% --format %_format% --threads %_threads% --sleep %_sleep% --only-test edgenext          2>nul
python python/openvino-perf.py --device %_device% --format %_format% --threads %_threads% --sleep %_sleep% --only-test mobilevitv2       2>nul
python python/openvino-perf.py --device %_device% --format %_format% --threads %_threads% --sleep %_sleep% --only-test mobilevit_        2>nul
python python/openvino-perf.py --device %_device% --format %_format% --threads %_threads% --sleep %_sleep% --only-test LeViT             2>nul
python python/openvino-perf.py --device %_device% --format %_format% --threads %_threads% --sleep %_sleep% --only-test net               2>nul
EXIT /B 0

SET _sleep=0

CALL :testsuite_series PCORE FP32 1 %_sleep%
CALL :testsuite_series PCORE INT8 1 %_sleep%

CALL :testsuite_series ECORE FP32 1 %_sleep%
CALL :testsuite_series ECORE INT8 1 %_sleep%

CALL :testsuite_series ECORE FP32 4 %_sleep%
CALL :testsuite_series ECORE INT8 4 %_sleep%

CALL :testsuite_series NPU FP16 1 %_sleep%
CALL :testsuite_series NPU INT8 1 %_sleep%

CALL :testsuite_series GPU FP32 1 %_sleep%
CALL :testsuite_series GPU FP16 1 %_sleep%
CALL :testsuite_series GPU INT8 1 %_sleep%