#/usr/src/tensorrt/bin/trtexec --batch=1 --loadEngine=.onnx/gpu-fp16/mobilevit_x_small.engine --dumpProfile --separateProfileRun | tee /tmp/fp16
#/usr/src/tensorrt/bin/trtexec --batch=1 --loadEngine=.onnx/gpu-int8/mobilevit_x_small.engine --dumpProfile --separateProfileRun | tee /tmp/int8
import sys
file = 'fp16'
if len(sys.argv) > 1:
    file = sys.argv[1]

lines = open('/tmp/'+file).readlines()
op_dict = {
    'Conv_GEMM':    [0.0, 0.0],
    'Attn_MatMul':  [0.0, 0.0],
    'Attn_SoftMax': [0.0, 0.0],
    'MISC_PWOP':    [0.0, 0.0],
}
total = [0, 0]
count = 0
begin = False
for l in lines:
    if 'Layer   Time (ms)   Avg. Time (ms)   Median Time (ms)   Time %' in l:
        begin = True
    elif begin:
        its = l.strip().split()
        if 'Conv' in l or ('ffn' in l and 'MatMul' in l) or \
            ('mha' in l and 'proj' in l and 'MatMul' in l):
            op_dict['Conv_GEMM'][0] += float(its[-3])
            op_dict['Conv_GEMM'][1] += float(its[-2])
        elif 'mha' in l and 'MatMul' in l:
            op_dict['Attn_MatMul'][0] += float(its[-3])
            op_dict['Attn_MatMul'][1] += float(its[-2])
            count += 1
        elif 'Softmax' in l:
            op_dict['Attn_SoftMax'][0] += float(its[-3])
            op_dict['Attn_SoftMax'][1] += float(its[-2])
        elif 'Total' in l:
            begin = False
            total[0] = float(its[-3])
            total[1] = float(its[-2])
        else:
            op_dict['MISC_PWOP'][0] += float(its[-3])
            op_dict['MISC_PWOP'][1] += float(its[-2])

print('Attn_MatMul op num = ' + str(count))
print('{0: >15}{1: >20}{2: >20}{3: >10}'.format('Layer', 'Avg. Time (ms)', 'Median Time (ms)', 'Time %'))
for k in op_dict:
    avg_time = "{:.4f}".format(op_dict[k][0])
    median_time = "{:.4f}".format(op_dict[k][1])
    time_percentage = "{:.1f}".format(op_dict[k][1] / total[1] * 100)
    print('{0: >15}{1: >20}{2: >20}{3: >10}'.format(k, avg_time, median_time, time_percentage))
print('{0: >15}{1: >20}{2: >20}{3: >10}'.format('Total', "{:.4f}".format(total[0]), "{:.4f}".format(total[1]), '100.0'))
