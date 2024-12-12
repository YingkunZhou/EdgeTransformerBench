#python python/cann-perf.py --format fp16 --only-test mobilevit_x_small --profile 2>/dev/null | tee /tmp/fp16
#python python/cann-perf.py --format int8 --only-test mobilevit_x_small --profile 2>/dev/null | tee /tmp/int8
import sys
file = 'fp16'
if len(sys.argv) > 1:
    file = sys.argv[1]

lines = open('/tmp/'+file).readlines()
op_dict = {
    'Conv_GEMM':    0.0,
    'Attn_MatMul':  0.0,
    'Attn_SoftMax': 0.0,
    'MISC_PWOP':    0.0,
}
total = 0
count = 0
begin = False
for l in lines:
    if begin:
        its = l.strip().split()
        latency = float(its[-1])/1000000
        total += latency
        if 'Conv' in l or \
            ('ffn' in l and 'MatMul' in l) or \
            ('mha' in l and 'proj' in l and 'MatMul' in l):
            op_dict['Conv_GEMM'] += latency
        elif 'mha' in l and 'MatMul' in l:
            op_dict['Attn_MatMul'] += latency
            count += 1
        elif 'Softmax' in l:
            op_dict['Attn_SoftMax'] += latency
        else:
            op_dict['MISC_PWOP'] += latency
    elif 'TransData' in l:
        begin = True

print('Attn_MatMul op num = ' + str(count))
print('{0: >15}{1: >20}{2: >10}'.format('Layer', 'Time (ms)', 'Time %'))
for k in op_dict:
    time = "{:.4f}".format(op_dict[k])
    time_percentage = "{:.1f}".format(op_dict[k] / total * 100)
    print('{0: >15}{1: >20}{2: >10}'.format(k, time, time_percentage))
print('{0: >15}{1: >20}{2: >10}'.format('Total', "{:.4f}".format(total), '100.0'))
