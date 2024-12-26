#python python/tvm-local-perf.py --only-test s0 2>/dev/null | tee /tmp/tmp.log
import sys
if len(sys.argv) > 1:
    file = sys.argv[1]
else:
    file = '/tmp/tmp.log'

lines = open(file).readlines()
op_dict = {
    'Conv_GEMM':    0.0,
    'Attn_MatMul':  0.0,
    'Attn_SoftMax': 0.0,
    'Reshape':      0.0,
    'MISC_PWOP':    0.0,
}
total = 0
count = 0
begin = False
stop = '--------- '
for l in lines:
    if begin:
        its = l.strip().split()
        if len(its) < 2: continue
        latency = float(its[2].replace(",", ""))
        if 'nn_conv2d' in l or 'nn_dense' in l:
            op_dict['Conv_GEMM'] += latency
        elif 'batch_matmul' in l:
            op_dict['Attn_MatMul'] += latency
            count += 1
        elif 'nn_softmax' in l:
            op_dict['Attn_SoftMax'] += latency
        elif 'reshape_squeeze' in l:
            op_dict['Reshape'] += latency
        elif 'Total_time' in l:
            # print(latency)
            total = latency
            begin = False
        else:
            op_dict['MISC_PWOP'] += latency
    elif stop in l[:len(stop)]:
        begin = True

print('Attn_MatMul op num = ' + str(count))
print('{0: >15}{1: >20}{2: >10}'.format('Layer', 'Time (ms)', 'Time %'))
for k in op_dict:
    time = "{:.4f}".format(op_dict[k])
    time_percentage = "{:.1f}".format(op_dict[k] / total * 100)
    print('{0: >15}{1: >20}{2: >10}'.format(k, time, time_percentage))
print('{0: >15}{1: >20}{2: >10}'.format('Total', "{:.4f}".format(total), '100.0'))
