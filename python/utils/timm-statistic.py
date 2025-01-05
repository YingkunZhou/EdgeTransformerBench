refer = open('results-imagenet.csv', 'r').readlines()
def check_size(name):
    size = name.split('_')[-1]
    for l in refer:
        if name in l:
            return l.split(',')[1]
    if size == '224' or size == '256':
        return size
    print(name)
    return '224'

vit_keword = [
    'eca_',
    'bat_resnext',
]

cnn_keyword = [
    'convnext',
    'darknet',
    'densenet',
    'dla',
    'dpn',
    'efficientnet',
    'vovnet',
    'fbnet',
    'gernet',
    'ghostnet',
    'hardcorenas',
    'hrnet',
    'lcnet',
    'resnet',
    'mixnet',
    'mnasnet',
    'mobilenet_',
    'mobilenetv1',
    'mobilenetv2',
    'mobilenetv3',
    'mobileone',
    'nf_regnet',
    'regnet',
    'repghostnet',
    'repvgg',
    'resnest',
    'resnext',
    'rexnet',
    'spnasnet',
    'tinynet',
]

def check_cnn(name):
    for k in vit_keword:
        if k in name:
            return False
    for k in cnn_keyword:
        if k in name:
            return True
    return False

def get_prefix(name, prev):
    prefix = name.split('_')[0]
    if 'densenet' in name:
        prefix = 'densenet'
    elif 'dla' in name:
        prefix = 'dla'
    elif 'dpn' in name:
        prefix = 'dpn'
    elif 'halonet' in name:
        prefix = 'halonet'
    elif 'resnest' in name:
        prefix = 'resnest'
    elif 'resnetv2' in name:
        prefix = 'resnetv2'
    elif 'resnet' in name:
        prefix = 'resnet'
    if 'tf_' in name:
        prefix = prev
    return prefix

import timm
params = [float(i.strip()) for i in open('timm-1.0.12.log', 'r').readlines()]
names = zip(timm.list_models(), params)
candidate = []
prev = ''
for (name,param) in list(names):
    if 'test_' in name: continue
    if param <= 15:
        prefix = get_prefix(name, prev)
        item = name+','+str(param)+','+check_size(name)
        if check_cnn(name):
            candidate.append('cnn,'+item)
        else:
            candidate.append('vit,'+item)
        prev = prefix
open('timm-15M.log', 'w').write('\n'.join(candidate))

lines = open('tmp.log', 'r').readlines()
inputs = open('timm-1.0.12-15M.log', 'r').readlines()
output = []
n = [0 for i in range(10)]
it = 0
name = ''
for l in lines:
    if 'C' == l[0]:
        name = l.strip().split()[-1]
    elif '[' == l[0]:
        pass
    else:
        gmacs = float(l.strip())
        if gmacs < 2000 and it < len(inputs):
            while name != inputs[it].strip().split(',')[1]:
                it += 1
            output.append(inputs[it][0:-1]+','+str(gmacs)+'\n')
            it += 1
        if gmacs <= 500: n[0] += 1
        elif gmacs <= 1000: n[1] += 1
        elif gmacs <= 1500: n[2] += 1
        elif gmacs <= 2000: n[3] += 1
        elif gmacs <= 2500: n[4] += 1
        elif gmacs <= 3000: n[5] += 1
        elif gmacs <= 3500: n[6] += 1
        elif gmacs <= 4000: n[7] += 1
        elif gmacs <= 4500: n[8] += 1
        else: n[9] += 1

open('timm-1.0.12-15M-2G.log', 'w').write(''.join(output))

lines = open('timm-1.0.12-15M-2G.log', 'r').readlines()
candidate = []
prev = ''
for l in lines:
    [vit, name] = l.split(',')[0:2]
    prefix = get_prefix(name, prev)
    if prefix != prev:
        candidate.append('-------'+vit+'-------\n')
    candidate.append(l)
    prev = prefix
open('timm-1.0.12-15M-2G-ViT.log', 'w').write(''.join(candidate))