import subprocess
lines = open("timm-models.log").readlines()
for l in lines:
    if "[-]" in l:
        w = l.split()
        model = w[0][2:-2]
        size = w[-1]
        cmd = f"python python/convert.py --extern-model {model},{size} --non-pretrained --get-metrics 2>/dev/null"
        subprocess.run(cmd, shell=True)
        cmd = f"python python/convert.py --extern-model {model},{size} --non-pretrained --format onnx 2>/dev/null | grep xxx"
        subprocess.run(cmd, shell=True)
        cmd = f".libs/MNN/build/MNNConvert -f ONNX --modelFile .onnx/fp32/{model}.onnx --MNNModel .mnn/{model}.mnn --bizCode MNN --fp16 | grep xxx"
        subprocess.run(cmd, shell=True)
        cmd = f"MODEL={model} SIZE={size} FP=16 make mnn-model-perf | grep load_percentage"
        subprocess.run(cmd, shell=True)
