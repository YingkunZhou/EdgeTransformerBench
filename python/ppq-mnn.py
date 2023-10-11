"""
reference code:
    - https://github.com/openppl-public/ppq/blob/master/ppq/samples/quantize_onnx_model.py
    - https://github.com/openppl-public/ppq/blob/master/md_doc/inference_with_ncnn.md
"""

"""
MODEL=resnet50
python python/ppq-mnn.py --only-convert $MODEL
.libs/MNN/install-2.6.2/bin/MNNConvert -f ONNX --modelFile .mnn/ppq-int8/$MODEL.quantized.onnx --MNNModel .mnn/ppq-int8/$MODEL.quantized.mnn --bizCode MNN
.libs/MNN/install-2.6.2/bin/quantized.out .mnn/ppq-int8/$MODEL.quantized.mnn .mnn/ppq-int8/$MODEL.mnn .mnn/ppq-int8/$MODEL.quantized.json
"""

import argparse
import torch
from ppq import *
from ppq.api import *
from main import build_dataset

def get_args_parser():
    parser = argparse.ArgumentParser(
        'ppq quantization script', add_help=False)
    parser.add_argument('--batch-size', default=1, type=int)
    # Model parameters
    parser.set_defaults(pretrained=True)
    parser.add_argument('--fuse', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--non-pretrained', action='store_false', dest='pretrained')
    parser.add_argument('--weights', default='weights', type=str, help='weigths path')
    parser.add_argument('--only-convert', default='', type=str, help='only test a certain model series')
    # Dataset parameters
    parser.add_argument('--validation', action='store_true', default=False)
    parser.add_argument('--data-path', default='.mnn/calibration', type=str, help='dataset path')
    parser.add_argument('--num_workers', default=2, type=int)

    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ppq quantization script', parents=[get_args_parser()])
    args = parser.parse_args()

    for name, resolution in [
        ('efficientformerv2_s0', 224),
        ('efficientformerv2_s1', 224),
        ('efficientformerv2_s2', 224),

        ('SwiftFormer_XS', 224),
        ('SwiftFormer_S' , 224),
        ('SwiftFormer_L1', 224),

        ('EMO_1M', 224),
        ('EMO_2M', 224),
        ('EMO_5M', 224),
        ('EMO_6M', 224),

        ('edgenext_xx_small', 256),
        ('edgenext_x_small' , 256),
        ('edgenext_small'   , 256),

        ('mobilevitv2_050', 256),
        ('mobilevitv2_075', 256),
        ('mobilevitv2_100', 256),
        ('mobilevitv2_125', 256),
        ('mobilevitv2_150', 256),
        ('mobilevitv2_175', 256),
        ('mobilevitv2_200', 256),

        ('mobilevit_xx_small', 256),
        ('mobilevit_x_small' , 256),
        ('mobilevit_small'   , 256),

        ('LeViT_128S', 224),
        ('LeViT_128' , 224),
        ('LeViT_192' , 224),
        ('LeViT_256' , 224),

        ('resnet50', 224),
        ('mobilenetv3_large_100', 224),
        ('tf_efficientnetv2_b0' , 224),
        ('tf_efficientnetv2_b1' , 240),
        ('tf_efficientnetv2_b2' , 260),
        ('tf_efficientnetv2_b3' , 300),
    ]:
        if args.only_convert and args.only_convert not in name:
            continue

        args.usi_eval = False
        args.model = name
        args.input_size = resolution

        DEVICE = 'cuda'
        PLATFORM = TargetPlatform.MNN_INT8
        quant_setting = QuantizationSettingFactory.mnn_setting()

        # run quantization
        dataset_val = build_dataset(args)
        calibration_dataset = [i[0] for i in dataset_val]
        calibration_dataloader = torch.utils.data.DataLoader(
            dataset=calibration_dataset,
            batch_size=1, shuffle=True)

        def collate_fn(batch: torch.Tensor) -> torch.Tensor:
            return batch.to(DEVICE)

        # AssertionError: Calibration steps is too large, ppq can quantize your network within 8-512 calibration steps. More calibration steps will greatly delay ppq's calibration procedure. Reset your calib_steps parameter please.
        calib_steps = max(min(512, len(dataset_val)), 8)   # 8 ~ 512
        graph = load_onnx_graph(onnx_import_file = ".onnx/" + args.model + ".onnx")
        print('网络正量化中，根据你的量化配置，这将需要一段时间:')
        quantized = quantize_native_model(
            setting=quant_setting,                     # setting 对象用来控制标准量化逻辑
            model=graph,
            calib_dataloader=calibration_dataloader,
            calib_steps=calib_steps,
            input_shape=[1, 3, resolution, resolution], # 如果你的网络只有一个输入，使用这个参数传参
            inputs=None,                    # 如果你的网络有多个输入，使用这个参数传参，就是 input_shape=None, inputs=[torch.zeros(1,3,224,224), torch.zeros(1,3,224,224)]
            collate_fn=lambda x: x.to(DEVICE),  # collate_fn 跟 torch dataloader 的 collate fn 是一样的，用于数据预处理，
                                                        # 你当然也可以用 torch dataloader 的那个，然后设置这个为 None
            platform=PLATFORM,
            device=DEVICE,
            do_quantize=True)

        # -------------------------------------------------------------------
        # PPQ 计算量化误差时，使用信噪比的倒数作为指标，即噪声能量 / 信号能量
        # 量化误差 0.1 表示在整体信号中，量化噪声的能量约为 10%
        # 你应当注意，在 graphwise_error_analyse 分析中，我们衡量的是累计误差
        # 网络的最后一层往往都具有较大的累计误差，这些误差是其前面的所有层所共同造成的
        # 你需要使用 layerwise_error_analyse 逐层分析误差的来源
        # -------------------------------------------------------------------
        print('正计算网络量化误差(SNR)，最后一层的误差应小于 0.1 以保证量化精度:')
        reports = graphwise_error_analyse(
            graph=quantized, running_device=DEVICE, steps=32,
            dataloader=calibration_dataloader, collate_fn=lambda x: x.to(DEVICE))
        for op, snr in reports.items():
            if snr > 0.1: ppq_warning(f'层 {op} 的累计量化误差显著，请考虑进行优化')

        REQUIRE_ANALYSE = False
        if REQUIRE_ANALYSE:
            print('正计算逐层量化误差(SNR)，每一层的独立量化误差应小于 0.1 以保证量化精度:')
            layerwise_error_analyse(graph=quantized, running_device=DEVICE,
                                    interested_outputs=None,
                                    dataloader=calibration_dataloader, collate_fn=lambda x: x.to(DEVICE))

        print('网络量化结束，正在生成目标文件:')
        export_ppq_graph(
            graph=quantized, platform=PLATFORM,
            graph_save_to=".mnn/ppq-int8/" + args.model + '.quantized.onnx',
            config_save_to=".mnn/ppq-int8/" + args.model + '.quantized.json')
