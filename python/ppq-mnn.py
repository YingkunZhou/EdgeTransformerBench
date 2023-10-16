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
        # use --opset-version 13
        ('efficientformerv2_s0', 224),
        ('efficientformerv2_s1', 224),
        ('efficientformerv2_s2', 224),

        ('SwiftFormer_XS', 224),
        ('SwiftFormer_S' , 224),
        ('SwiftFormer_L1', 224),

        # NotImplementedError: Graph op: /stage4.1/Mod_2(Mod) has no backend implementation on target platform TargetPlatform.SOI.
        # Register this op to ppq.executor.base.py and ppq.executor.op first
        #('EMO_1M', 224),
        #('EMO_2M', 224),
        #('EMO_5M', 224),
        #('EMO_6M', 224),

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
        # https://github.com/openppl-public/ppq/blob/master/ppq/api/setting.py#L333
        # https://github.com/openppl-public/ppq/blob/master/ppq/samples/Tutorial/bestPractice.py
        print(quant_setting.quantize_activation)
        print(quant_setting.quantize_activation_setting.calib_algorithm)
        print(quant_setting.quantize_parameter)
        print(quant_setting.quantize_parameter_setting.calib_algorithm)
        print(quant_setting.lsq_optimization)
        print(quant_setting.equalization)
        print(quant_setting.dispatcher)
        #quant_setting.equalization = True # use layerwise equalization algorithm.
        #quant_setting.dispatcher   = 'conservative' # dispatch this network in conservertive way.
        #QSetting.quantize_activation_setting.calib_algorithm = 'kl'
        #QSetting.quantize_parameter_setting.calib_algorithm  = 'minmax'

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
        # TODO: use onnxsim to sim the onnx model first
        quantized = quantize_onnx_model(
            onnx_import_file=".onnx/" + args.model + ".sim.onnx",
            calib_dataloader=calibration_dataloader,
            calib_steps=calib_steps, input_shape=[1, 3, resolution, resolution],
            setting=quant_setting, collate_fn=collate_fn,
            platform=PLATFORM, device=DEVICE, verbose=0
        )
        debug = False
        if debug:
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

        export_ppq_graph(
            graph=quantized, platform=PLATFORM,
            graph_save_to=".mnn/ppq-int8/" + args.model + '.quantized.onnx',
            config_save_to=".mnn/ppq-int8/" + args.model + '.quantized.json')
