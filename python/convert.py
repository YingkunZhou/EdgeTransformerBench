"""
Modified from
https://github.com/facebookresearch/LeViT/blob/main/speed_test.py
"""

import os
import argparse
import subprocess
import torch
from timm.models import create_model
from main import build_dataset
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

import copy

import sota.efficientformer_v2
import sota.swiftformer
import sota.edgenext
import sota.edgenext_bn_hs
import sota.emo
import sota.mobilevit
import sota.mobilevit_v2
import sota.levit
import sota.levit_c

torch.autograd.set_grad_enabled(False)

def get_args_parser():
    parser = argparse.ArgumentParser(
        'EdgeTransformerPerf model format conversion script', add_help=False)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--opset-version', default=None, type=int)
    # by: ln -sf ../.ncnn/calibration .
    parser.add_argument('--data-path', default='.pt/calibration', type=str, help='dataset path')
    # Model parameters
    parser.set_defaults(pretrained=True)
    parser.add_argument('--non-pretrained', action='store_false', dest='pretrained')
    parser.add_argument('--fuse', action='store_true', default=False)
    parser.add_argument('--mobile', default=None, type=str, help='cpu, vulkan, nnapi')
    parser.add_argument('--only-convert', default='', type=str, help='only test a certain model series')
    parser.add_argument('--format', default='', type=str, help='conversion format')
    parser.add_argument('--debug', default=None, type=str, help='e,g --debug 32,4')
    parser.add_argument('--int8', action='store_true', default=False)
    parser.add_argument('--trt-dev', default="gpu", type=str, help='gpu, dla')

    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser('EdgeTransformerPerf evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    # BackendIsNotSupposedToImplementIt: Unsqueeze version 13 is not implemented.
    # https://github.com/onnx/onnx-tensorflow/issues/997
    for name, resolution, weight in [
        ('efficientformerv2_s0' , 224, "eformer_s0_450.pth"),
        ('efficientformerv2_s1' , 224, "eformer_s1_450.pth"),
        ('efficientformerv2_s2' , 224, "eformer_s2_450.pth"),
        ('SwiftFormer_XS'       , 224, "SwiftFormer_XS_ckpt.pth"),
        ('SwiftFormer_S'        , 224, "SwiftFormer_S_ckpt.pth"),
        ('SwiftFormer_L1'       , 224, "SwiftFormer_L1_ckpt.pth"),
        ('EMO_1M'               , 224, "EMO_1M.pth"),
        ('EMO_2M'               , 224, "EMO_2M.pth"),
        ('EMO_5M'               , 224, "EMO_5M.pth"),
        ('EMO_6M'               , 224, "EMO_6M.pth"),
        ('edgenext_xx_small'    , 256, "edgenext_xx_small.pth"),
        ('edgenext_x_small'     , 256, "edgenext_x_small.pth"),
        ('edgenext_small'       , 256, "edgenext_small_usi.pth"),
        ('mobilevitv2_050'      , 256, "mobilevitv2-0.5.pt"),
        ('mobilevitv2_075'      , 256, "mobilevitv2-0.75.pt"),
        ('mobilevitv2_100'      , 256, "mobilevitv2-1.0.pt"),
        ('mobilevitv2_125'      , 256, "mobilevitv2-1.25.pt"),
        ('mobilevitv2_150'      , 256, "mobilevitv2-1.5.pt"),
        ('mobilevitv2_175'      , 256, "mobilevitv2-1.75.pt"),
        ('mobilevitv2_200'      , 256, "mobilevitv2-2.0.pt"),
        ('mobilevit_xx_small'   , 256, "mobilevit_xxs.pt"),
        ('mobilevit_x_small'    , 256, "mobilevit_xs.pt"),
        ('mobilevit_small'      , 256, "mobilevit_s.pt"),
        ('LeViT_128S'           , 224, "LeViT-128S.pth"),
        ('LeViT_128'            , 224, "LeViT-128.pth"),
        ('LeViT_192'            , 224, "LeViT-192.pth"),
        ('LeViT_256'            , 224, "LeViT-256.pth"),
        ('resnet50'             , 224, None),
        ('mobilenetv3_large_100', 224, None),
        ('tf_efficientnetv2_b0' , 224, None),
        ('tf_efficientnetv2_b1' , 240, None),
        ('tf_efficientnetv2_b2' , 260, None),
        ('tf_efficientnetv2_b3' , 300, None),
    ]:
        if args.only_convert and args.only_convert not in name:
            continue
        args.usi_eval = False
        args.model = name
        args.input_size = resolution

        print(f"Creating model: {name}")
        model = create_model(
            name,
            pretrained= not weight and args.pretrained,
        )
        if weight and args.pretrained:
            # load model weights
            if not os.path.exists('weights'):
                if not os.path.exists('EdgeTransformerPerf-weights.tar'):
                    print("============Downloading weights============")
                    print("============you should install gdown first: pip install gdown============")
                    subprocess.run(['gdown', '19irI6H_c1w2OaDOVzPIj2v0Dy30pq-So'])
                print("============Extracting weights============")
                subprocess.run(['tar', 'xf', 'EdgeTransformerPerf-weights.tar'])

            weights_dict = torch.load('weights/'+weight, map_location="cpu")
            # print(weights_dict.keys())

            if "state_dict" in weights_dict:
                print(args.model)
                args.usi_eval = True
                weights_dict = weights_dict["state_dict"]
            elif "model" in weights_dict:
                weights_dict = weights_dict["model"]

            if "LeViT_c_" in name:
                D = model.state_dict()
                for k in weights_dict.keys():
                    if D[k].shape != weights_dict[k].shape:
                        weights_dict[k] = weights_dict[k][:, :, None, None]

            model.load_state_dict(weights_dict)

        if args.fuse:
            sota.levit.replace_batchnorm(model)  # TODO: speedup levit

        # TODO: does onnx export need this?
        model.eval()

        channels = 3
        if args.debug:
            name = 'debug'
            shapes = args.debug.split(',')
            if len(shapes) > 0 and shapes[0] != '':
                channels = int(shapes[0])
                if len(shapes) > 1:
                    resolution = int(shapes[1])
            print(channels, resolution)

        inputs = torch.randn(
            1, #args.batch_size, TODO: here we only support single batch size benchmarking
            channels, resolution, resolution,
        )

        if not args.format or args.format == 'onnx':
            if not os.path.exists(".onnx/fp32"):
                os.makedirs(".onnx/fp32")
            torch.onnx.export(
                model,
                inputs,
                '.onnx/fp32/'+name+'.onnx',
                export_params=True,
                input_names=['input'],
                output_names=['output'],
                do_constant_folding=True,
                opset_version=args.opset_version
            )
        if not args.format or args.format == 'trt':
            trace_model = torch.jit.trace(model, inputs).cuda()

            import torch_tensorrt
            import torch.utils.data as tdata
            import torch_tensorrt.ptq as tptq

            calib_dataset = build_dataset(args)
            calib_dataset = tdata.random_split(calib_dataset, [len(calib_dataset)-200, 200])[1]
            calib_dataloader = tdata.DataLoader(calib_dataset, batch_size=1, shuffle=False, drop_last=True)
            dev = args.trt_dev
            if args.int8:
                if not os.path.exists(".pt/"+dev+"-int8"):
                    os.makedirs(".pt/"+dev+"-int8")
                calibrator = tptq.DataLoaderCalibrator(
                    calib_dataloader,
                    use_cache=False,
                    # Network built for DLA requires kENTROPY_CALIBRATION_2 calibrator.
                    algo_type=tptq.CalibrationAlgo.MINMAX_CALIBRATION if dev == 'gpu' \
                         else tptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
                    device=torch.device('cuda:0'))

                compile_spec = {
                    "inputs": [torch_tensorrt.Input([1, 3, resolution, resolution])],
                    "enabled_precisions": torch.int8,
                    "calibrator": calibrator,
                    "truncate_long_and_double": True,
                }
                if dev == 'dla':
                    compile_spec["device"] = torch_tensorrt.Device("dla:0", allow_gpu_fallback=True)
                trt_int8 = torch_tensorrt.compile(trace_model, **compile_spec)
                torch.jit.save(trt_int8, ".pt/"+dev+"-int8/"+name+'.ts')
            else:
                if not os.path.exists(".pt/"+dev+"-fp16"):
                    os.makedirs(".pt/"+dev+"-fp16")
                compile_spec = {
                    "inputs": [torch_tensorrt.Input(
                        [1, 3, resolution, resolution],
                        # dtype=torch.half,
                    )],
                    "enabled_precisions": torch.half,
                    "truncate_long_and_double": True,
                    "workspace_size": 8192,
                }
                if dev == 'dla':
                    compile_spec["device"] = torch_tensorrt.Device("dla:0", allow_gpu_fallback=True)
                trt_fp16 = torch_tensorrt.compile(trace_model, **compile_spec)
                torch.jit.save(trt_fp16, ".pt/"+dev+"-fp16/"+name+'.ts')


        if not args.format or args.format == 'coreml':
            if not os.path.exists(".coreml"):
                os.makedirs(".coreml")
            if not os.path.exists(".coreml/int8"):
                os.makedirs(".coreml/int8")
            if not os.path.exists(".coreml/fp16"):
                os.makedirs(".coreml/fp16")

            trace_model = torch.jit.trace(model, inputs)
            import coremltools as ct
            model = ct.convert(
                trace_model,
                # convert_to="neuralnetwork",
                convert_to="mlprogram",
                inputs=[ct.TensorType(name = "inputs", shape=inputs.shape)],
                minimum_deployment_target=ct.target.macOS14,
            )
            model.save(".coreml/fp16/"+name+".mlpackage")
            # model.save(".coreml/"+name+".mlmodel")

            import coremltools.optimize.coreml as cto
            op_config = cto.OpLinearQuantizerConfig(mode="linear_symmetric", weight_threshold=512)
            config = cto.OptimizationConfig(global_config=op_config)
            compressed_8_bit_model = cto.linear_quantize_weights(model, config=config)
            compressed_8_bit_model.save(".coreml/int8/"+name+".mlpackage")

        if not args.format or args.format == 'cann':
            # pip install numpy scipy attrs psutil decorator
            # pip install onnx onnxruntime
            opt_shape = ['--input_shape', 'input:1,3,{},{}'.format(resolution,resolution)]
            opt_model = ['--model', '.onnx/fp32/{}.onnx'.format(args.model)]
            convert_cmd = ['atc', '--mode', '0', '--framework', '5',
                           '--input_format', 'NCHW', '--soc_version', 'Ascend310B4']
            convert_cmd += opt_shape
            if args.int8:
                calib_dataset = build_dataset(args)
                if not os.path.exists(".cann/calibration"):
                    os.makedirs(".cann/calibration")
                num_images = 64
                for i, (image, _) in enumerate(calib_dataset):
                    if i >= num_images: break
                    torch.unsqueeze(image, dim=0).numpy().tofile(".cann/calibration/{}.bin".format(i))
                subprocess.run(['amct_onnx', 'calibration'] + opt_model + opt_shape +
                               ['--data_type', 'float32', '--data_dir', '.cann/calibration',
                                '--save_path', '.cann/int8/{}'.format(args.model),
                                '--batch_num', str(num_images)])
                subprocess.run(convert_cmd + ['--output', '.cann/int8/'+args.model,
                               '--model', '.cann/int8/{}_deploy_model.onnx'.format(args.model)])
            else:
                subprocess.run(convert_cmd + opt_model + ['--output', '.cann/fp16/'+args.model])

        if not args.format or args.format == 'pt':
            if not os.path.exists(".pt/fp32"):
                os.makedirs(".pt/fp32")
            if args.mobile:
                if args.mobile == "nnapi":
                    inputs = inputs.contiguous(memory_format=torch.channels_last)
                    inputs.nnapi_nhwc = True
                trace_model = torch.jit.trace(model, inputs)
                if args.mobile == "nnapi":
                    from torch.backends._nnapi.prepare import convert_model_to_nnapi
                    mobile_model = convert_model_to_nnapi(trace_model, inputs)
                else:
                    from torch.utils.mobile_optimizer import optimize_for_mobile
                    mobile_model = optimize_for_mobile(trace_model, backend=args.mobile)
                mobile_model._save_for_lite_interpreter(".pt/fp32/"+name+'.'+args.mobile[0]+'.ptl', _use_flatbuffer=True)
            else:
                trace_model = torch.jit.trace(model, inputs)
                trace_model.save(".pt/fp32/"+name+'.pt')
