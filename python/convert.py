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

    parser.add_argument('--tvm_dev', default="orpi5b", type=str, help='reference to remote_device_list')
    parser.add_argument('--tvm_frontend', default="pytorch", type=str, help='pytorch, onnx')
    parser.add_argument('--tvm_backend', default="cpu", type=str, help='reference to remote_device_list')
    parser.add_argument('--tvm_tune_method',default="None", type=str, help='None,AutoTVM, AutoScheduler')
    parser.add_argument('--tvm_only_tune', action='store_true', default=False)
    parser.add_argument('--tvm_only_upload', action='store_true', default=False)
    parser.add_argument('--tvm_remote_run', action='store_true', default=True)
    parser.add_argument('--tvm_data_precision',default="fp32", type=str, help='fp32,mixed,fp16')
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

        if not args.format or args.format == 'tvm':
            """
                python -m tvm.exec.rpc_server --tracker=192.168.3.170:9190 --key={remote_device_name}
            """
            import subprocess
            import tvm
            import tvm.relay as relay
            import onnx
            from tvm import rpc
            from tvm.contrib import utils, graph_executor
            from tvm.autotvm.measure import request_remote
            from tvm_utils import find_device_by_name

            # cmd = "python -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190"
            # rpc_process =subprocess.Popen(cmd, shell=True)

            remote_device_name=args.tvm_dev
            remote_device=find_device_by_name(remote_device_name)
            # remote = request_remote(remote_device_name, "127.0.0.1", port=9190)
            # print(remote.cl().exist)
            print(remote_device_name)
            local_lib_dir = os.path.join(".tvm",remote_device_name,"lib")
            if(os.path.exists(local_lib_dir)==False):
                os.makedirs(local_lib_dir)
            # local_lib_filename = name+"_"+args.tvm_tune_method+".tar"
            local_lib_filename = "_".join([name,args.tvm_backend ,args.tvm_data_precision,args.tvm_tune_method])+".tar"
            local_lib_path = os.path.join(local_lib_dir,local_lib_filename)
            remote_lib_path = local_lib_filename
            if(args.tvm_backend=="cpu"):
                target = tvm.target.Target(remote_device.cpu_target)
            elif(args.tvm_backend=="opencl"):
                target = tvm.target.Target(target=remote_device.opencl_target,host=remote_device.cpu_target)
            elif(args.tvm_backend=="vulkan"):
                target = remote_device.vulkan_target
            else:
                raise NotImplementedError

            print(str(target))

            input_name = 'input'
            if args.tvm_frontend == 'onnx':
                onnx_model  = onnx.load('.onnx/fp32/'+name+'.onnx')
                mod, params = relay.frontend.from_onnx(onnx_model, {input_name: inputs.shape})
            elif args.tvm_frontend == 'pytorch':
                trace_model = torch.jit.trace(model, inputs)
                mod, params = relay.frontend.from_pytorch(trace_model, [(input_name, inputs.shape)])
            else:
                raise NotImplementedError

            if args.tvm_data_precision!="fp32":
                print("************************")
                from tvm.driver.tvmc.transform import convert_to_mixed_precision
                mod = convert_to_mixed_precision(
                    mod,
                    ops=None,
                    calculation_type="float16",
                    acc_type="float32" if args.tvm_data_precision == "mixed" else "float16",
                )
            tuning_records_dir = os.path.join(".tvm/",remote_device_name,args.tvm_backend,args.tvm_data_precision,"tuning_records")
            if(os.path.exists(tuning_records_dir)==False):
                os.makedirs(tuning_records_dir)
            tuning_records_filename=name+"_"+args.tvm_tune_method+".json"
            tuning_records = os.path.join(tuning_records_dir,tuning_records_filename)
            if not args.tvm_only_upload:
                if args.tvm_tune_method!="None":
                    from tvm.driver import tvmc
                    from tvm.driver.tvmc.model import TVMCModel
                    # model = tvmc.frontends.load_model('.onnx/fp32/'+name+'.onnx')
                    model = TVMCModel(mod,params)
                    tvmc.tune(
                        model,
                        target=str(target),
                        # Compilation target as string // Device to compile for
                        hostname='127.0.0.1', # The IP address of an RPC tracker, used when benchmarking remotely.
                        port=9190, # The port of the RPC tracker to connect to. Defaults to 9090.
                        rpc_key=remote_device_name, # The RPC tracker key of the target device. Required when rpc_tracker is provided
                        tuning_records=tuning_records,
                        number=10,
                        repeat=1,
                        parallel=1,
                        early_stopping=100,
                        min_repeat_ms=0, # since we're tuning on a CPU, can be set to 0
                        timeout=10, # in seconds
                        enable_autoscheduler=(args.tvm_tune_method == 'AutoScheduler'),
                        trials=1000 if args.tvm_tune_method == 'AutoScheduler' else  5000,
                        # mixed_precision = False if args.tvm_data_precision == "fp32" else True,
                        # # mixed_precision_ops = ["nn.conv2d", "nn.dense"],
                        # mixed_precision_calculation_type = "float16",
                        # mixed_precision_acc_type = "float32" if args.tvm_data_precision == "mixed" else "float16",
                        # hardware_params=hardware_params,
                    )

            if not args.tvm_only_tune:
                """
                python -m tvm.exec.rpc_server --host 0.0.0.0 --port=9090
                """
                # from tvm.autotvm.measure import request_remote

                # target=tvm.target.Target(target)
                if args.tvm_tune_method == 'AutoTVM':
                    from tvm import autotvm
                    with autotvm.apply_history_best(tuning_records):
                        with tvm.transform.PassContext(opt_level=3, config={}):
                            built_lib = relay.build(mod, target=target, params=params)
                elif args.tvm_tune_method == 'AutoScheduler':
                    from tvm import auto_scheduler
                    with auto_scheduler.ApplyHistoryBest(tuning_records):
                        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
                            built_lib = relay.build(mod, target=target, params=params)
                else:
                    with tvm.transform.PassContext(opt_level=3):
                        built_lib = relay.build(mod, target, params=params)
                    #  built_lib = relay.build(mod, target, params=params)




                built_lib.export_library(local_lib_path)

                 # TODO
                print(f"export success, find it in{remote_lib_path}")


            if args.tvm_remote_run:
                remote = request_remote(remote_device_name, "127.0.0.1", port=9190)

                remote.upload(local_lib_path, remote_lib_path)
                print(f"upload lib success, in {remote_lib_path }")
                rlib = remote.load_module(remote_lib_path)

                # create the remote runtime module
                if args.tvm_backend == 'cpu':
                    dev = remote.cpu()
                elif args.tvm_backend == 'opencl':
                    dev = remote.cl()

                module = graph_executor.GraphModule(rlib["default"](dev))
                module.set_input(input_name, tvm.nd.array(inputs.numpy()))
                module.run()
                print(module.benchmark(dev, repeat=1))
                print(f"build .so success, find it in {remote_lib_path }.so")
            # rpc_process.kill()

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