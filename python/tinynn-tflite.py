#https://github.com/alibaba/TinyNeuralNetwork/blob/main/examples/converter/convert.py
import argparse

import torch
import torch.quantization as torch_q
from timm.models import create_model
from tinynn.converter import TFLiteConverter
from tinynn.graph.tracer import model_tracer
from tinynn.graph.quantization.quantizer import QATQuantizer
from tinynn.graph.quantization.fake_quantize import set_ptq_fake_quantize
from tinynn.util.quantization_analysis_util import graph_error_analysis, layer_error_analysis
from tinynn.util.train_util import DLContext

import sota.efficientformer_v2
import sota.swiftformer
import sota.edgenext
import sota.edgenext_bn_hs
import sota.emo
import sota.mobilevit
import sota.mobilevit_v2
import sota.levit
import sota.levit_c
from main import build_dataset

def get_args_parser():
    parser = argparse.ArgumentParser(
        'tinynn quantization script', add_help=False)
    parser.add_argument('--batch-size', default=1, type=int)
    # Model parameters
    parser.set_defaults(pretrained=True)
    parser.add_argument('--non-pretrained', action='store_false', dest='pretrained')
    parser.add_argument('--weights', default='weights', type=str, help='weigths path')
    parser.add_argument('--only-convert', default='', type=str, help='only test a certain model series')
    parser.add_argument('--format', default='fp32', type=str, help='model datatype')
    # Dataset parameters
    parser.add_argument('--data-path', default='.ncnn/calibration', type=str, help='dataset path')
    parser.add_argument('--num_workers', default=4, type=int)

    return parser

def calibrate(model, context: DLContext):
    """Calibrates the fake-quantized model

    Args:
        model: The model to be validated
        context (DLContext): The context object
        eval: Flag to set train mode when used to do BN restore
    """
    model.eval()

    with torch.no_grad():
        for i, (image, _) in enumerate(context.train_loader):
            if context.max_iteration is not None and i >= context.max_iteration:
                break
            model(image)
            if i % 10 == 0:
                print(f'Calibrate: [{i}/{len(context.train_loader)}]')
            context.iteration += 1

def main(args):
     for model_name, resolution, weight in [
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
        # ERROR (tinynn.converter.base) Unsupported ops: aten::ceil
        #('tf_efficientnetv2_b0' , 224, None),
        #('tf_efficientnetv2_b1' , 240, None),
        #('tf_efficientnetv2_b2' , 260, None),
        #('tf_efficientnetv2_b3' , 300, None),
    ]:
        args.usi_eval = False
        args.model = model_name
        args.input_size = resolution

        if args.only_convert and args.only_convert not in args.model:
            continue

        with model_tracer():
            print(f"Creating model: {args.model}")
            model = create_model(
                args.model,
                pretrained= not weight and args.pretrained,
            )
            if weight and args.pretrained:
                # load model weights
                weights_dict = torch.load(args.weights+'/'+weight, map_location="cpu")

                if "state_dict" in weights_dict:
                    print(args.model)
                    args.usi_eval = True
                    weights_dict = weights_dict["state_dict"]
                elif "model" in weights_dict:
                    weights_dict = weights_dict["model"]

                if "LeViT_c_" in args.model:
                    D = model.state_dict()
                    for k in weights_dict.keys():
                        if D[k].shape != weights_dict[k].shape:
                            weights_dict[k] = weights_dict[k][:, :, None, None]

                model.load_state_dict(weights_dict)

            if "LeViT" in args.model:
                sota.levit.replace_batchnorm(model)  # TODO: speedup levit

        dummy_input = torch.rand((1, 3, args.input_size, args.input_size))
        model.eval()
        # The code section below is used to convert the model to the TFLite format
        # If you want perform dynamic quantization on the float models,
        # you may refer to `dynamic.py`, which is in the same folder.
        # As for static quantization (e.g. quantization-aware training and post-training quantization),
        # please refer to the code examples in the `examples/quantization` folder.
        if args.format == "fp32":
            converter = TFLiteConverter(
                model,
                dummy_input,
                ".tflite/tinynn-32/"+args.model+'.tflite',
                nchw_transpose=False
            )
            converter.convert()
        elif args.format == "dynamic":
            converter = TFLiteConverter(
                model,
                dummy_input,
                tflite_path='.tflite/tinynn-d8/'+args.model+'.tflite',
                quantize_target_type='int8', # Defaults to 'uint8'
                strict_symmetric_check=True,
                # Enable hybrid quantization
                hybrid_quantization_from_float=True,
                # Enable hybrid per-channel quantization (lower q-loss, but slower??? really)
                hybrid_per_channel=True,
                # Use asymmetric inputs for hybrid quantization (probably lower q-loss, but a bit slower)
                hybrid_asymmetric_inputs=True,
                # Enable hybrid per-channel quantization for `Conv2d` and `DepthwiseConv2d`
                hybrid_conv=True,
                nchw_transpose=False,
            )
            converter.convert()
        elif args.format == "int8":
            # https://github.com/alibaba/TinyNeuralNetwork/blob/main/examples/quantization/specific/vit/vit_post.py
            # Use quantizer to get layerwise config to do mixed quantization
            with model_tracer():
                # More information for QATQuantizer initialization, see `examples/quantization/qat.py`.
                # We set 'override_qconfig_func' when initializing QATQuantizer to use fake-quantize to do post quantization.
                quantizer = QATQuantizer(
                    model,
                    dummy_input,
                    work_dir='out',
                    config={
                        'override_qconfig_func': set_ptq_fake_quantize,
                    },
                )
                quantizer.quantize()

            layerwise_config_cur = quantizer.layerwise_config
            # Keep all residual add and gelu layer FP calculation,
            # and keep two quantization sensitive fc layer FP calculation additionally.
            for name in quantizer.layerwise_config:
                if (
                    name.startswith('add_')
                    or name.startswith('gelu')
                    or 'native_batch_norm' in name
                ):
                    layerwise_config_cur[name] = False
                else:
                    layerwise_config_cur[name] = True

            with model_tracer():
                # More information for QATQuantizer initialization, see `examples/quantization/qat.py`.
                # We set 'override_qconfig_func' when initializing QATQuantizer to use fake-quantize to do post quantization.
                quantizer = QATQuantizer(
                    model,
                    dummy_input,
                    work_dir='out',
                    config={
                        "layerwise_config": layerwise_config_cur,
                        'override_qconfig_func': set_ptq_fake_quantize,
                        "force_overwrite": True,
                        'set_quantizable_op_stats': True,
                    },
                )
                ptq_model = quantizer.quantize()

                context = DLContext()
                dataset_val = build_dataset(args)
                data_loader_val = torch.utils.data.DataLoader(
                    dataset_val,
                    batch_size=1,
                    # shuffle=True,
                    num_workers=args.num_workers, # acc val speed
                    pin_memory=True,
                    sampler=None,
                )

                context.train_loader = data_loader_val
                context.max_iteration = 100
                # Post quantization calibration
                ptq_model.apply(torch_q.disable_fake_quant)
                ptq_model.apply(torch_q.enable_observer)
                calibrate(ptq_model, context)

                # Disable observer and enable fake quantization to validate model with quantization error
                ptq_model.apply(torch_q.disable_observer)
                ptq_model.apply(torch_q.enable_fake_quant)
                ptq_model(dummy_input)

                # Perform quantization error analysis with real dummy input
                dummy_input_real = next(iter(context.train_loader))[0][:1]

                # The error is accumulated by directly giving the difference in layer output
                # between the quantized model and the floating model. If you want a quantized model with high accuracy,
                # the layers closest to the final output should be less than 10%, which means the final
                # layer's cosine similarity should be greater than 90%.
                graph_error_analysis(ptq_model, dummy_input_real, metric='cosine')

                # We quantize each layer separately and compare the difference
                # between the original output and the output with quantization error for each layer,
                # which is used to calculate the quantization sensitivity of each layer.
                layer_error_analysis(ptq_model, dummy_input_real, metric='cosine')

                ptq_model.eval()
                ptq_model = torch.quantization.convert(ptq_model)
                ptq_model(dummy_input)

                torch.backends.quantized.engine = quantizer.backend

                # The code section below is used to convert the model to the TFLite format
                # If you need a quantized model with a specific data type (e.g. int8)
                # you may specify `quantize_target_type='int8'` in the following line.
                # If you need a quantized model with strict symmetric quantization check (with pre-defined zero points),
                # you may specify `strict_symmetric_check=True` in the following line.
                converter = TFLiteConverter(
                    ptq_model,
                    dummy_input,
                    tflite_path='.tflite/tinynn-s8/'+args.model+'.tflite',
                    quantize_target_type='int8',
                    rewrite_quantizable=True,
                    nchw_transpose=False,
                )
                converter.convert()



if __name__ == '__main__':
    parser = argparse.ArgumentParser('tinynn quantization script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
