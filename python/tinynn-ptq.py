import argparse
import torch
import torch.nn as nn
from timm.models import create_model

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

from tinynn.converter import TFLiteConverter
from tinynn.graph.quantization.quantizer import PostQuantizer
from tinynn.graph.tracer import model_tracer
from tinynn.util.cifar10 import calibrate
from tinynn.util.train_util import DLContext, get_device
from tinynn.graph.quantization.algorithm.cross_layer_equalization import cross_layer_equalize

def main_worker(args):
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

        args.model = name
        args.input_size = resolution
        args.usi_eval = False

        with model_tracer():
            print(f"Creating model: {name}")
            model = create_model(
                name,
                pretrained= not weight and args.pretrained,
            )
            if weight and args.pretrained:
                # load model weights
                weights_dict = torch.load(args.weights+'/'+weight, map_location="cpu")

                if "state_dict" in weights_dict:
                    args.usi_eval = True
                    weights_dict = weights_dict["state_dict"]
                elif args.model_ema: # for EdgeNeXt
                    weights_dict = weights_dict["model_ema"]
                elif "model" in weights_dict:
                    weights_dict = weights_dict["model"]

                if "LeViT_c_" in name:
                    D = model.state_dict()
                    for k in weights_dict.keys():
                        if D[k].shape != weights_dict[k].shape:
                            weights_dict[k] = weights_dict[k][:, :, None, None]

                model.load_state_dict(weights_dict)

            if args.fuse:
                sota.levit.replace_batchnorm(model)  # TODO: acc val speed & acc

            # Provide a viable input for the model
            dummy_input = torch.rand((1, 3, 224, 224))

            # For per-tensor quantization, if there are many outliers in the weight, CLE can significantly improve the
            # quantization accuracy
            if args.cle:
                cross_layer_equalize(model, dummy_input, get_device())
            quantizer = PostQuantizer(model, dummy_input, work_dir='.tflite')
            ptq_model = quantizer.quantize()

        # print(ptq_model)

        # Use DataParallel to speed up calibrating when possible
        if torch.cuda.device_count() > 1:
            ptq_model = nn.DataParallel(ptq_model)

        # Move model to the appropriate device
        device = get_device()
        ptq_model.to(device=device)

        context = DLContext()
        context.device = device
        dataset_val = build_dataset(args)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        context.train_loader = torch.utils.data.DataLoader(
            dataset_val,
            sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.workers, # acc val speed
            pin_memory=args.pin_mem,
            drop_last=False
        )
        context.max_iteration = 1000

        # Post quantization calibration
        calibrate(ptq_model, context)

        with torch.no_grad():
            ptq_model.eval()
            ptq_model.cpu()

            # The step below converts the model to an actual quantized model, which uses the quantized kernels.
            ptq_model = quantizer.convert(ptq_model)

            # When converting quantized models, please ensure the quantization backend is set.
            torch.backends.quantized.engine = quantizer.backend

            # The code section below is used to convert the model to the TFLite format
            # If you need a quantized model with a specific data type (e.g. int8)
            # you may specify `quantize_target_type='int8'` in the following line.
            # If you need a quantized model with strict symmetric quantization check (with pre-defined zero points),
            # you may specify `strict_symmetric_check=True` in the following line.
            converter = TFLiteConverter(ptq_model, dummy_input, tflite_path='.tflite/ptq_model.tflite')
            converter.convert()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.set_defaults(pretrained=True)
    parser.add_argument('--non-pretrained', action='store_false', dest='pretrained')
    parser.add_argument('--data-path', metavar='DIR', default=".ncnn/calibration", help='path to dataset')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--weights', default='./weights', metavar='DIR', help='weights path')
    parser.add_argument('--cle', type=bool, default=False)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--only-convert', default='', type=str, help='only test a certain model series')
    parser.add_argument('--fuse', action='store_true', default=False)

    args = parser.parse_args()

    main_worker(args)