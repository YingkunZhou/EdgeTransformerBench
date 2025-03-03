"""
Reference code:
- https://github.com/Ascend/samples
- https://blog.csdn.net/m0_37605642/article/details/125691134
"""

import os
import time
import acl
import numpy as np
import torch
import argparse
from main import build_dataset, MetricLogger, WARMUP_SEC, TEST_SEC, load_image
from timm.utils import accuracy

ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2

# 6.实现管道读取订阅数据的函数。
# 6.1 自定义函数，实现从用户内存中读取订阅数据的函数。
def get_model_info(data, data_len):
    # 获取算子信息个数。
    op_number, ret = acl.prof.get_op_num(data, data_len)
    # 遍历用户内存的算子信息。
    for i in range(op_number):
        # 获取算子的模型id。
        # model_id = acl.prof.get_model_id(data, data_len, i)
        # 获取算子的类型名称。
        op_type, ret = acl.prof.get_op_type(data, data_len, i, 65)
        # 获取算子的名称。
        op_name, ret = acl.prof.get_op_name(data, data_len, i, 275)
        # 获取算子的执行开始时间。
        # op_start = acl.prof.get_op_start(data, data_len, i)
        # 获取算子的执行结束时间。
        # op_end = acl.prof.get_op_end(data, data_len, i)
        # 获取算子执行的耗时时间。
        # https://www.hiascend.com/doc_center/source/zh/canncommercial/80RC3/apiref/appdevgapi/aclpythondevg_01_0885.html
        op_duration = acl.prof.get_op_duration(data, data_len, i)
        print(op_type, op_name, op_duration)

# 6.2 自定义函数，实现从管道中读取数据到用户内存的函数。
def prof_data_read(args):
    fd, ctx = args
    ret = acl.rt.set_context(ctx)
    # 获取单位算子信息的大小（Byte）。
    buffer_size, ret = acl.prof.get_op_desc_size()
    # 设置每次从管道中读取的算子信息个数。
    N = 10
    # 计算存储算子信息的内存的大小。
    data_len = buffer_size * N
    # 从管道中读取数据到申请的内存中，读取到的实际数据大小可能小于buffer_size * N，如果管道中没有数据，默认会阻塞直到读取到数据为止。
    while True:
        data = os.read(fd, data_len)
        if len(data) == 0:
            break
        np_data = np.array(data)

        bytes_data = np_data.tobytes()
        np_data_ptr = acl.util.bytes_to_ptr(bytes_data)
        size = np_data.itemsize * np_data.size
        # 调用6.1实现的函数解析内存中的数据。
        get_model_info(np_data_ptr, size)


class net:
    def __init__(self, model_path):
        self.device_id = 0

        # step1: init ACL
        ret = acl.init()
        # check_ret("acl.init", ret)
        # set device id
        ret = acl.rt.set_device(self.device_id)
        # https://github.com/search?q=repo%3AAscend%2Fsamples+context+language%3APython&type=code&l=Python
        self.context, ret = acl.rt.create_context(self.device_id)

        # step2: load offline model and return model ID
        self.model_id, ret = acl.mdl.load_from_file(model_path)

        # create an empty model description and get the pointer address of the model description
        self.model_desc = acl.mdl.create_desc()

        # by model_id, fill the model description information into model_desc
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)

        # step3: create a I/O stream
        # create input dataset
        self.input_dataset, self.input_data = self.prepare_dataset('input')
        # create output dataset
        self.output_dataset, self.output_data = self.prepare_dataset('output')

    def prepare_dataset(self, io_type):
        # prepare dataset
        if io_type == "input":
            # acquire the number of inputs of the model
            io_num = acl.mdl.get_num_inputs(self.model_desc) # io_num = 1
            acl_mdl_get_size_by_index = acl.mdl.get_input_size_by_index
        else:
            # acquire the number of outputs of the model
            io_num = acl.mdl.get_num_outputs(self.model_desc) # io_num = 1
            acl_mdl_get_size_by_index = acl.mdl.get_output_size_by_index
        # create aclmdlDataset type data to describe the input of model inference
        dataset = acl.mdl.create_dataset()
        datas = []
        for i in range(io_num):
            # get the size of the input buffer
            buffer_size = acl_mdl_get_size_by_index(self.model_desc, i)
            # apply for memory for the input buffer
            buffer, ret = acl.rt.malloc(buffer_size, ACL_MEM_MALLOC_HUGE_FIRST)
            # create a data buffer from the memory
            data_buffer = acl.create_data_buffer(buffer, buffer_size)
            # add buffer to dataset
            _, ret = acl.mdl.add_dataset_buffer(dataset, data_buffer)
            datas.append({"buffer": buffer, "data": data_buffer, "size": buffer_size})
        return dataset, datas

    def forward(self, inputs):
        # trival all inputs, copy to corresponding buffer memory
        input_num = len(inputs)
        for i in range(input_num):
            bytes_data = inputs[i].tobytes()
            bytes_ptr = acl.util.bytes_to_ptr(bytes_data)
            # transfer image data from host to device
            ret = acl.rt.memcpy(self.input_data[i]["buffer"],   # device: destination address
                    self.input_data[i]["size"],     # device: destination address size
                    bytes_ptr,                      # host: source address
                    len(bytes_data),                # host: source address size
                    ACL_MEMCPY_HOST_TO_DEVICE)      # mode: from host to device

        # execute model inference
        ret = acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset)
        # manage model inference output
        inference_result = []
        for i, item in enumerate(self.output_data):
            buffer_host, ret = acl.rt.malloc_host(self.output_data[i]["size"])
            # transfer inference output data from device to host
            ret = acl.rt.memcpy(buffer_host,                    # host: destination address
                                self.output_data[i]["size"],    # host: destination address size
                                self.output_data[i]["buffer"],  # device: source address
                                self.output_data[i]["size"],    # device: source address size
                                ACL_MEMCPY_DEVICE_TO_HOST)      # mode: from device to host
            # from memory address get bytes object
            bytes_out = acl.util.ptr_to_bytes(buffer_host, self.output_data[i]["size"])
            # transfer bytes object to numpy array in float32 format
            data = np.frombuffer(bytes_out, dtype=np.float32)
            inference_result.append(data)
        vals = np.array(inference_result).flatten()

        return vals

    def __del__(self):
        # destructor function, release resources in reverse order of initialization
        # destroy input/output dataset
        for dataset in [self.input_data, self.output_data]:
            while dataset:
                item = dataset.pop()
                ret = acl.destroy_data_buffer(item["data"])    # destroy buffer data
                ret = acl.rt.free(item["buffer"])              # release buffer memory
        ret = acl.mdl.destroy_dataset(self.input_dataset)      # destroy input dataset
        ret = acl.mdl.destroy_dataset(self.output_dataset)     # destroy output dataset
        # destroy model description
        ret = acl.mdl.destroy_desc(self.model_desc)
        # unload model
        ret = acl.mdl.unload(self.model_id)
        # release device
        ret = acl.rt.reset_device(self.device_id)
        # acl finalize
        ret = acl.finalize()

def get_args_parser():
    parser = argparse.ArgumentParser(
        'cann evaluation and benchmark script', add_help=False)
    parser.add_argument('--extern-model', default=None, type=str, help='extern model name;resolution')
    parser.add_argument('--batch-size', default=1, type=int)
    # Model parameters
    parser.set_defaults(pretrained=True)
    parser.add_argument('--format', default='fp16', type=str)
    parser.add_argument('--only-test', default='', type=str, help='only test a certain model series')
    # Dataset parameters
    parser.add_argument('--validation', action='store_true', default=False)
    parser.add_argument('--data-path', default='imagenet-div50', type=str, help='dataset path')

    parser.add_argument('--profile', action='store_true', default=False)

    return parser


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    for name, resolution, usi_eval in [
        ('efficientformerv2_s0', 224, False),
        ('efficientformerv2_s1', 224, False),
        ('efficientformerv2_s2', 224, False),

        ('SwiftFormer_XS', 224, False),
        ('SwiftFormer_S' , 224, False),
        ('SwiftFormer_L1', 224, False),

        ('EMO_1M', 224, False),
        ('EMO_2M', 224, False),
        ('EMO_5M', 224, False),
        ('EMO_6M', 224, False),

        ('edgenext_xx_small', 256, False),
        ('edgenext_x_small' , 256, False),
        ('edgenext_small'   , 256, True),

        ('mobilevitv2_050', 256, False),
        ('mobilevitv2_075', 256, False),
        ('mobilevitv2_100', 256, False),
        # ('mobilevitv2_125', 256, False),
        # ('mobilevitv2_150', 256, False),
        # ('mobilevitv2_175', 256, False),
        # ('mobilevitv2_200', 256, False),

        ('mobilevit_xx_small', 256, False),
        ('mobilevit_x_small' , 256, False),
        ('mobilevit_small'   , 256, False),

        ('LeViT_128S', 224, False),
        ('LeViT_128' , 224, False),
        ('LeViT_192' , 224, False),
        # ('LeViT_256' , 224, False),

        ('resnet50', 224, False),
        ('mobilenetv3_large_100', 224, False),
        ('tf_efficientnetv2_b0' , 224, False),
        ('tf_efficientnetv2_b1' , 240, False),
        ('tf_efficientnetv2_b2' , 260, False),
        # ('tf_efficientnetv2_b3' , 300, False),
    ]:
        if args.only_test and args.only_test != name and not args.extern_model:
            continue

        if args.extern_model:
            name = args.extern_model.split(',')[0]
            resolution = int(args.extern_model.split(',')[1])

        args.model = name
        args.input_size = resolution
        args.usi_eval = usi_eval

        if not os.path.exists('.cann/'+args.format+'/'+args.model+'.om'):
            print(args.model + " model doesn't exist!!!")
            continue

        print(f"Creating cann om network: {name}")
        om_net = net('.cann/'+args.format+'/'+args.model+'.om')

        if args.validation:
            dataset_val = build_dataset(args)
            data_loader_val = torch.utils.data.DataLoader(
                dataset_val,
                batch_size=args.batch_size,
                shuffle=False
            )
            args.len_dataset_val = len(dataset_val)
            criterion = torch.nn.CrossEntropyLoss()
            metric_logger = MetricLogger(delimiter="  ")
            header = 'Test:'

            dataset_scale = 50000//args.len_dataset_val
            for images, target in metric_logger.log_every(data_loader_val, 50, header):
                batch_size = images.shape[0]
                non_blocking = batch_size > 1
                target = target * dataset_scale + (15 if dataset_scale == 50 else 0)

                images = images.numpy()

                result = om_net.forward(images)
                output = torch.from_numpy(result).unsqueeze(0)

                loss = criterion(output, target)

                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                metric_logger.update(loss=loss.item())
                metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
                metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

            # gather the stats from all processes
            metric_logger.synchronize_between_processes()
            print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
                .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
            print(output.mean().item(), output.std().item())

            test_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
            print(f"Accuracy on {args.len_dataset_val} test images: {test_stats['acc1']:.1f}%")

        else:
            images = load_image(args).numpy()
            # warmup
            start = time.perf_counter()
            while time.perf_counter() - start < WARMUP_SEC:
                result = om_net.forward(images)
            outputs = torch.from_numpy(result).unsqueeze(0)

            val, idx = outputs.topk(3)
            print(list(zip(idx[0].tolist(), val[0].tolist())))

            if args.profile:
                # https://www.hiascend.com/document/detail/zh/canncommercial/60RC1/inferapplicationdev/aclpythondevg/aclpythondevg_0128.html
                # 4.创建管道，用于读取以及写入模型订阅的数据。
                r, w = os.pipe()

                # 5.创建模型订阅的配置并且进行模型订阅。
                ACL_AICORE_NONE = 0xFF
                subscribe_config = acl.prof.create_subscribe_config(1, ACL_AICORE_NONE, w)
                # 模型订阅需要传入模型的model_id。
                ret = acl.prof.model_subscribe(om_net.model_id, subscribe_config)
                # 7.启动线程读取管道数据并解析。
                thr_id, ret = acl.util.start_thread(prof_data_read, [r, om_net.context])

                om_net.forward(images)

                # 11.取消订阅，释放订阅相关资源。
                acl.prof.model_un_subscribe(om_net.model_id)
                acl.util.stop_thread(thr_id)
                os.close(r)
                ret = acl.prof.destroy_subscribe_config(subscribe_config)
            else:
                time_list = []
                while sum(time_list) < TEST_SEC:
                    start = time.perf_counter()
                    om_net.forward(images)
                    time_list.append(time.perf_counter() - start)

                time_max = max(time_list) * 1000
                time_min = min(time_list) * 1000
                time_mean   = np.mean(time_list)   * 1000
                time_median = np.median(time_list) * 1000
                print("min = {:7.2f}ms  max = {:7.2f}ms  mean = {:7.2f}ms, median = {:7.2f}ms".format(time_min, time_max, time_mean, time_median))

        if args.extern_model: break
