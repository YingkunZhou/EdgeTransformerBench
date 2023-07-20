import tflite_runtime.interpreter as tflite
from speed_test import load_image
import argparse
import time

def get_args_parser():
    parser = argparse.ArgumentParser(
        'EdgeTransformerPerf tflite_runtime evaluation and benchmark script', add_help=False)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.set_defaults(IOBinding=True)
    # Dataset parameters
    parser.add_argument('--validation', action='store_true', default=False)
    parser.add_argument('--data-path', default='imagenet-div50', type=str, help='dataset path')
    parser.add_argument('--num_workers', default=2, type=int)
    # Benchmark parameters
    parser.set_defaults(cpu=True)
    parser.add_argument('--no-cpu', action='store_false', dest='cpu')
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--only-test', default='', type=str, help='only test a certain model series')

    return parser

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    args.usi_eval = False
    args.input_size = 256
    args.model = 'mobilevit_xx_small'
    # Load the TFLite model and allocate tensors
    interpreter = tflite.Interpreter(model_path=args.model+".tflite", num_threads=1)
    interpreter.allocate_tensors()
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Test the model on random input data
    interpreter.set_tensor(input_details[0]['index'], load_image(args))

    warmup_iterations = 10
    test_iterations = 100

    for i in range(warmup_iterations):
        interpreter.invoke()

    time_min = 1e5
    time_avg = 0
    time_max = 0

    for i in range(test_iterations):
        start_time = time.perf_counter()

        interpreter.invoke()

        end_time = time.perf_counter()
        exec_time = (end_time - start_time) * 1000.0
        time_min = exec_time if exec_time < time_min else time_min
        time_max = exec_time if exec_time > time_max else time_max
        time_avg += exec_time

    time_avg /= test_iterations
    print("min = {:7.2f}  max = {:7.2f}  avg = {:7.2f}".format(time_min, time_max, time_avg))

    # get_tensor() returns a copy of the tensor data
    # use tensor() in order to get a pointer to the tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
