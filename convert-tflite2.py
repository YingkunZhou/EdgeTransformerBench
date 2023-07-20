import tensorflow as tf
# make a converter object from the saved tensorflow file
converter = tf.lite.TFLiteConverter.from_frozen_graph('mobilevit_xx_small.pb', #TensorFlow freezegraph .pb model file
                                                      # input_arrays=['main_input'], # name of input arrays as defined in torch.onnx.export function before.
                                                      # output_arrays=['main_output']  # name of output arrays defined in torch.onnx.export function before.
                                                      )
# tell converter which type of optimization techniques to use
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# to view the best option for optimization read documentation of tflite about optimization
# go to this link https://www.tensorflow.org/lite/guide/get_started#4_optimize_your_model_optional
# convert the model
tf_lite_model = converter.convert()
# save the converted model
open('mobilevit_xx_small.tflite', 'wb').write(tf_lite_model)