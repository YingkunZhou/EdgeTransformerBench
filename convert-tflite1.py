import tensorflow as tf
# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model("mobilevit_xx_small.pb") # path to the SavedModel directory
converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS
]
tf_lite_model = converter.convert()
# Save the model.
open('mobilevit_xx_small.tflite', 'wb').write(tf_lite_model)