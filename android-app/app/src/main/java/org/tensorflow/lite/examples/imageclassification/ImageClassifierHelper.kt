/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.imageclassification

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import android.view.Surface
import java.io.Closeable
import java.util.PriorityQueue
import kotlin.math.min
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.TensorProcessor
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import org.tensorflow.lite.support.label.TensorLabel
import org.tensorflow.lite.support.label.Category
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class ImageClassifierHelper(
    var threshold: Float = 0.5f,
    var numThreads: Int = 2,
    var maxResults: Int = 3,
    var currentDelegate: Int = 0,
    var currentModel: Int = 0,
    val context: Context,
    val imageClassifierListener: ClassifierListener?
) {

    private val labels by lazy { FileUtil.loadLabels(context, "imagenet-labels.txt") }

    // Processor to apply post processing of the output probability
    private val probabilityProcessor = TensorProcessor.Builder().build()

    private var interpreter: Interpreter? = null

    init {
        setupImageClassifier()
    }

    fun clearImageClassifier() {
        interpreter = null
    }

    private fun setupImageClassifier() {
        val interpreterOption = Interpreter.Options()
        interpreterOption.setNumThreads(numThreads)
        interpreterOption.setUseXNNPACK(true)

         if (currentDelegate == DELEGATE_GPU) {
            //https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/gpu/java/src/main/java/org/tensorflow/lite/gpu/GpuDelegateFactory.java
            val gpuOptions = GpuDelegate.Options()
            // use fp16
            gpuOptions.setPrecisionLossAllowed(true)
            interpreterOption.addDelegate(GpuDelegate(gpuOptions))
         }

        val modelName =
            when (currentModel) {
                MODEL_MOBILENETV3_L100 -> "mobilenetv3_large_100.tflite"
                MODEL_EFFICIENTNETV2_B0 -> "tf_efficientnetv2_b0.tflite"
                MODEL_EFFICIENTFORMERV2_S0 -> "efficientformerv2_s0.tflite"
                MODEL_SWIFTFORMER_XS -> "SwiftFormer_XS.tflite"
                MODEL_EMO_1M -> "EMO_1M.tflite"
                MODEL_LEVIT_128S -> "LeViT_128S.tflite"
                MODEL_EDGENEXT_XXS -> "edgenext_xx_small.tflite"
                MODEL_MOBILEVITV2_050 -> "mobilevitv2_050.tflite"
                MODEL_MOBILEVIT_XXS -> "mobilevit_xx_small.tflite"
                else -> "mobilenetv3_large_100.tflite"
            }

        try {
            interpreter = Interpreter(FileUtil.loadMappedFile(context, modelName), interpreterOption)
        } catch (e: IllegalStateException) {
            imageClassifierListener?.onError(
                "Image classifier failed to initialize. See error logs for details"
            )
            Log.e(TAG, "TFLite failed to load model with error: " + e.message)
        }
    }

    // Output probability TensorBuffer
    private val outputProbabilityBuffer: TensorBuffer by lazy {
        val probabilityTensorIndex = 0
        // {1, NUM_CLASSES}
        val probabilityShape = interpreter?.getOutputTensor(probabilityTensorIndex)?.shape()
        val probabilityDataType = interpreter?.getOutputTensor(probabilityTensorIndex)?.dataType()
        TensorBuffer.createFixedSize(probabilityShape, probabilityDataType)
    }

    fun classify(image: Bitmap, rotation: Int) {
        if (interpreter == null) {
            setupImageClassifier()
        }

        val aspectRatio = image.getWidth().toDouble() / image.getHeight().toDouble()
        val width: Int
        val height: Int
        val input_size: Int
        if (currentModel < MODEL_EDGENEXT_XXS) {
            input_size = 224
        } else {
            input_size = 256
        }
        val size = input_size + 32

        if (aspectRatio < 1) {
            width = size
            height = (size / aspectRatio).toInt()
        } else {
            height = size
            width = (size * aspectRatio).toInt()
        }
        // Create preprocessor for the image.
        // See https://www.tensorflow.org/lite/inference_with_metadata/
        //            lite_support#imageprocessor_architecture
        // https://ai.google.dev/edge/litert/android/metadata/lite_support?hl=zh-tw
        val mean = if (currentModel < MODEL_MOBILEVITV2_050) IMAGENET_DEFAULT_MEAN
                    else floatArrayOf(0f, 0f, 0f)
        val std = if (currentModel < MODEL_MOBILEVITV2_050) IMAGENET_DEFAULT_STD
                    else floatArrayOf(255f, 255f, 255f)
        val imageProcessor =
            ImageProcessor.Builder()
                // Resize using Bilinear or Nearest neighbour
                .add(ResizeOp(height, width, ResizeOp.ResizeMethod.BILINEAR))
                // Center crop the image to the largest square possible
                .add(ResizeWithCropOrPadOp(input_size, input_size))
                .add(Rot90Op(rotation+1))
                .add(NormalizeOp(mean, std))
                .build()
        // Preprocess the image and convert it into a TensorImage for classification.
        val tensorImage = imageProcessor.process(TensorImage.fromBitmap(image))
        // Get the underlying float array from the TensorImage
        val nhwcArray = tensorImage.getTensorBuffer().floatArray
        // Create a new FloatArray to hold the NCHW format image
        val channels = 3
        val nchwArray = FloatArray(channels * input_size * input_size)

        // Transpose to NCHW format (1 x channels x input_size x input_size)
        for (h in 0 until input_size) {
            for (w in 0 until input_size) {
                for (c in 0 until channels) {
                    val nhwcIndex = h * (input_size * channels) + w * channels + c
                    val nchwIndex = c * (input_size * input_size) + h * input_size + w
                    nchwArray[nchwIndex] = nhwcArray[nhwcIndex]
                }
            }
        }
        val nchwBuffer = TensorBuffer.createFixedSize(intArrayOf(channels, input_size, input_size), DataType.FLOAT32)
        nchwBuffer.loadArray(nchwArray)

        // Inference time is the difference between the system time at the start and finish of the
        // process
        var inferenceTime = SystemClock.uptimeMillis()

        // Runs the inference call
        interpreter?.run(nchwBuffer.buffer, outputProbabilityBuffer.buffer.rewind())

        inferenceTime = SystemClock.uptimeMillis() - inferenceTime

        // Gets the map of label and probability
        val labeledProbability =
        TensorLabel(labels, probabilityProcessor.process(outputProbabilityBuffer)).mapWithFloatValue

        imageClassifierListener?.onResults(
            getTopKProbability(labeledProbability),
            inferenceTime
        )
    }

    /** Gets the top-k results. */
    private fun getTopKProbability(labelProb: Map<String, Float>): List<Category> {
        val MAX_REPORT = 5
        // Sort the recognition by confidence from high to low.
        val pq: PriorityQueue<Category> =
        PriorityQueue(MAX_REPORT, compareByDescending<Category> { it.score })
        pq += labelProb.map { (label, prob) -> Category(label, prob) }
        return List(min(MAX_REPORT, pq.size)) { pq.poll()!! }
    }

    interface ClassifierListener {
        fun onError(error: String)
        fun onResults(
            results: List<Category>?,
            inferenceTime: Long
        )
    }

    companion object {
        const val DELEGATE_GPU = 0
        const val DELEGATE_CPU = 1
        const val MODEL_MOBILENETV3_L100 = 0
        const val MODEL_EFFICIENTNETV2_B0 = 1
        const val MODEL_EFFICIENTFORMERV2_S0 = 2
        const val MODEL_SWIFTFORMER_XS = 3
        const val MODEL_EMO_1M = 4
        const val MODEL_LEVIT_128S = 5
        const val MODEL_EDGENEXT_XXS = 6
        const val MODEL_MOBILEVITV2_050 = 7
        const val MODEL_MOBILEVIT_XXS = 8

        private const val TAG = "ImageClassifierHelper"
        private val IMAGENET_DEFAULT_MEAN = floatArrayOf(123.675f, 116.28f, 103.53f)
        private val IMAGENET_DEFAULT_STD = floatArrayOf(58.395f, 57.12f, 57.375f)

    }
}
