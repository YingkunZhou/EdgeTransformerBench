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

package org.tensorflow.lite.examples.imageclassification.fragments

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.RecyclerView
import org.tensorflow.lite.examples.imageclassification.databinding.ItemClassificationResultBinding
import org.tensorflow.lite.support.label.Category
import kotlin.math.min

class ClassificationResultsAdapter :
    RecyclerView.Adapter<ClassificationResultsAdapter.ViewHolder>() {
    companion object {
        private const val NO_VALUE = "--"
    }

    private var categories: MutableList<Category?> = mutableListOf()
    private var adapterSize: Int = 0

    fun updateResults(sortedCategories: List<Category>?) {
        categories = MutableList(adapterSize) { null }
        sortedCategories?.let { it ->
            if (it.isNotEmpty()) {
                val min = min(sortedCategories.size, categories.size)
                for (i in 0 until min) {
                    categories[i] = sortedCategories[i]
                }
            }
        }
    }

    fun updateAdapterSize(size: Int) {
        adapterSize = size
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val binding = ItemClassificationResultBinding.inflate(
            LayoutInflater.from(parent.context),
            parent,
            false
        )
        return ViewHolder(binding)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        categories[position].let { category ->
            holder.bind(category?.label, category?.score)
        }
    }

    override fun getItemCount(): Int = categories.size

    inner class ViewHolder(private val binding: ItemClassificationResultBinding) :
        RecyclerView.ViewHolder(binding.root) {

        fun bind(label: String?, score: Float?) {
            with(binding) {
                tvLabel.text = label ?: NO_VALUE
                tvScore.text = if (score != null) String.format("%.2f", score) else NO_VALUE
            }
        }
    }
}
