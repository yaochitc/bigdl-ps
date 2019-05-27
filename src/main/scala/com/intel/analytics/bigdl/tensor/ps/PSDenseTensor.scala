package com.intel.analytics.bigdl.tensor.ps

import com.intel.analytics.bigdl.tensor.{ArrayStorage, DenseTensor, Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class PSDenseTensor[@specialized(Float, Double) T: ClassTag]
(_storage: ArrayStorage[T],
 _storageOffset: Int,
 _size: Array[Int],
 _stride: Array[Int],
 nDimension: Int,
 clock: Int = 0)
(implicit ev: TensorNumeric[T]) extends DenseTensor(_storage, _storageOffset, _size, _stride, nDimension) {

}
