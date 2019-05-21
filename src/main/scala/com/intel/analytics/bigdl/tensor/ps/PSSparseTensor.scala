package com.intel.analytics.bigdl.tensor.ps

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{SparseTensor, Storage}

import scala.reflect.ClassTag

class PSSparseTensor[@specialized(Float, Double) T: ClassTag]
(_indices: Array[Storage[Int]],
 _values: Storage[T],
 _storageOffset: Int,
 _nElement: Int,
 _shape: Array[Int],
 _indicesOffset: Array[Int],
 nDimension: Int)
(implicit ev: TensorNumeric[T]) extends SparseTensor[T](_indices, _values, _storageOffset, _nElement, _shape, _indicesOffset, nDimension) {

}
