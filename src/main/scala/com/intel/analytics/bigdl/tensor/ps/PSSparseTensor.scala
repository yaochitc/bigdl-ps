package com.intel.analytics.bigdl.tensor.ps

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{SparseTensor, Storage, Tensor}

import scala.reflect.ClassTag

class PSSparseTensor[@specialized(Float, Double) T: ClassTag]
(_indices: Array[Storage[Int]],
 _values: Storage[T],
 _storageOffset: Int,
 _nElement: Int,
 _shape: Array[Int],
 _indicesOffset: Array[Int],
 nDimension: Int,
 matrixId: Int,
 clock: Int)
(implicit ev: TensorNumeric[T]) extends SparseTensor[T](_indices, _values, _storageOffset, _nElement, _shape, _indicesOffset, nDimension) {
  def this(tensor: SparseTensor[T], matrixId: Int, clock: Int)(implicit ev: TensorNumeric[T]) =
    this(tensor._indices,
      tensor._values,
      tensor._storageOffset,
      tensor._nElement,
      tensor._shape,
      tensor._indicesOffset,
      tensor.nDimension,
      matrixId,
      clock)
}

object PSSparseTensor {
  def apply[@specialized(Float, Double) T: ClassTag](tensor: Tensor[T], matrixId: Int, clock: Int)(
    implicit ev: TensorNumeric[T]): PSSparseTensor[T] = new PSSparseTensor[T](tensor.asInstanceOf[SparseTensor[T]], matrixId, clock)
}