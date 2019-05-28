package com.intel.analytics.bigdl.tensor.ps

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{ArrayStorage, DenseTensor, Tensor}

import scala.reflect.ClassTag

class PSDenseTensor[@specialized(Float, Double) T: ClassTag]
(_storage: ArrayStorage[T],
 _storageOffset: Int,
 _size: Array[Int],
 _stride: Array[Int],
 nDimension: Int,
 val matrixId: Int,
 val clock: Int)
(implicit ev: TensorNumeric[T]) extends DenseTensor(_storage, _storageOffset, _size, _stride, nDimension) {
  def this(tensor: DenseTensor[T], matrixId: Int, clock: Int)(implicit ev: TensorNumeric[T]) =
    this(tensor._storage,
      tensor._storageOffset,
      tensor._size,
      tensor._stride,
      tensor.nDimension,
      matrixId,
      clock)
}

object PSDenseTensor {
  def apply[@specialized(Float, Double) T: ClassTag](tensor: Tensor[T], matrixId: Int, clock: Int = 0)(
    implicit ev: TensorNumeric[T]): PSDenseTensor[T] = new PSDenseTensor[T](tensor.asInstanceOf[DenseTensor[T]], matrixId, clock)
}