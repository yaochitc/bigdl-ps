package com.intel.analytics.bigdl.nn.ps.abstractnn

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.tencent.angel.ml.core.optimizer.Optimizer

import scala.reflect.ClassTag

trait ParameterSupport[T] {
  def init(): Unit

  def update(optimizer: Optimizer, epoch: Int, batchSize: Int): Unit
}

abstract class PSTensorModule[T: ClassTag]
(implicit ev: TensorNumeric[T]) extends PSAbstractModule[Tensor[T], Tensor[T], T]

abstract class PSAbstractModule[A <: Activity : ClassTag, B <: Activity : ClassTag, T: ClassTag]
(implicit ev: TensorNumeric[T]) extends AbstractModule[A, B, T] with ParameterSupport[T]