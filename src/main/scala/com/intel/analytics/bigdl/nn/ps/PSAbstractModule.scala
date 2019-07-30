package com.intel.analytics.bigdl.nn.ps

import java.util.concurrent.Future

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.tencent.angel.ml.core.optimizer.Optimizer
import com.tencent.angel.ml.matrix.psf.update.base.VoidResult

import scala.reflect.ClassTag

abstract class PSTensorModule[T: ClassTag]
(implicit ev: TensorNumeric[T]) extends PSAbstractModule[Tensor[T], Tensor[T], T] {
  def pullParameters(input: Tensor[T]): Unit

  def pushGradient(): Unit
}

abstract class PSAbstractModule[A <: Activity : ClassTag, B <: Activity : ClassTag, T: ClassTag]
(implicit ev: TensorNumeric[T]) extends AbstractModule[A, B, T] {
  def pullParameters(input: Tensor[T]): Unit

  def pushGradient(): Unit

  def update(optimizer: Optimizer, epoch: Int, batchSize: Int): Future[VoidResult]
}