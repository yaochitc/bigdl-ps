package com.intel.analytics.bigdl.nn.ps

import java.util.concurrent.Future

import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.tencent.angel.ml.matrix.psf.update.base.VoidResult

import scala.reflect.ClassTag

abstract class PSTensorModule[T: ClassTag]
(implicit ev: TensorNumeric[T]) extends TensorModule[T] {
  def pullParameters(): Unit

  def pushGradient(): Unit

  def update(epoch: Int, batchSize: Int): Future[VoidResult]
}