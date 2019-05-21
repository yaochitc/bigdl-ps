package com.intel.analytics.bigdl.nn.ps

import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

abstract class PSTensorModule[T: ClassTag]
(implicit ev: TensorNumeric[T]) extends TensorModule[T] {
  def pullParameters(): Unit

  def pushGradient(): Unit
}