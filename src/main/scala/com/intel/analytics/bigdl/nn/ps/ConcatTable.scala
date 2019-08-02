package com.intel.analytics.bigdl.nn.ps

import com.intel.analytics.bigdl.nn.ps.abstractnn.ParameterSupport
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.tencent.angel.ml.core.optimizer.Optimizer

import scala.reflect.ClassTag

class ConcatTable[T: ClassTag]
(implicit ev: TensorNumeric[T]) extends com.intel.analytics.bigdl.nn.ConcatTable with ParameterSupport[T] {
  override def init(): Unit = {
    var i = 0
    while (i < modules.length) {
      val module = modules(i)
      module match {
        case m: ParameterSupport[T] => m.init()
        case _ =>
      }

      i += 1
    }
  }

  override def update(optimizer: Optimizer, epoch: Int, batchSize: Int): Unit = {
    var i = 0
    while (i < modules.length) {
      val module = modules(i)
      module match {
        case m: ParameterSupport[T] => m.update(optimizer, epoch, batchSize)
        case _ =>
      }

      i += 1
    }
  }
}

object ConcatTable {
  def apply[@specialized(Float, Double) T: ClassTag]()(implicit ev: TensorNumeric[T]): ConcatTable[T] = {
    new ConcatTable[T]()
  }
}
