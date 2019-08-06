package com.intel.analytics.bigdl.nn.mkldnn.ps

import com.intel.analytics.bigdl.nn.ps.abstractnn.ParameterSupport
import com.tencent.angel.ml.core.optimizer.Optimizer

class Sequential extends com.intel.analytics.bigdl.nn.mkldnn.Sequential with ParameterSupport[Float] {
  override def init(): Unit = {
    var i = 0
    while (i < modules.length) {
      val module = modules(i)
      module match {
        case m: ParameterSupport[Float] => m.init()
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
        case m: ParameterSupport[Float] => m.update(optimizer, epoch, batchSize)
        case _ =>
      }

      i += 1
    }
  }
}

object Sequential {
  def apply(): Sequential = new Sequential()
}