package com.intel.analytics.bigdl.nn.ps

import com.intel.analytics.bigdl.nn.ps.abstractnn.ParameterSupport
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.ps.{PSTensorNumeric, PSUtils}
import com.tencent.angel.ml.core.optimizer.Optimizer
import com.tencent.angel.ml.core.utils.PSMatrixUtils

import scala.reflect.ClassTag

class Linear[T: ClassTag]
(name: String,
 numSlot: Int,
 inputSize: Int,
 outputSize: Int,
 withBias: Boolean = true,
 initWeight: Tensor[T] = null,
 initBias: Tensor[T] = null,
 initGradWeight: Tensor[T] = null,
 initGradBias: Tensor[T] = null
)(implicit ev: TensorNumeric[T], psEv: PSTensorNumeric[T])
  extends com.intel.analytics.bigdl.nn.Linear[T](inputSize, outputSize, withBias, null, null, initWeight, initBias, initGradWeight, initGradBias)
    with ParameterSupport[T] {
  private val weightCtx =
    PSMatrixUtils.createPSMatrixCtx(s"${name}_weight", 1 + numSlot, inputSize * outputSize, PSUtils.getRowType(ev.getType()))
  PSMatrixUtils.createPSMatrix(weightCtx)

  private val biasCtx = if (withBias) {
    val biasCtx = PSMatrixUtils.createPSMatrixCtx(s"${name}_bias", 1 + numSlot, outputSize, PSUtils.getRowType(ev.getType()))
    PSMatrixUtils.createPSMatrix(biasCtx)
    biasCtx
  } else null

  lazy val weightId: Int = PSMatrixUtils.getMatrixId(s"${name}_weight")
  lazy val biasId: Int = PSMatrixUtils.getMatrixId(s"${name}_bias")

  override def init(): Unit = {
    psEv.incrementRowByMatrix(weightId, 0, weight)
    if (withBias) {
      psEv.incrementRowByMatrix(biasId, 0, bias)
    }
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    pullParameters()
    super.updateOutput(input)
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    super.accGradParameters(input, gradOutput)
    pushGradient()
  }

  def pullParameters(): Unit = {
    weight.copy(psEv.getRowAsMatrix(weightId, 0, outputSize, inputSize))
    if (withBias) {
      bias.copy(psEv.getRow(biasId, 0))
    }
  }

  def pushGradient(): Unit = {
    psEv.incrementRowByMatrix(weightId, numSlot, gradWeight)
    gradWeight.zero()

    if (withBias) {
      psEv.incrementRow(biasId, numSlot, gradBias)
      gradBias.zero()
    }
  }

  override def update(optimizer: Optimizer, epoch: Int, batchSize: Int): Unit = {
    optimizer.update(weightId, 1, epoch, 1).get()

    if (withBias) {
      optimizer.update(biasId, 1, epoch, 1).get()
    }
  }
}

object Linear {
  def apply[@specialized(Float, Double) T: ClassTag]
  (
    name: String,
    numSlot: Int,
    inputSize: Int,
    outputSize: Int,
    withBias: Boolean = true,
    initWeight: Tensor[T] = null,
    initBias: Tensor[T] = null,
    initGradWeight: Tensor[T] = null,
    initGradBias: Tensor[T] = null
  )(implicit ev: TensorNumeric[T], psEv: PSTensorNumeric[T]): Linear[T] = {
    new Linear[T](name, numSlot, inputSize, outputSize,
      withBias, initWeight, initBias, initGradWeight, initGradBias)
  }
}
