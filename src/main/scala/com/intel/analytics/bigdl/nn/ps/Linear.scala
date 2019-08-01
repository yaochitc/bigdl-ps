package com.intel.analytics.bigdl.nn.ps

import com.intel.analytics.bigdl.nn.abstractnn.Initializable
import com.intel.analytics.bigdl.nn.ps.abstractnn.ParameterSupport
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.ps.{PSTensorNumeric, PSUtils}
import com.tencent.angel.ml.core.optimizer.Optimizer
import com.tencent.angel.ml.core.utils.PSMatrixUtils

import scala.reflect.ClassTag

class Linear[T: ClassTag]
(name: String,
 inputSize: Int,
 outputSize: Int,
 withBias: Boolean = true,
 initWeight: Tensor[T] = null,
 initBias: Tensor[T] = null,
 initGradWeight: Tensor[T] = null,
 initGradBias: Tensor[T] = null
)(implicit ev: TensorNumeric[T], psEv: PSTensorNumeric[T])
  extends com.intel.analytics.bigdl.nn.Linear[T](inputSize, outputSize, withBias, null, null, initWeight, initBias, initGradWeight, initGradBias)
    with ParameterSupport[T] with Initializable {
  private val weightCtx =
    PSMatrixUtils.createPSMatrixCtx(s"${name}_weight", 2, inputSize * outputSize, PSUtils.getRowType(ev.getType()))
  PSMatrixUtils.createPSMatrix(weightCtx)

  private val biasCtx = if (withBias) {
    val biasCtx = PSMatrixUtils.createPSMatrixCtx(s"${name}_bias", 2, outputSize, PSUtils.getRowType(ev.getType()))
    PSMatrixUtils.createPSMatrix(biasCtx)
    biasCtx
  } else null

  lazy val weightId: Int = PSMatrixUtils.getMatrixId(s"${name}_weight")
  lazy val biasId: Int = PSMatrixUtils.getMatrixId(s"${name}_bias")

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    pullParameters(input)
    super.updateOutput(input)
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    super.accGradParameters(input, gradOutput)
    pushGradient()
  }

  def pullParameters(input: Tensor[T]): Unit = {
    weight.copy(psEv.getRowAsMatrix(weightId, 0, outputSize, inputSize))
    if (withBias) {
      bias.copy(psEv.getRow(biasId, 0))
    }
  }

  def pushGradient(): Unit = {
    psEv.incrementRowByMatrix(weightId, 1, gradWeight)
    if (withBias) {
      psEv.incrementRow(biasId, 1, gradBias)
    }
  }

  override def update(optimizer: Optimizer, epoch: Int, batchSize: Int): Unit = {
    optimizer.update(weightId, 1, epoch, batchSize)
    optimizer.update(biasId, 1, epoch, batchSize)
  }
}

object Linear {
  def apply[@specialized(Float, Double) T: ClassTag]
  (
    name: String,
    inputSize: Int,
    outputSize: Int,
    withBias: Boolean = true,
    initWeight: Tensor[T] = null,
    initBias: Tensor[T] = null,
    initGradWeight: Tensor[T] = null,
    initGradBias: Tensor[T] = null
  )(implicit ev: TensorNumeric[T], psEv: PSTensorNumeric[T]): Linear[T] = {
    new Linear[T](name, inputSize, outputSize,
      withBias, initWeight, initBias, initGradWeight, initGradBias)
  }
}
