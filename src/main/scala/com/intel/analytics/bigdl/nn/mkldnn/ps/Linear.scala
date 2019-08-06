package com.intel.analytics.bigdl.nn.mkldnn.ps

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.ps.abstractnn.ParameterSupport
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.ps.PSTensorNumeric
import com.tencent.angel.ml.core.optimizer.Optimizer
import com.tencent.angel.ml.core.utils.PSMatrixUtils
import com.tencent.angel.ml.matrix.RowType

class Linear
(
  name: String,
  numSlot: Int,
  inputSize: Int,
  outputSize: Int,
  initWeight: Tensor[Float] = null,
  initBias: Tensor[Float] = null,
  initGradWeight: Tensor[Float] = null,
  initGradBias: Tensor[Float] = null
)(implicit psEv: PSTensorNumeric[Float])
  extends com.intel.analytics.bigdl.nn.mkldnn.Linear(inputSize, outputSize, null, null, initWeight, initBias, initGradWeight, initGradBias)
    with ParameterSupport[Float] {
  private val weightCtx =
    PSMatrixUtils.createPSMatrixCtx(s"${name}_weight", 1 + numSlot, inputSize * outputSize, RowType.T_FLOAT_DENSE)
  PSMatrixUtils.createPSMatrix(weightCtx)

  private val biasCtx = PSMatrixUtils.createPSMatrixCtx(s"${name}_bias", 1 + numSlot, outputSize, RowType.T_FLOAT_DENSE)
  PSMatrixUtils.createPSMatrix(biasCtx)

  lazy val weightId: Int = PSMatrixUtils.getMatrixId(s"${name}_weight")
  lazy val biasId: Int = PSMatrixUtils.getMatrixId(s"${name}_bias")


  override def init(): Unit = {
    psEv.incrementRowByMatrix(weightId, 0, weight.dense)
    psEv.incrementRowByMatrix(biasId, 0, bias.dense)
  }

  override def updateOutput(input: Activity): Activity = {
    pullParameters()
    super.updateOutput(input)
  }

  override def accGradParameters(input: Activity, gradOutput: Activity): Unit = {
    super.accGradParameters(input, gradOutput)
    pushGradient()
  }

  def pullParameters(): Unit = {
    weight.dense.copy(psEv.getRowAsMatrix(weightId, 0, outputSize, inputSize))
    bias.dense.copy(psEv.getRow(biasId, 0))
  }

  def pushGradient(): Unit = {
    psEv.incrementRowByMatrix(weightId, numSlot, gradWeight.dense)
    gradWeight.dense.zero()

    psEv.incrementRow(biasId, numSlot, gradBias.dense)
    gradWeight.dense.zero()
  }

  override def update(optimizer: Optimizer, epoch: Int, batchSize: Int): Unit = {
    optimizer.update(weightId, 1, epoch, 1).get()
    optimizer.update(biasId, 1, epoch, 1).get()
  }
}

object Linear {
  def apply
  (
    name: String,
    numSlot: Int,
    inputSize: Int,
    outputSize: Int,
    withBias: Boolean = true,
    initWeight: Tensor[Float] = null,
    initBias: Tensor[Float] = null,
    initGradWeight: Tensor[Float] = null,
    initGradBias: Tensor[Float] = null): Linear = {
    new Linear(name, numSlot, inputSize, outputSize, initWeight, initBias, initGradWeight, initGradBias)
  }
}