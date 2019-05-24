package com.intel.analytics.bigdl.nn.ps

import java.util.concurrent.Future

import com.intel.analytics.bigdl.nn.ErrorInfo
import com.intel.analytics.bigdl.nn.abstractnn.{Initializable, TensorModule}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.ps.{PSTensorNumeric, PSUtils}
import com.tencent.angel.ml.core.utils.PSMatrixUtils
import com.tencent.angel.ml.matrix.psf.update.base.VoidResult

import scala.reflect.ClassTag

class Linear[T: ClassTag]
(name: String,
 val inputSize: Int,
 val outputSize: Int,
 val withBias: Boolean = true,
 var wRegularizer: Regularizer[T] = null,
 var bRegularizer: Regularizer[T] = null,
 private val initGradWeight: Tensor[T] = null,
 private val initGradBias: Tensor[T] = null
)(implicit ev: TensorNumeric[T], psEv: PSTensorNumeric[T]) extends PSTensorModule[T] with Initializable {
  val weightCtx =
    PSMatrixUtils.createPSMatrixCtx(s"${name}_weight", 2, inputSize * outputSize, PSUtils.getRowType(ev.getType()))
  PSMatrixUtils.createPSMatrix(weightCtx)

  val biasCtx = if (withBias) {
    val biasCtx = PSMatrixUtils.createPSMatrixCtx(s"${name}_bias", 2, outputSize, PSUtils.getRowType(ev.getType()))
    PSMatrixUtils.createPSMatrix(biasCtx)
    biasCtx
  } else null

  lazy val weightId: Int = PSMatrixUtils.getMatrixId(s"${name}_weight")
  lazy val biasId: Int = PSMatrixUtils.getMatrixId(s"${name}_bias")

  @transient var weight: Tensor[T] = _
  @transient var bias: Tensor[T] = _

  val addBuffer: Tensor[T] = Tensor[T]()

  val gradWeight: Tensor[T] =
    if (initGradWeight != null) initGradWeight else Tensor[T]()
  val gradBias: Tensor[T] =
    if (initGradBias != null) initGradBias else if (withBias) Tensor[T]() else null

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim() == 1 || input.dim() == 2,
      "Linear: " + ErrorInfo.constrainInputAsVectorOrBatch +
        s"input dim ${input.dim()}")


    if (input.dim() == 1) {
      output.resize(Array(outputSize))
      if (withBias) output.copy(bias) else output.zero()
      output.addmv(ev.fromType[Int](1), weight, input)
    }
    else if (input.dim() == 2) {
      val nFrame = input.size(1)
      val nElement = output.nElement
      val t = Array(nFrame, weight.size(1))
      output.resize(t)
      if (output.nElement() != nElement) {
        output.zero()
      }

      if (addBuffer.nElement() != nFrame) {
        addBuffer.resize(Array(nFrame)).fill(ev.one)
      }

      output.addmm(ev.zero, output, ev.one, input, weight.t)
      if (withBias) output.addr(ev.one, addBuffer, bias)
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    require(input.dim() == 1 || input.dim() == 2,
      "Linear: " + ErrorInfo.constrainInputAsVectorOrBatch +
        s"input dim ${input.dim()}")

    val nElement = gradInput.nElement()
    gradInput.resizeAs(input)
    if (nElement != gradInput.nElement()) {
      gradInput.zero()
    }

    if (input.dim() == 1) {
      gradInput.addmv(ev.fromType[Int](0), ev.fromType[Int](1), weight.t(), gradOutput)
    } else if (input.dim() == 2) {
      gradInput.addmm(ev.fromType[Int](0), ev.fromType[Int](1), gradOutput, weight)
    }
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    require(input.dim() == 1 || input.dim() == 2,
      "Linear: " + ErrorInfo.constrainInputAsVectorOrBatch +
        s"input dim ${input.dim()}")

    gradWeight.resize(outputSize, inputSize)
    if (withBias) {
      gradBias.resize(outputSize)
    }

    if (input.dim() == 1) {
      if (scaleW != 0) {
        gradWeight.addr(ev.fromType[Double](scaleW), gradOutput, input)
      }

      if (withBias && scaleB != 0) {
        gradBias.add(ev.fromType[Double](scaleB), gradOutput)
      }
    }
    else if (input.dim() == 2) {
      if (scaleW != 0) {
        gradWeight.addmm(ev.fromType[Double](scaleW), gradOutput.t, input)
      }

      if (withBias && scaleB != 0) {
        gradBias.addmv(ev.fromType[Double](scaleB), gradOutput.t, addBuffer)
      }
    }

    if (null != wRegularizer && scaleW != 0) {
      wRegularizer.accRegularization(weight, gradWeight, scaleW)
    }
    if (null != bRegularizer && scaleB != 0) {
      bRegularizer.accRegularization(bias, gradBias, scaleB)
    }
  }

  override def clearState(): this.type = {
    super.clearState()
    addBuffer.set()
    this
  }

  override def pullParameters(): Unit = {
    weight = psEv.getRowAsMatrix(weightId, 0, inputSize, outputSize)
    if (withBias) {
      bias = psEv.getRow(biasId, 0)
    }
  }

  override def pushGradient(): Unit = {
    psEv.incrementRowByMatrix(weightId, 1, gradWeight)
    if (withBias) {
      psEv.incrementRow(biasId, 1, gradBias)
    }
  }

  override def update(epoch: Int, batchSize: Int): Future[VoidResult] = ???
}
