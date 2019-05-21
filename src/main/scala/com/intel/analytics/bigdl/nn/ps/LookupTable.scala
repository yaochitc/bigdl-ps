package com.intel.analytics.bigdl.nn.ps

import java.util.concurrent.Future

import com.intel.analytics.bigdl.nn.ErrorInfo
import com.intel.analytics.bigdl.nn.abstractnn.Initializable
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.ps.{PSTensorNumeric, PSUtils}
import com.tencent.angel.ml.core.utils.PSMatrixUtils
import com.tencent.angel.ml.matrix.psf.update.base.VoidResult

import scala.reflect.ClassTag

class LookupTable[T: ClassTag]
(name: String, val nIndex: Int, val nOutput: Int)
(implicit ev: TensorNumeric[T], psEv: PSTensorNumeric[T]) extends PSTensorModule[T] with Initializable {

  private val embedMatCtx = PSMatrixUtils.createPSMatrixCtx(s"${name}_embedding", 2 * nOutput, nIndex,
    PSUtils.getRowType(ev.getType()))

  lazy val matrixId: Int = PSMatrixUtils.getMatrixId(s"${name}_embedding")

  private var inputBuffer = Tensor[T]()

  @transient var weight: Tensor[T] = _
  @transient var gradWeight: Tensor[T] = _

  private def resetCount(count: Tensor[T], input: Tensor[T]): Unit = {
    var i = 1
    val numEle = input.nElement()

    while (i <= numEle) {
      val k = ev.toType[Int](input.valueAt(i))
      count.update(k, ev.zero)
      i += 1
    }

    i = 1
    while (i <= numEle) {
      val k = ev.toType[Int](input.valueAt(i))
      count.update(k, ev.plus(count.valueAt(k), ev.one))
      i += 1
    }
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim() == 1 || input.dim() == 2,
      s"LookupTable: ${ErrorInfo.constrainInputAsVectorOrBatch}, input dim [${input.dim()}]")

    inputBuffer = input.contiguous()
    try {
      if (inputBuffer.dim() == 1) {
        output.index(1, inputBuffer, weight)
      } else if (inputBuffer.dim() == 2) {
        output.index(1, inputBuffer.view(inputBuffer.nElement()), weight)
        output = output.view(inputBuffer.size(1), inputBuffer.size(2), weight.size(2))
      }
    } catch {
      case e: IllegalArgumentException =>
        throw new IllegalArgumentException(
          s"LookupTable updateOutput get exception:${e.getMessage}\n" +
            s"please ensure elements of your input will not exceed ${nIndex}")
      case e: Exception =>
        throw e
    }

    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    if (!gradInput.isSameSizeAs(input)) {
      gradInput.resizeAs(input).zero()
    }
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {

  }

  override def pullParameters(): Unit = {
  }

  override def pushGradient(): Unit = {

  }

  override def update(epoch: Int, batchSize: Int): Future[VoidResult] = ???
}
