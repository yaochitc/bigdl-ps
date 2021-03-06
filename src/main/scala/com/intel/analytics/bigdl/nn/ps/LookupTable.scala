package com.intel.analytics.bigdl.nn.ps

import breeze.numerics.{abs, pow}
import com.intel.analytics.bigdl.nn.ErrorInfo
import com.intel.analytics.bigdl.nn.abstractnn.Initializable
import com.intel.analytics.bigdl.nn.ps.abstractnn.PSTensorModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.ps.PSSparseRowTensor
import com.intel.analytics.bigdl.utils.ps.{PSTensorNumeric, PSUtils}
import com.tencent.angel.ml.core.optimizer.Optimizer
import com.tencent.angel.ml.core.utils.PSMatrixUtils
import com.tencent.angel.ml.math2.VFactory
import com.tencent.angel.ml.math2.vector.Vector
import com.tencent.angel.ml.matrix.psf.update.RandomNormal
import com.tencent.angel.ml.psf.columns.{UpdateColsFunc, UpdateColsParam}
import com.tencent.angel.psagent.PSAgentContext

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.reflect.ClassTag

class LookupTable[T: ClassTag]
(val name: String, val numSlot: Int,
 val nIndex: Int, val nOutput: Int,
 val paddingValue: Double = 0,
 val maxNorm: Double = Double.MaxValue,
 val normType: Double = 2.0,
 shouldScaleGradByFreq: Boolean = false,
 val maskZero: Boolean = false)
(implicit ev: TensorNumeric[T], psEv: PSTensorNumeric[T]) extends PSTensorModule[T] with Initializable {

  private val embedMatCtx = PSMatrixUtils.createPSMatrixCtx(s"${name}_embedding", (1 + numSlot) * nOutput, nIndex,
    PSUtils.getRowType(ev.getType()))
  PSMatrixUtils.createPSMatrix(embedMatCtx)

  lazy val matrixId: Int = PSMatrixUtils.getMatrixId(s"${name}_embedding")

  @transient var weight: PSSparseRowTensor[T] = _
  @transient var gradWeight: PSSparseRowTensor[T] = _

  private var inputBuffer = Tensor[T]()
  private var normBuffer = Tensor[T]()
  private val countBuffer = Tensor[T]()

  override def init(): Unit = {
    val bound: Double = 1.0
    val randFunc = new RandomNormal(matrixId, 0, nOutput, 0.0, bound)
    PSAgentContext.get().getUserRequestAdapter.update(randFunc).get()
  }

  private def renorm(input: Tensor[T]): Unit = {
    if (Double.MaxValue == maxNorm) {
      return
    }
    normBuffer.resize(input.size()).copy(input)
    if (normBuffer.dim() == 2) {
      normBuffer = normBuffer.view(normBuffer.nElement())
    }
    require(weight.isContiguous(), "LookupTable: weight must be contiguous")
    require(normBuffer.isContiguous(), "LookupTable: input must be contiguous")
    require(normBuffer.nDimension() == 1, "LookupTable: idx must be a vector")
    require(normType > 0, "LookupTable: non-positive-norm not supported")

    val rowIdx = normBuffer.storage().array()
    val rowOffset = normBuffer.storageOffset() - 1
    var numEle = normBuffer.nElement()
    val stride = weight.stride(1)

    val gw = weight.storage().array()
    val gw_offset = weight.storageOffset() - 1

    var i = 0
    while (i < numEle) {
      require(ev.isGreater(ev.fromType(weight.size(1) + 1), rowIdx(i + rowOffset)),
        s"LookupTable: elements of input should be little than or equal to $nIndex + 1")
      require(ev.isGreaterEq(rowIdx(i + rowOffset), ev.one),
        "LookupTable: elements of input should be greater than or equal to 1")
      i += 1
    }

    implicit val ord = Ordering.fromLessThan[T]((e1, e2) => (ev.isGreater(e1, e2)))
    scala.util.Sorting.quickSort(rowIdx)

    var ptr = 0
    i = 0
    while (i < numEle) {
      if (i == 0 || rowIdx(i + rowOffset) != rowIdx(i - 1 + rowOffset)) {
        rowIdx(ptr + rowOffset) = rowIdx(i + rowOffset)
        ptr += 1
      }
      i += 1
    }
    numEle = ptr

    i = 0
    while (i < numEle) {
      val k = ev.toType[Int](rowIdx(i + rowOffset)) - 1
      renormRow(gw, k * stride + gw_offset, stride, maxNorm, normType)
      i += 1
    }
  }

  private def renormRow(row_data: Array[T], offset: Int, stride: Int,
                        maxNorm: Double, normType: Double): Unit = {
    var norm = 0.0
    var j = 0
    while (j < stride) {
      if (normType == 1) {
        norm += ev.toType[Double](ev.abs(row_data(j + offset)))
      } else if (normType == 2) {
        norm += ev.toType[Double](ev.times(row_data(j + offset), row_data(j + offset)))
      } else {
        norm += math.pow(abs(ev.toType[Double](row_data(j + offset))), normType)
      }
      j += 1
    }
    norm = pow(norm, 1.0 / normType)

    // Keep the norm of weight smaller than maxNorm
    if (norm > maxNorm) {
      val new_norm = maxNorm / (norm + 1e-7)
      j = 0
      while (j < stride) {
        row_data(j + offset) = ev.times(row_data(j + offset), ev.fromType(new_norm))
        j += 1
      }
    }
  }

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
    pullParameters(input)

    if (maskZero && paddingValue != 0) {
      weight.select(1, paddingValue.toInt).zero()
    }
    require(input.dim() == 1 || input.dim() == 2,
      s"LookupTable: ${ErrorInfo.constrainInputAsVectorOrBatch}, input dim [${input.dim()}]")
    renorm(input)
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
    inputBuffer = input.contiguous()
    require(inputBuffer.dim() == 1 || inputBuffer.dim() == 2,
      s"LookupTable: input must be a vector or matrix, input dim ${inputBuffer.dim()}")

    if (inputBuffer.dim() == 2) {
      inputBuffer.view(inputBuffer.nElement())
    }
    val _gradOutput = gradOutput.contiguous()
    var count_data: Array[T] = null
    if (shouldScaleGradByFreq) {
      countBuffer.resize(gradWeight.size(1))
      resetCount(countBuffer, inputBuffer)
      count_data = countBuffer.storage().array()
    }

    val input_data = inputBuffer.storage().array()
    val input_offset = inputBuffer.storageOffset() - 1
    val numEle = inputBuffer.nElement()

    var i = 0
    while (i < numEle) {
      require(ev.isGreaterEq(input_data(i + input_offset), ev.one),
        "LookupTable: elements of input should be greater than or equal to 1")
      i += 1
    }
    if (scaleW != 0) {
      val go = _gradOutput.storage().array()
      val stride = nOutput


      val map = mutable.Map[Long, Vector]()
      (0 until numEle).filter(i => input_data(i + input_offset) != paddingValue)
        .foreach(i => {
          val k = ev.toType[Int](input_data(i + input_offset)) - 1
          val scale_ = if (null != count_data) scaleW /
            ev.toType[Double](count_data(k)) else scaleW

          val index = k.toLong
          val gradWeightFrame = map.getOrElse(index, psEv.zeroVector(nOutput))
          val gradOutputFrame = psEv.array2Vector(go, i * stride + _gradOutput.storageOffset() - 1, stride).imul(scale_)

          map(index) = gradWeightFrame.add(gradOutputFrame)
        })

      gradWeight = PSSparseRowTensor(map.toMap, nIndex, nOutput)

      pushGradient()
    }
  }

  override def clearState(): this.type = {
    super.clearState()
    inputBuffer.set()
    countBuffer.set()
    normBuffer.set()
    this
  }

  def pullParameters(input: Tensor[T]): Unit = {
    inputBuffer = input.contiguous()
    require(inputBuffer.dim() == 1 || inputBuffer.dim() == 2,
      s"LookupTable: input must be a vector or matrix, input dim ${inputBuffer.dim()}")

    if (inputBuffer.dim() == 2) {
      inputBuffer.view(inputBuffer.nElement())
    }

    val input_data = inputBuffer.storage().array()
    val input_offset = inputBuffer.storageOffset() - 1
    val numEle = inputBuffer.nElement()

    var i = 0
    while (i < numEle) {
      require(ev.isGreaterEq(input_data(i + input_offset), ev.one),
        "LookupTable: elements of input should be greater than or equal to 1")
      i += 1
    }

    val rows = (0 until nOutput).toArray
    val indices = (0 until numEle).map(i => ev.toType[Long](input_data(i + input_offset)) - 1)
      .distinct
      .toArray
      .sorted

    weight = psEv.getRowAsSparseMatrix(matrixId, nIndex, rows, indices)
  }

  def pushGradient(): Unit = {
    val rowNums = (nOutput * numSlot until nOutput * (1 + numSlot)).toArray
    val indices = VFactory.denseLongVector(gradWeight.getVectors.keys.toArray.sorted)
    val vectors = gradWeight.getVectors.map { case (key, vector) => (long2Long(key), vector) }.asJava

    val param = new UpdateColsParam(matrixId, rowNums, indices, vectors)
    val func = new UpdateColsFunc(param)
    PSAgentContext.get().getUserRequestAdapter.update(func).get()

    gradWeight = null
  }

  override def update(optimizer: Optimizer, epoch: Int, batchSize: Int): Unit = {
    optimizer.update(matrixId, nOutput, epoch, 1).get()
  }
}


object LookupTable {
  def apply[@specialized(Float, Double) T: ClassTag]
  (
    name: String,
    numSlot: Int,
    nIndex: Int,
    nOutput: Int,
    paddingValue: Double = 0,
    maxNorm: Double = Double.MaxValue,
    normType: Double = 2.0,
    shouldScaleGradByFreq: Boolean = false,
    maskZero: Boolean = false
  )
  (implicit ev: TensorNumeric[T], psEv: PSTensorNumeric[T]): LookupTable[T] =
    new LookupTable[T](name, numSlot, nIndex, nOutput, paddingValue,
      maxNorm, normType, shouldScaleGradByFreq, maskZero)
}
