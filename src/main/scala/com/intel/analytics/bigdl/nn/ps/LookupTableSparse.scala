package com.intel.analytics.bigdl.nn.ps

import com.intel.analytics.bigdl.nn.abstractnn.{Activity, Initializable}
import com.intel.analytics.bigdl.nn.ps.abstractnn.PSAbstractModule
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.ps.PSSparseRowTensor
import com.intel.analytics.bigdl.tensor.{SparseType, Tensor}
import com.intel.analytics.bigdl.utils.ps.{PSTensorNumeric, PSUtils}
import com.tencent.angel.ml.core.optimizer.Optimizer
import com.tencent.angel.ml.core.utils.PSMatrixUtils
import com.tencent.angel.ml.math2.VFactory
import com.tencent.angel.ml.matrix.psf.update.RandomNormal
import com.tencent.angel.ml.psf.columns.{UpdateColsFunc, UpdateColsParam}
import com.tencent.angel.psagent.PSAgentContext

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.reflect.ClassTag

class LookupTableSparse[T: ClassTag]
(val name: String, val nIndex: Int, val nOutput: Int,
 val combiner: String = "sum",
 val maxNorm: Double = -1)
(implicit ev: TensorNumeric[T], psEv: PSTensorNumeric[T]) extends PSAbstractModule[Activity, Tensor[T], T] with Initializable {

  private val embedMatCtx = PSMatrixUtils.createPSMatrixCtx(s"${name}_embedding", 2 * nOutput, nIndex,
    PSUtils.getRowType(ev.getType()))
  PSMatrixUtils.createPSMatrix(embedMatCtx)

  lazy val matrixId: Int = PSMatrixUtils.getMatrixId(s"${name}_embedding")

  @transient var weight: PSSparseRowTensor[T] = _
  @transient var gradWeight: PSSparseRowTensor[T] = _

  protected val inputBuffer: Tensor[T] = Tensor()
  protected val inputWeightBuffer: Tensor[T] = Tensor()
  protected val frameBuffer: Tensor[T] = Tensor()
  protected val ids: Tensor[T] = Tensor()
  protected val indices: Tensor[Int] = Tensor[Int]()
  protected val batchScaleBuffer: Tensor[T] = Tensor[T]()
  protected var nonZeroCount: Array[Int] = _
  protected val normScale: mutable.HashMap[Int, T] = mutable.HashMap[Int, T]()

  override def init(): Unit = {
    val bound: Double = 1.0
    val randFunc = new RandomNormal(matrixId, 0, nOutput, 0.0, bound)
    PSAgentContext.get().getUserRequestAdapter.update(randFunc).get()
  }

  override def updateOutput(input: Activity): Tensor[T] = {
    val (inputTensor, weightTensor) = if (input.isTable) {
      (input.toTable[Tensor[T]](1), Some(input.toTable[Tensor[T]](2)))
    } else {
      (input.toTensor[T], None)
    }
    require(inputTensor.getTensorType == SparseType, "LookupTableSparse's input" +
      s"must be SparseTensor, but got ${inputTensor.getTensorType}")

    pullParameters(inputTensor)

    val batchSize = inputTensor.size(1)
    inputBuffer.set(inputTensor.storage(),
      inputTensor.storageOffset(),
      Array(inputTensor.nElement()))
    if (weightTensor.isDefined) {
      val weight = weightTensor.get
      inputWeightBuffer.set(weight.storage(),
        weight.storageOffset(),
        Array(weight.nElement()))
    }

    Tensor.unique(inputBuffer, ids, indices)

    if (maxNorm > 0) {
      normScale.clear()
      LookupTableSparse.norm2ScaleWithIndices[T](
        weight, ids, ev.fromType(maxNorm), normScale)
    }

    nonZeroCount = inputTensor.numNonZeroByRow()
    output.resize(batchSize, nOutput).zero()
    batchScaleBuffer.resize(batchSize)

    var i = 0 // index for all the ids in the input
    var b = 0
    while (b < batchSize) {
      val times = nonZeroCount(b)
      // compute a overall scale for this batch
      val batchScale = if (combiner == "sum") {
        // if combiner == sum, batchScale = 1
        ev.one
      } else {
        var count = times.toFloat
        if (weightTensor.isDefined) {
          count = 0
          var j = 0
          while (j < times) {
            if (combiner == "mean") {
              count += ev.toType[Float](inputWeightBuffer.valueAt(i + j + 1))
            } else {
              count += math.pow(ev.toType[Float](inputWeightBuffer.valueAt(i + j + 1)), 2).toFloat
            }
            j += 1
          }
        }
        if (combiner == "mean") {
          // if combiner == mean, batchScale = sum(inputWeightBuffer) / times
          ev.fromType(1f / count)
        } else {
          // if combiner == sqrtn, batchScale = sqrt(sum(inputWeightBuffer^2)) / times
          ev.fromType(1f / math.sqrt(count))
        }
      }
      // save this batchScale
      batchScaleBuffer.setValue(b + 1, batchScale)

      var j = 0
      while (j < times) {
        val index = ev.toType[Int](inputBuffer.valueAt(i + 1))
        // scale = normScale * batchScale * sp_weights
        val scale = ev.times(
          if (normScale != null && normScale.contains(index)) normScale(index) else ev.one,
          ev.times(batchScale,
            if (weightTensor.isDefined) inputWeightBuffer.valueAt(i + 1) else ev.one))
        // output += scale * weight(index)
        output.select(1, b + 1).add(scale, weight.select(1, index))
        i += 1
        j += 1
      }
      b += 1
    }

    output
  }

  override def updateGradInput(input: Activity, gradOutput: Tensor[T]): Activity = {
    // Input is not derivable
    gradInput
  }

  override def accGradParameters(input: Activity, gradOutput: Tensor[T]): Unit = {
    val batchSize = output.size(1)
    val three = ev.fromType(3)

    var b = 0
    var i = 0
    val map = mutable.Map[Int, Tensor[T]]()
    while (b < batchSize) {
      val times = nonZeroCount(b)
      var j = 0
      while (j < times) {
        val index = ev.toType[Int](inputBuffer.valueAt(i + 1))

        val gradOutputFrame = gradOutput.select(1, b + 1)
        // scale = normScale * batchScale * sp_weights
        val scale = ev.times(
          if (normScale != null) normScale.getOrElse(index, ev.one) else ev.one,
          ev.times(batchScaleBuffer.valueAt(b + 1),
            if (!inputWeightBuffer.isEmpty) inputWeightBuffer.valueAt(i + 1) else ev.one))
        // gradWeight += gradOutput * scale
        val gradWeightFrame = map.getOrElse(index, Tensor().resize(gradOutputFrame.size()).zero())
        gradWeightFrame.add(scale, gradOutputFrame)

        // if norm2 clipping is invoked, need to compute the clipping's gradient.
        if (normScale != null && normScale.contains(index)) {
          val weightFrame = weight.select(1, index)
          // sum = sum(weightFrame * gradOutputFrame) * maxNorm * sp_weights * batchScale
          val sum = ev.times(frameBuffer.resizeAs(weightFrame).copy(weightFrame)
            .cmul(gradOutputFrame).sum,
            ev.times(ev.fromType(maxNorm), ev.divide(scale, normScale(index))))
          // gradWeight += - (normScale / maxNorm)^3 * sum * gradOutput
          gradWeightFrame.add(ev.times(sum, ev.negative(
            ev.pow(ev.divide(normScale(index), ev.fromType(maxNorm)), three))),
            weight.select(1, index))
        }

        map(index) = gradWeightFrame
        i += 1
        j += 1
      }
      b += 1
    }

    val vectors = map.map { case (index, tensor) => {
      val array = tensor.storage().array()
      val zeroBaseIndex = index - 1
      (zeroBaseIndex.toLong, psEv.array2Vector(array, 0, array.length))
    }
    }.toMap

    gradWeight = PSSparseRowTensor(vectors, nIndex, nOutput)

    pushGradient()
  }

  def pullParameters(input: Tensor[T]): Unit = {
    inputBuffer.set(input.storage(),
      input.storageOffset(),
      Array(input.nElement()))

    val input_data = inputBuffer.storage().array()
    val input_offset = inputBuffer.storageOffset() - 1
    val numEle = inputBuffer.nElement()

    var i = 0
    while (i < numEle) {
      require(ev.isGreaterEq(input_data(i + input_offset), ev.one),
        "LookupTableSparse: elements of input should be greater than or equal to 1")
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
    val rowNums = (nOutput until 2 * nOutput).toArray
    val indices = VFactory.denseLongVector(gradWeight.getVectors.keys.toArray.sorted)
    val vectors = gradWeight.getVectors.map { case (key, vector) => (long2Long(key), vector) }.asJava

    val param = new UpdateColsParam(matrixId, rowNums, indices, vectors)
    val func = new UpdateColsFunc(param)
    PSAgentContext.get().getUserRequestAdapter.update(func).get()

    gradWeight = null
  }

  override def update(optimizer: Optimizer, epoch: Int, batchSize: Int): Unit = {
    optimizer.update(matrixId, nOutput, epoch, batchSize).get()
  }
}

object LookupTableSparse {
  protected def norm2ScaleWithIndices[T: ClassTag](tensor: Tensor[T],
                                                   indices: Tensor[T],
                                                   maxNorm: T,
                                                   scaleBuffer: mutable.HashMap[Int, T])
                                                  (implicit ev: TensorNumeric[T]): mutable.HashMap[Int, T] = {
    val indicesArray = indices.storage.array()
    var i = indices.storageOffset() - 1
    while (i < indices.nElement() + indices.storageOffset() - 1) {
      val index = ev.toType[Int](indicesArray(i))
      val norm = tensor(index).norm(2)
      if (ev.isGreater(norm, maxNorm)) scaleBuffer(index) = ev.divide(maxNorm, norm)
      i += 1
    }

    scaleBuffer
  }
}
