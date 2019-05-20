package com.intel.analytics.bigdl.nn.ps

import java.util.{Map => JMap}

import com.intel.analytics.bigdl.nn.abstractnn.{Initializable, TensorModule}
import com.intel.analytics.bigdl.ps.utils.PSTensorNumeric
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.tencent.angel.ml.core.utils.PSMatrixUtils
import com.tencent.angel.ml.math2.vector.Vector
import org.json4s.JsonAST.JLong

import scala.reflect.ClassTag

class LookupTable[T: ClassTag]
(name: String, val nIndex: Int, val nOutput: Int)
(implicit ev: TensorNumeric[T]) extends TensorModule[T] with Initializable {

  private val embedMatCtx = PSMatrixUtils.createPSMatrixCtx(s"${name}_embedding", 2 * nOutput, nIndex,
    PSTensorNumeric.getRowType(ev.getType()))

  lazy val matrixId: Int = PSMatrixUtils.getMatrixId(s"${name}_embedding")

  @transient var embeddings: JMap[JLong, Vector] = _

  override def updateOutput(input: Tensor[T]): Tensor[T] = ???

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = ???
}
