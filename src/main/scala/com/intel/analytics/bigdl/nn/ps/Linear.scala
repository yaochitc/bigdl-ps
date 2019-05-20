package com.intel.analytics.bigdl.nn.ps

import com.intel.analytics.bigdl.nn.abstractnn.{Initializable, TensorModule}
import com.intel.analytics.bigdl.ps.utils.PSTensorNumeric
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.tencent.angel.ml.core.utils.PSMatrixUtils
import com.tencent.angel.ml.math2.matrix.Matrix
import com.tencent.angel.ml.math2.vector.Vector

import scala.reflect.ClassTag

class Linear[T: ClassTag]
(name: String,
 val inputSize: Int,
 val outputSize: Int
)(implicit ev: TensorNumeric[T]) extends TensorModule[T] with Initializable {
  val weightCtx =
    PSMatrixUtils.createPSMatrixCtx(s"${name}_weight", inputSize * 2, outputSize, PSTensorNumeric.getRowType(ev.getType()))

  val biasCtx =
    PSMatrixUtils.createPSMatrixCtx(s"${name}_bias", 2, outputSize, PSTensorNumeric.getRowType(ev.getType()))

  lazy val weightId: Int = PSMatrixUtils.getMatrixId(s"${name}_weight")
  lazy val biasId: Int = PSMatrixUtils.getMatrixId(s"${name}_bias")

  @transient var weight: Matrix = _
  @transient var bias: Vector = _

  override def updateOutput(input: Tensor[T]): Tensor[T] = ???

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = ???
}
