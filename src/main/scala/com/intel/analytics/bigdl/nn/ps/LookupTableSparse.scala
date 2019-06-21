package com.intel.analytics.bigdl.nn.ps

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, Initializable}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.ps.PSTensorNumeric

import scala.reflect.ClassTag

class LookupTableSparse[T: ClassTag]
(val nIndex: Int, val nOutput: Int,
 val combiner: String = "sum",
 val maxNorm: Double = -1,
 var wRegularizer: Regularizer[T] = null
)
(implicit ev: TensorNumeric[T], psEv: PSTensorNumeric[T]) extends AbstractModule[Activity, Tensor[T], T] with Initializable {
  override def updateOutput(input: Activity): Tensor[T] = ???

  override def updateGradInput(input: Activity, gradOutput: Tensor[T]): Activity = ???
}
