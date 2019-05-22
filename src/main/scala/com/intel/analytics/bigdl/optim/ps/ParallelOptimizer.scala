package com.intel.analytics.bigdl.optim.ps

import com.intel.analytics.bigdl.{Criterion, Module}
import com.intel.analytics.bigdl.dataset.{DistributedDataSet, MiniBatch}
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.optim.Optimizer
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class ParallelOptimizer[T: ClassTag]
(_model: Module[T],
 _dataset: DistributedDataSet[MiniBatch[T]],
 _criterion: Criterion[T]
)(implicit ev: TensorNumeric[T]) extends Optimizer[T, MiniBatch[T]](
  _model, _dataset, _criterion) {

  override def optimize(): Module[T] = {
    val sc = _dataset.originRDD().sparkContext
    val modelBroadcast = ModelBroadcast[T]().broadcast(sc, model)



    null
  }
}
