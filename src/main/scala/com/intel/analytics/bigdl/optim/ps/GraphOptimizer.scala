package com.intel.analytics.bigdl.optim.ps

import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.optim.Optimizer
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.{Criterion, Module}
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

object GraphOptimizer {
  def apply[T: ClassTag]
  (model: Module[T],
   sampleRDD: RDD[Sample[T]],
   criterion: Criterion[T],
   batchSize: Int,
   miniBatchImpl: MiniBatch[T]
  )(implicit ev: TensorNumeric[T]): Optimizer[T, MiniBatch[T]] = {
    new GraphOptimizer[T](
      _model = model,
      _dataset = (DataSet.rdd(sampleRDD) ->
        SampleToMiniBatch(miniBatchImpl, batchSize, None))
        .asInstanceOf[DistributedDataSet[MiniBatch[T]]],
      _criterion = criterion
    ).asInstanceOf[Optimizer[T, MiniBatch[T]]]
  }
}

class GraphOptimizer[T: ClassTag]
(_model: Module[T],
 _dataset: DistributedDataSet[MiniBatch[T]],
 _criterion: Criterion[T]
)(implicit ev: TensorNumeric[T]) extends Optimizer[T, MiniBatch[T]](
  _model, _dataset, _criterion) {

  override def optimize(): Module[T] = ???
}
