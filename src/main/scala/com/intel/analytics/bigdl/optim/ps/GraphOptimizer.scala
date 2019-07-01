package com.intel.analytics.bigdl.optim.ps

import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.optim.Optimizer
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.{Criterion, Module}
import com.tencent.angel.ml.core.optimizer.{Optimizer => AngelOptimizer}
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

object GraphOptimizer {
  def apply[T: ClassTag]
  (model: Module[T],
   sampleRDD: RDD[Sample[T]],
   criterion: Criterion[T],
   optimizer: AngelOptimizer,
   batchSize: Int,
   featurePaddingParam: PaddingParam[T] = null,
   labelPaddingParam: PaddingParam[T] = null
  )(implicit ev: TensorNumeric[T]): Optimizer[T, MiniBatch[T]] = {

    val _featurePaddingParam = if (featurePaddingParam != null) Some(featurePaddingParam) else None
    val _labelPaddingParam = if (labelPaddingParam != null) Some(labelPaddingParam) else None

    new GraphOptimizer[T](
      _model = model,
      _dataset = (DataSet.rdd(sampleRDD) ->
        SampleToMiniBatch(batchSize, _featurePaddingParam, _labelPaddingParam))
        .asInstanceOf[DistributedDataSet[MiniBatch[T]]],
      _criterion = criterion,
      _optimizer = optimizer
    ).asInstanceOf[Optimizer[T, MiniBatch[T]]]
  }
}

class GraphOptimizer[T: ClassTag]
(_model: Module[T],
 _dataset: DistributedDataSet[MiniBatch[T]],
 _criterion: Criterion[T],
 _optimizer: AngelOptimizer
)(implicit ev: TensorNumeric[T]) extends Optimizer[T, MiniBatch[T]](
  _model, _dataset, _criterion) {

  override def optimize(): Module[T] = {
    val sc = _dataset.originRDD().sparkContext
    val dataRDD = _dataset.originRDD()

    null
  }
}
