package com.intel.analytics.bigdl.optim.ps

import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.nn.ps.abstractnn.ParameterSupport
import com.intel.analytics.bigdl.optim.DistriOptimizer.logger
import com.intel.analytics.bigdl.optim.Optimizer
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.T
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

    val _featurePaddingParam = Option(featurePaddingParam)
    val _labelPaddingParam = Option(labelPaddingParam)

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
    val distDataset = dataset.asInstanceOf[DistributedDataSet[MiniBatch[T]]]

    val sc = distDataset.originRDD().sparkContext
    var dataRDD = distDataset.data(train = true)

    val driverState = T(
      "epoch" -> optimMethods.values.head.state("epoch"),
      "neval" -> optimMethods.values.head.state("neval")
    )

    var recordsProcessedThisEpoch = 0
    val shuffleBefore = System.nanoTime()
    logger.info("Shuffle data")
    distDataset.shuffle()
    val shuffleEnd = System.nanoTime()
    logger.info(s"Shuffle data complete. Takes ${(shuffleEnd - shuffleBefore) / 1e9}s")

    val numSamples = distDataset.data(train = false).map(_.size()).reduce(_ + _)

    val broadcast = sc.broadcast((model, criterion, _optimizer))

    while (!endWhen(driverState)) {
      val recordsNum = sc.accumulator(0, "record number")

      dataRDD.foreachPartition(data => {
        val batch = data.next()

        val (localModel, localCriterion, localOptimizer) = broadcast.value
        val input = batch.getInput()
        val target = batch.getTarget()
        val output = localModel.forward(input)
        val loss = localCriterion.forward(output, target)
        val errors = localCriterion.backward(output, target)
        localModel.backward(input, errors)

        val batchSize = batch.size()
        val epoch = driverState[Int]("epoch")
        localModel.asInstanceOf[ParameterSupport[T]].update(localOptimizer, epoch, batchSize)

        recordsNum += batchSize
      })

      recordsProcessedThisEpoch += recordsNum.value

      driverState("neval") = driverState[Int]("neval") + 1
      if (recordsProcessedThisEpoch >= numSamples) {
        // Epoch is finished
        driverState("epoch") = driverState[Int]("epoch") + 1
        distDataset.shuffle()
        dataRDD = distDataset.data(train = true)
        recordsProcessedThisEpoch = 0
      }
    }

    model
  }
}
