package com.intel.analytics.bigdl.optim.ps

import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.nn.ps.abstractnn.ParameterSupport
import com.intel.analytics.bigdl.optim.DistriOptimizer.logger
import com.intel.analytics.bigdl.optim.Optimizer.header
import com.intel.analytics.bigdl.optim.{OptimMethod, Optimizer, Trigger}
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
      optimizer = optimizer
    ).asInstanceOf[Optimizer[T, MiniBatch[T]]]
  }

  def optimize[T: ClassTag]
  (
    model: Module[T],
    dataset: DistributedDataSet[MiniBatch[T]],
    criterion: Criterion[T],
    optimizer: AngelOptimizer,
    endWhen: Trigger,
    optimMethods: Map[String, OptimMethod[T]]
  )(implicit ev: TensorNumeric[T]): Unit = {
    val sc = dataset.originRDD().sparkContext
    var wallClockTime = 0L

    var dataRDD = dataset.data(train = true)

    val driverState = T(
      "epoch" -> optimMethods.values.head.state("epoch"),
      "neval" -> optimMethods.values.head.state("neval")
    )

    var recordsProcessedThisEpoch = 0
    val shuffleBefore = System.nanoTime()
    logger.info("Shuffle data")
    dataset.shuffle()
    val shuffleEnd = System.nanoTime()
    logger.info(s"Shuffle data complete. Takes ${(shuffleEnd - shuffleBefore) / 1e9}s")

    val numSamples = dataset.data(train = false).map(_.size()).reduce(_ + _)

    model.asInstanceOf[ParameterSupport[T]].init()
    val broadcast = sc.broadcast((model, criterion, optimizer))

    while (!endWhen(driverState)) {
      val lossSum = sc.doubleAccumulator
      val recordsNum = sc.longAccumulator
      val start = System.nanoTime()

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

        lossSum.add(ev.toType[Float](loss))
        recordsNum.add(batchSize)
      })

      recordsProcessedThisEpoch += recordsNum.value.toInt
      val end = System.nanoTime()
      wallClockTime += end - start
      driverState("isGradientUpdated") = true
      driverState("Loss") = lossSum.value.toFloat

      driverState("Throughput") = recordsNum.value.toFloat / ((end - start) / 1e9f)
      val _header = header(driverState[Int]("epoch"), recordsProcessedThisEpoch, numSamples,
        driverState[Int]("neval"), wallClockTime)
      logger.info(s"${_header} Trained ${recordsNum.value} records in ${(end - start) / 1e9} " +
        s"seconds. Throughput is ${driverState("Throughput")} records/second. Loss is ${
          driverState("Loss")
        }.")

      driverState("neval") = driverState[Int]("neval") + 1
      if (recordsProcessedThisEpoch >= numSamples) {
        // Epoch is finished
        driverState("epoch") = driverState[Int]("epoch") + 1
        dataset.shuffle()
        dataRDD = dataset.data(train = true)
        recordsProcessedThisEpoch = 0
      }
    }

    model
  }
}

class GraphOptimizer[T: ClassTag]
(_model: Module[T],
 _dataset: DistributedDataSet[MiniBatch[T]],
 _criterion: Criterion[T],
 optimizer: AngelOptimizer
)(implicit ev: TensorNumeric[T]) extends Optimizer[T, MiniBatch[T]](
  _model, _dataset, _criterion) {

  override def optimize(): Module[T] = {
    val distDataset = dataset.asInstanceOf[DistributedDataSet[MiniBatch[T]]]

    GraphOptimizer.optimize(model,
      distDataset,
      criterion,
      optimizer,
      endWhen,
      optimMethods)
    model
  }
}
