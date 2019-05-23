package com.intel.analytics.bigdl.optim.ps

import com.intel.analytics.bigdl.dataset.{DistributedDataSet, MiniBatch}
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.nn.Container
import com.intel.analytics.bigdl.nn.mkldnn.MklDnnContainer
import com.intel.analytics.bigdl.nn.mkldnn.Phase.TrainingPhase
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}
import com.intel.analytics.bigdl.{Criterion, Module}
import org.apache.log4j.Logger
import org.apache.spark.TaskContext
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

object ParallelOptimizer {

  val logger: Logger = Logger.getLogger(getClass)

  case class Cache[T]
  (localModel: Module[T],
   localCriterion: Criterion[T],
   localState: Table,
   localMethods: Option[Array[ValidationMethod[T]]],
   optimMethods: Map[String, OptimMethod[T]]
  )

  private[optim] def optimize[T: ClassTag]
  (dataset: DistributedDataSet[MiniBatch[T]],
   state: Table,
   endWhen: Trigger,
   models: RDD[Cache[T]],
   optimMethods: Map[String, OptimMethod[T]]
  )(implicit ev: TensorNumeric[T]): Unit = {
    val sc = dataset.originRDD().sparkContext
    val partitionNum = dataset.originRDD().partitions.length

    val driverState = T(
      "epoch" -> optimMethods.values.head.state("epoch"),
      "neval" -> optimMethods.values.head.state("neval"),
      "Loss" -> optimMethods.values.head.state("Loss"),
      "score" -> optimMethods.values.head.state("score")
    )

    var dataRDD = dataset.data(train = true)

    while (!endWhen(driverState)) {
      dataRDD.zipPartitions(models, preservesPartitioning = true) { (data, modelIter) => {
        val batch = data.next()

        val cached = modelIter.next()
        val localModel = cached.localModel

        localModel.training()
        val localCriterion = cached.localCriterion
        val input = batch.getInput()
        val target = batch.getTarget()
        val output = localModel.forward(input)
        val loss = ev.toType[Double](localCriterion.forward(output, target))
        val errors = localCriterion.backward(output, target)
        localModel.backward(input, errors)

        Iterator.single(1)
      }
      }.reduce(_ + _)

      models.mapPartitions { modelIter =>
        val modelCache = modelIter.next()

        Iterator.empty
      }.count()
    }
  }

  private def initThreadModels[T: ClassTag]
  (model: Module[T],
   dataset: DistributedDataSet[MiniBatch[T]],
   criterion: Criterion[T],
   state: Table,
   validationMethods: Option[Array[ValidationMethod[T]]],
   optimMethod: Map[String, OptimMethod[T]])
  (implicit ev: TensorNumeric[T]): (RDD[Cache[T]], ModelBroadcast[T]) = {
    val sc = dataset.originRDD().sparkContext
    val broadcast = sc.broadcast((criterion, state, validationMethods, optimMethod))

    val modelBroadcast = ModelBroadcast[T]().broadcast(sc, model)

    val models = dataset.originRDD().mapPartitions(_ => {
      val partitionId = TaskContext.getPartitionId
      val (broadcastCriterion, broadcastState, broadcastMethod,
      broadcastOptim) = broadcast.value

      val localModel = modelBroadcast.value(true)
      localModel match {
        case container: MklDnnContainer => container.compile(TrainingPhase)
        case _ =>
      }

      // differentiate partition models from each other by partition ID
      setModelId(localModel, partitionId)
      val localCriterion = broadcastCriterion.cloneCriterion()
      val localState = broadcastState.clone()
      val localMethod =
        if (broadcastMethod.isDefined) Some(broadcastMethod.get.map(_.clone())) else None

      Iterator.single(Cache(
        localModel, // model
        criterion, // criterion
        state, // state
        localMethod,
        broadcastOptim.map(v => (v._1, v._2.clone()))
      ))
    }).persist()

    models.setName("Thread Model RDD")
    logger.info("Cache thread models...")
    models.count()
    logger.info("Cache thread models... done")
    (models, modelBroadcast)
  }

  private def setModelId[T: ClassTag](model: Module[T], partitionId: Int): Unit = {
    model.setId(partitionId)
    if (model.isInstanceOf[Container[_, _, T]]) {
      model.asInstanceOf[Container[_, _, T]].modules.
        foreach(sub => setModelId(sub, partitionId))
    }
  }
}

class ParallelOptimizer[T: ClassTag]
(_model: Module[T],
 _dataset: DistributedDataSet[MiniBatch[T]],
 _criterion: Criterion[T]
)(implicit ev: TensorNumeric[T]) extends Optimizer[T, MiniBatch[T]](
  _model, _dataset, _criterion) {

  override def optimize(): Module[T] = {
    val distDataset = dataset.asInstanceOf[DistributedDataSet[MiniBatch[T]]]

    optimMethods.values.foreach { optimMethod =>
      optimMethod.clearHistory()
    }

    null
  }
}
