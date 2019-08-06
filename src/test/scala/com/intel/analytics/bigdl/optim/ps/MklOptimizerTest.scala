package com.intel.analytics.bigdl.optim.ps

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.nn.{CrossEntropyCriterion, KullbackLeiblerDivergenceCriterion}
import com.intel.analytics.bigdl.nn.mkldnn.{HeapData, Input, ReorderMemory, SoftMax}
import com.intel.analytics.bigdl.nn.mkldnn.Phase.TrainingPhase
import com.intel.analytics.bigdl.nn.mkldnn.ps.{Linear, Sequential}
import com.intel.analytics.bigdl.optim.Trigger
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Engine
import com.tencent.angel.ml.core.optimizer.Adam
import com.tencent.angel.spark.context.PSContext
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.junit.Test

class MklOptimizerTest {
  @Test
  def testOptimize(): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
    Logger.getLogger("breeze").setLevel(Level.ERROR)
    Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)

    System.setProperty("bigdl.localMode", "true")
    val sparkConf = new SparkConf()
      .setAppName("linear")
      .setMaster("local")
    val sc = new SparkContext(sparkConf)
    PSContext.getOrCreate(sc)
    Engine.init

    val sampleRDD = sc.textFile("data/iris.data", 1).filter(!"".equals(_)).map(line => {
      val subs = line.split(",") // "," may exist in content.
      val feature = Tensor(subs.slice(0, 4).map(_.toFloat), Array(4))
      val getLabel: String => Float = {
        case "Iris-setosa" => 1.0f
        case "Iris-versicolor" => 2.0f
        case "Iris-virginica" => 3.0f
      }
      Sample[Float](feature, Tensor(Array(getLabel(subs(4))), Array(1)))
    })

    val batchSize = 150
    val inputShape = Array(batchSize, 4)
    val outputShape = Array(batchSize, 3)
    val model = Sequential()
    model.add(Input(inputShape, Memory.Format.nc))
    model.add(Linear("firstLayer", 3, 4, 100))
    model.add(Linear("secondLayer", 3, 100, 100))
    model.add(Linear("thirdLayer", 3, 100, 3))

    model.add(ReorderMemory(HeapData(outputShape, Memory.Format.nc)))
    model.compile(TrainingPhase)

    val criterion = new CrossEntropyCriterion[Float]()
    val optimizer = new Adam(0.02, 0.99, 0.9)

    val opt = GraphOptimizer[Float](model, sampleRDD, criterion, optimizer, batchSize).setEndWhen(Trigger.maxEpoch(1000))
    val trainedModel = opt.optimize()
  }
}
