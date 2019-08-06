package com.intel.analytics.bigdl.optim.ps

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.nn.CrossEntropyCriterion
import com.intel.analytics.bigdl.nn.mkldnn.Phase.TrainingPhase
import com.intel.analytics.bigdl.nn.mkldnn.ps.Linear
import com.intel.analytics.bigdl.nn.mkldnn.ps.Sequential
import com.intel.analytics.bigdl.optim.Trigger
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Engine
import com.tencent.angel.ml.core.optimizer.Adam
import com.tencent.angel.spark.context.PSContext
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.junit.{Assert, Test}

class GraphOptimizerTest extends Assert {
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

    val model = Sequential()
    model.add(Linear("firstLayer", 3, 4, 40))
    model.add(Linear("secondLayer", 3, 40, 20))
    model.add(Linear("thirdLayer", 3, 20, 3))

    model.compile(TrainingPhase)

    val criterion = new CrossEntropyCriterion[Float]()
    val optimizer = new Adam(0.02, 0.99, 0.9)

    val opt = GraphOptimizer[Float](model, sampleRDD, criterion, optimizer, 150).setEndWhen(Trigger.maxEpoch(5000))
    val trainedModel = opt.optimize()
  }
}
