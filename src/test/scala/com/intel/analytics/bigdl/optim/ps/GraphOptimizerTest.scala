package com.intel.analytics.bigdl.optim.ps

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.nn.{Linear, MSECriterion, Sequential}
import com.intel.analytics.bigdl.optim.{Optimizer, Trigger}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Engine
import org.apache.spark.{SparkConf, SparkContext}
import org.junit.{Assert, Test}

class GraphOptimizerTest extends Assert {
  @Test
  def testOptimize(): Unit = {
    System.setProperty("bigdl.localMode", "true")
    val sparkConf = new SparkConf()
      .setAppName("linear")
      .setMaster("local")
    val sc = new SparkContext(sparkConf)
    Engine.init

    val samples = (0 until 10000).map(i => {
      val feature = Tensor[Float](15).zero()
      val label = Tensor[Float](1).zero()

      Sample(feature, label)
    })

    val sampleRDD = sc.parallelize(samples, 2)

    val model = Sequential[Float]()
    model.add(Linear[Float](15, 10))
    model.add(Linear[Float](10, 1))

    val criterion = new MSECriterion[Float]()
//    val optimizer = new SGD(0.01)

    val graphOptimizer = Optimizer(model, sampleRDD, criterion, 500)
    graphOptimizer.setEndWhen(Trigger.maxEpoch(1)).optimize()
  }
}
