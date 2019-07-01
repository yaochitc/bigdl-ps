package com.intel.analytics.bigdl.optim.ps

import com.intel.analytics.bigdl.nn.{Linear, MSECriterion, Sequential}
import com.intel.analytics.bigdl.utils.Engine
import com.tencent.angel.ml.core.optimizer.SGD
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

    val model = Sequential[Float]()
    model.add(Linear[Float](10, 10))
    model.add(Linear[Float](10, 1))

    val criterion = new MSECriterion[Float]()
    val optimizer = new SGD(0.01)

    val graphOptimizer = GraphOptimizer[Float](model, null, criterion, optimizer, 100)
    graphOptimizer.optimize()
  }
}
