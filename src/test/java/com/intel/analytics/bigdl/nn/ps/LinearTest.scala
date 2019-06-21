package com.intel.analytics.bigdl.nn.ps

import com.intel.analytics.bigdl.tensor.Tensor
import com.tencent.angel.spark.context.PSContext
import org.apache.spark.{SparkConf, SparkContext}
import org.junit.{Assert, Test}

class LinearTest extends Assert {
  @Test
  def testForward(): Unit = {
    val sparkConf = new SparkConf()
      .setAppName("linear")
      .setMaster("local")
    PSContext.getOrCreate(new SparkContext(sparkConf))
    val linear = new Linear[Float]("linear", 10, 10)
    val input = Tensor[Float](5, 10).zero()
    linear.pullParameters(input)
    linear.forward(input)
  }
}
