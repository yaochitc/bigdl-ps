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
    val linear = new Linear[Float]("linear", 1, 10, 1)
    val input = Tensor[Float](5, 10).zero()
    linear.pullParameters()
    linear.forward(input)

    val gradOutput = Tensor[Float](5, 1).zero()
    linear.backward(input, gradOutput)

    linear.pushGradient()
  }
}
