package com.intel.analytics.bigdl.nn.ps

import com.intel.analytics.bigdl.tensor.Tensor
import com.tencent.angel.spark.context.PSContext
import org.apache.spark.{SparkConf, SparkContext}
import org.junit.{Assert, Test}

class LookupTableTest extends Assert {
  @Test
  def testForward(): Unit = {
    val sparkConf = new SparkConf()
      .setAppName("linear")
      .setMaster("local")
    PSContext.getOrCreate(new SparkContext(sparkConf))
    val lookupTable = new LookupTable[Float]("lookupTable", 10, 10)
    val input = Tensor[Float](2, 1)
    input.setValue(1, 1, 1)
    input.setValue(2, 1, 2)
    lookupTable.pullParameters(input)
    lookupTable.forward(input)
  }
}
