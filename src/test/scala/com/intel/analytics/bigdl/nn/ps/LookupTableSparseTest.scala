package com.intel.analytics.bigdl.nn.ps

import com.intel.analytics.bigdl.tensor.Tensor
import com.tencent.angel.spark.context.PSContext
import org.apache.spark.{SparkConf, SparkContext}
import org.junit.{Assert, Test}

class LookupTableSparseTest extends Assert {
  @Test
  def testForward(): Unit = {
    val sparkConf = new SparkConf()
      .setAppName("linear")
      .setMaster("local")
    PSContext.getOrCreate(new SparkContext(sparkConf))
    val lookupTableSparse = new LookupTableSparse[Float]("lookupTableSparse", 10, 10)
    val input = Tensor.sparse(Tensor[Float](1, 1).setValue(1, 1, 1))
    lookupTableSparse.pullParameters(input)
    lookupTableSparse.forward(input)

    val gradOutput = Tensor[Float](2, 1, 10).zero()
    lookupTableSparse.backward(input, gradOutput)

    lookupTableSparse.pushGradient()
  }
}
