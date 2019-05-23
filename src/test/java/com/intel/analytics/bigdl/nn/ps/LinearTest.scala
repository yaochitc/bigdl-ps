package com.intel.analytics.bigdl.nn.ps

import org.junit.{Assert, Test}

class LinearTest extends Assert {
  @Test
  def testBuild(): Unit = {
    val linear = new Linear[Float]("linear", 10, 10)
  }

  @Test
  def testForward(): Unit = {
  }
}
