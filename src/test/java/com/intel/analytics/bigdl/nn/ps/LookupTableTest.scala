package com.intel.analytics.bigdl.nn.ps

import org.junit.{Assert, Test}

class LookupTableTest extends Assert {
  @Test
  def testForward(): Unit = {
    val lookupTable = new LookupTable[Float]("lookupTable", 100, 10)
  }
}
