package com.intel.analytics.bigdl.utils.ps

import com.intel.analytics.bigdl.tensor.{DoubleType, FloatType, TensorDataType}
import com.tencent.angel.ml.matrix.RowType

object PSTensorNumeric {
  def getRowType(dataType: TensorDataType) = {
    dataType match {
      case FloatType =>
        RowType.T_FLOAT_DENSE
      case DoubleType =>
        RowType.T_DOUBLE_DENSE
    }
  }
}
