package com.intel.analytics.bigdl.utils.ps

import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.tencent.angel.ml.core.utils.PSMatrixUtils
import com.tencent.angel.ml.math2.storage.{DoubleVectorStorage, FloatVectorStorage}

trait PSTensorNumeric[@specialized(Float, Double) T] extends Serializable {
  def getRow(matrixId: Int, rowId: Int): Tensor[T]

  def getRowAsMatrix(matrixId: Int, rowId: Int, matRows: Int, matCols: Int): Tensor[T]

}

object PSTensorNumeric {

  implicit object NumericFloat extends PSTensorNumeric[Float] {
    override def getRow(matrixId: Int, rowId: Int): Tensor[Float] = {
      val vector = PSMatrixUtils.getRow(0, matrixId, rowId)
      val arrayStorage = Storage(vector.asInstanceOf[FloatVectorStorage].getValues)
      Tensor(arrayStorage)
    }

    override def getRowAsMatrix(matrixId: Int, rowId: Int, matRows: Int, matCols: Int): Tensor[Float] = {
      val vector = PSMatrixUtils.getRow(0, matrixId, rowId)
      Tensor(vector.asInstanceOf[FloatVectorStorage].getValues, Array(matRows, matCols))
    }
  }

  implicit object NumericDouble extends PSTensorNumeric[Double] {
    override def getRow(matrixId: Int, rowId: Int): Tensor[Double] = {
      val vector = PSMatrixUtils.getRow(0, matrixId, rowId)
      val arrayStorage = Storage(vector.asInstanceOf[DoubleVectorStorage].getValues)
      Tensor(arrayStorage)
    }

    override def getRowAsMatrix(matrixId: Int, rowId: Int, matRows: Int, matCols: Int): Tensor[Double] = {
      val vector = PSMatrixUtils.getRow(0, matrixId, rowId)
      Tensor(vector.asInstanceOf[DoubleVectorStorage].getValues, Array(matRows, matCols))
    }
  }

}