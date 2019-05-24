package com.intel.analytics.bigdl.utils.ps

import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.tencent.angel.ml.core.utils.PSMatrixUtils
import com.tencent.angel.ml.math2.storage.{DoubleVectorStorage, FloatVectorStorage}

trait PSTensorNumeric[@specialized(Float, Double) T] extends Serializable {
  def getRow(matrixId: Int, rowId: Int): Tensor[T]

  def getRowAsMatrix(matrixId: Int, rowId: Int, matRows: Int, matCols: Int): Tensor[T]

  def incrementRow(matrixId: Int, rowId: Int, row: Tensor[T]): Unit

  def incrementRowByMatrix(matrixId: Int, rowId: Int, mat: Tensor[T]): Unit
}

object PSTensorNumeric {

  implicit object NumericFloat extends PSTensorNumeric[Float] {
    override def getRow(matrixId: Int, rowId: Int): Tensor[Float] = {
      val vector = PSMatrixUtils.getRow(0, matrixId, rowId)
      val arrayStorage = Storage(vector.getStorage.asInstanceOf[FloatVectorStorage].getValues)
      Tensor(arrayStorage)
    }

    override def getRowAsMatrix(matrixId: Int, rowId: Int, matRows: Int, matCols: Int): Tensor[Float] = {
      val vector = PSMatrixUtils.getRow(0, matrixId, rowId)
      Tensor(vector.getStorage.asInstanceOf[FloatVectorStorage].getValues, Array(matRows, matCols))
    }

    override def incrementRow(matrixId: Int, rowId: Int, row: Tensor[Float]): Unit = {
    }

    override def incrementRowByMatrix(matrixId: Int, rowId: Int, mat: Tensor[Float]): Unit = {
    }
  }

  implicit object NumericDouble extends PSTensorNumeric[Double] {
    override def getRow(matrixId: Int, rowId: Int): Tensor[Double] = {
      val vector = PSMatrixUtils.getRow(0, matrixId, rowId)
      val arrayStorage = Storage(vector.getStorage.asInstanceOf[DoubleVectorStorage].getValues)
      Tensor(arrayStorage)
    }

    override def getRowAsMatrix(matrixId: Int, rowId: Int, matRows: Int, matCols: Int): Tensor[Double] = {
      val vector = PSMatrixUtils.getRow(0, matrixId, rowId)
      Tensor(vector.getStorage.asInstanceOf[DoubleVectorStorage].getValues, Array(matRows, matCols))
    }

    override def incrementRow(matrixId: Int, rowId: Int, row: Tensor[Double]): Unit = {

    }

    override def incrementRowByMatrix(matrixId: Int, rowId: Int, mat: Tensor[Double]): Unit = {

    }
  }

}