package com.intel.analytics.bigdl.utils.ps

import com.intel.analytics.bigdl.tensor.ps.PSDenseTensor
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.tencent.angel.ml.core.utils.PSMatrixUtils
import com.tencent.angel.ml.math2.VFactory
import com.tencent.angel.ml.math2.storage.{DoubleVectorStorage, FloatVectorStorage}

trait PSTensorNumeric[@specialized(Float, Double) T] extends Serializable {
  def getRow(matrixId: Int, rowId: Int): PSDenseTensor[T]

  def getRowAsMatrix(matrixId: Int, rowId: Int, matRows: Int, matCols: Int): PSDenseTensor[T]

  def incrementRow(matrixId: Int, rowId: Int, row: PSDenseTensor[T]): Unit

  def incrementRowByMatrix(matrixId: Int, rowId: Int, mat: PSDenseTensor[T]): Unit
}

object PSTensorNumeric {

  implicit object NumericFloat extends PSTensorNumeric[Float] {
    override def getRow(matrixId: Int, rowId: Int): PSDenseTensor[Float] = {
      val vector = PSMatrixUtils.getRow(0, matrixId, rowId)
      val arrayStorage = Storage(vector.getStorage.asInstanceOf[FloatVectorStorage].getValues)
      PSDenseTensor[Float](Tensor(arrayStorage), vector.getMatrixId, vector.getClock)
    }

    override def getRowAsMatrix(matrixId: Int, rowId: Int, matRows: Int, matCols: Int): PSDenseTensor[Float] = {
      val vector = PSMatrixUtils.getRow(0, matrixId, rowId)
      PSDenseTensor[Float](Tensor(vector.getStorage.asInstanceOf[FloatVectorStorage].getValues,
        Array(matRows, matCols)), vector.getMatrixId, vector.getClock)
    }

    override def incrementRow(matrixId: Int, rowId: Int, row: PSDenseTensor[Float]): Unit = {
      val data = row.storage().slice(row.storageOffset(), row.nElement()).toArray
      val vector = VFactory.denseFloatVector(row.matrixId, rowId, row.clock, data)
      PSMatrixUtils.incrementRow(matrixId, rowId, vector)
    }

    override def incrementRowByMatrix(matrixId: Int, rowId: Int, mat: PSDenseTensor[Float]): Unit = {
      incrementRow(matrixId, rowId, mat)
    }
  }

  implicit object NumericDouble extends PSTensorNumeric[Double] {
    override def getRow(matrixId: Int, rowId: Int): PSDenseTensor[Double] = {
      val vector = PSMatrixUtils.getRow(0, matrixId, rowId)
      val arrayStorage = Storage(vector.getStorage.asInstanceOf[DoubleVectorStorage].getValues)
      PSDenseTensor[Double](Tensor(arrayStorage), vector.getMatrixId, vector.getClock)
    }

    override def getRowAsMatrix(matrixId: Int, rowId: Int, matRows: Int, matCols: Int): PSDenseTensor[Double] = {
      val vector = PSMatrixUtils.getRow(0, matrixId, rowId)
      PSDenseTensor[Double](Tensor(vector.getStorage.asInstanceOf[DoubleVectorStorage].getValues,
        Array(matRows, matCols)), vector.getMatrixId, vector.getClock)
    }

    override def incrementRow(matrixId: Int, rowId: Int, row: PSDenseTensor[Double]): Unit = {
      val data = row.storage().slice(row.storageOffset(), row.nElement()).toArray
      val vector = VFactory.denseDoubleVector(row.matrixId, rowId, row.clock, data)
      PSMatrixUtils.incrementRow(matrixId, rowId, vector)
    }

    override def incrementRowByMatrix(matrixId: Int, rowId: Int, mat: PSDenseTensor[Double]): Unit = {
      incrementRow(matrixId, rowId, mat)
    }
  }

}