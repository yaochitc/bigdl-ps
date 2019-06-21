package com.intel.analytics.bigdl.utils.ps

import com.intel.analytics.bigdl.tensor.ps.PSSparseTensor
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.tencent.angel.ml.core.utils.PSMatrixUtils
import com.tencent.angel.ml.math2.VFactory
import com.tencent.angel.ml.math2.storage.{DoubleVectorStorage, FloatVectorStorage}
import com.tencent.angel.ml.psf.columns.{GetColsFunc, GetColsParam, GetColsResult}
import com.tencent.angel.psagent.PSAgentContext

import scala.collection.JavaConverters._

trait PSTensorNumeric[@specialized(Float, Double) T] extends Serializable {
  def getRow(matrixId: Int, rowId: Int): Tensor[T]

  def getRowAsMatrix(matrixId: Int, rowId: Int, matRows: Int, matCols: Int): Tensor[T]

  def getRowAsSparseMatrix(matrixId: Int, rows: Array[Int], cols: Array[Long]): PSSparseTensor[T]

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
      Tensor(vector.getStorage.asInstanceOf[FloatVectorStorage].getValues,
        Array(matRows, matCols))
    }

    override def getRowAsSparseMatrix(matrixId: Int, rows: Array[Int], cols: Array[Long]): PSSparseTensor[Float] = {
      val param = new GetColsParam(matrixId, rows, cols)
      val func = new GetColsFunc(param)
      val result = PSAgentContext.get.getUserRequestAdapter.get(func).asInstanceOf[GetColsResult]
      val vectors = result.results.asScala.map { case (id, vector) => (id.toInt, vector) }.toMap
      PSSparseTensor(vectors)
    }

    override def incrementRow(matrixId: Int, rowId: Int, row: Tensor[Float]): Unit = {
      val data = row.storage().slice(row.storageOffset(), row.nElement()).toArray
      val vector = VFactory.denseFloatVector(matrixId, rowId, 0, data)
      PSMatrixUtils.incrementRow(matrixId, rowId, vector)
    }

    override def incrementRowByMatrix(matrixId: Int, rowId: Int, mat: Tensor[Float]): Unit = {
      incrementRow(matrixId, rowId, mat)
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
      Tensor(vector.getStorage.asInstanceOf[DoubleVectorStorage].getValues,
        Array(matRows, matCols))
    }

    override def getRowAsSparseMatrix(matrixId: Int, rows: Array[Int], cols: Array[Long]): PSSparseTensor[Double] = {
      val param = new GetColsParam(matrixId, rows, cols)
      val func = new GetColsFunc(param)
      val result = PSAgentContext.get.getUserRequestAdapter.get(func).asInstanceOf[GetColsResult]
      val vectors = result.results.asScala.map { case (id, vector) => (id.toInt, vector) }.toMap
      PSSparseTensor(vectors)
    }

    override def incrementRow(matrixId: Int, rowId: Int, row: Tensor[Double]): Unit = {
      val data = row.storage().slice(row.storageOffset(), row.nElement()).toArray
      val vector = VFactory.denseDoubleVector(matrixId, rowId, 0, data)
      PSMatrixUtils.incrementRow(matrixId, rowId, vector)
    }

    override def incrementRowByMatrix(matrixId: Int, rowId: Int, mat: Tensor[Double]): Unit = {
      incrementRow(matrixId, rowId, mat)
    }
  }

}