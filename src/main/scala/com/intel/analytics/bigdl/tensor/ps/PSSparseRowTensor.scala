package com.intel.analytics.bigdl.tensor.ps

import breeze.linalg.{DenseMatrix, DenseVector}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.bigdl.utils.ps.PSTensorNumeric
import com.tencent.angel.ml.math2.vector._
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Matrix

import scala.reflect.ClassTag

class PSSparseRowTensor[@specialized(Float, Double) T: ClassTag]
(_vectors: Map[Long, Vector],
 _size: Array[Int])
(implicit ev: TensorNumeric[T], psEv: PSTensorNumeric[T]) extends Tensor[T] {
  def getVectors: Map[Long, Vector] = _vectors

  override def isEmpty: Boolean = ???

  override def isScalar: Boolean = ???

  override def nDimension(): Int = 2

  override def dim(): Int = ???

  override def size(): Array[Int] = _size

  override def size(dim: Int): Int = {
    require(dim == 1 || dim == 2,
      s"PSSparseRowTensor: input must be equal to 1 or 2, input dim ${dim}")

    _size(dim - 1)
  }

  override def stride(): Array[Int] = ???

  override def stride(dim: Int): Int = ???

  override def fill(v: T): Tensor[T] = ???

  override def forceFill(v: Any): Tensor[T] = ???

  override def zero(): Tensor[T] = ???

  override def randn(): Tensor[T] = ???

  override def randn(mean: Double, stdv: Double): Tensor[T] = ???

  override def rand(): Tensor[T] = ???

  override def rand(lowerBound: Double, upperBound: Double): Tensor[T] = ???

  override def bernoulli(p: Double): Tensor[T] = ???

  override def transpose(dim1: Int, dim2: Int): Tensor[T] = ???

  override def t(): Tensor[T] = ???

  override def apply(index: Int): Tensor[T] = ???

  override def apply(indexes: Array[Int]): T = ???

  override def value(): T = ???

  override def valueAt(d1: Int): T = ???

  override def valueAt(d1: Int, d2: Int): T = ???

  override def valueAt(d1: Int, d2: Int, d3: Int): T = ???

  override def valueAt(d1: Int, d2: Int, d3: Int, d4: Int): T = ???

  override def valueAt(d1: Int, d2: Int, d3: Int, d4: Int, d5: Int): T = ???

  override def apply(t: Table): Tensor[T] = ???

  override def update(index: Int, value: T): Unit = ???

  override def update(index: Int, src: Tensor[T]): Unit = ???

  override def update(indexes: Array[Int], value: T): Unit = ???

  override def setValue(value: T): PSSparseRowTensor.this.type = ???

  override def setValue(d1: Int, value: T): PSSparseRowTensor.this.type = ???

  override def setValue(d1: Int, d2: Int, value: T): PSSparseRowTensor.this.type = ???

  override def setValue(d1: Int, d2: Int, d3: Int, value: T): PSSparseRowTensor.this.type = ???

  override def setValue(d1: Int, d2: Int, d3: Int, d4: Int, value: T): PSSparseRowTensor.this.type = ???

  override def setValue(d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, value: T): PSSparseRowTensor.this.type = ???

  override def update(t: Table, value: T): Unit = ???

  override def update(t: Table, src: Tensor[T]): Unit = ???

  override def update(filter: T => Boolean, value: T): Unit = ???

  override def isContiguous(): Boolean = ???

  override def contiguous(): Tensor[T] = ???

  override def isSameSizeAs(other: Tensor[_]): Boolean = ???

  override def emptyInstance(): Tensor[T] = ???

  override def resizeAs(src: Tensor[_]): Tensor[T] = ???

  override def cast[D](castTensor: Tensor[D])(implicit evidence$1: ClassTag[D], ev: TensorNumeric[D]): Tensor[D] = ???

  override def resize(sizes: Array[Int], strides: Array[Int]): Tensor[T] = ???

  override def resize(size1: Int): Tensor[T] = ???

  override def resize(size1: Int, size2: Int): Tensor[T] = ???

  override def resize(size1: Int, size2: Int, size3: Int): Tensor[T] = ???

  override def resize(size1: Int, size2: Int, size3: Int, size4: Int): Tensor[T] = ???

  override def resize(size1: Int, size2: Int, size3: Int, size4: Int, size5: Int): Tensor[T] = ???

  override def nElement(): Int = ???

  override def select(dim: Int, index: Int): Tensor[T] = {
    require(dim == 1,
      s"PSSparseRowTensor: input must be equal to 1, input dim ${dim}")

    psEv.vector2Tensor(_vectors(index - 1))
  }

  override def storage(): Storage[T] = ???

  override def storageOffset(): Int = ???

  override def set(other: Tensor[T]): Tensor[T] = ???

  override def set(storage: Storage[T], storageOffset: Int, sizes: Array[Int], strides: Array[Int]): Tensor[T] = ???

  override def set(): Tensor[T] = ???

  override def narrow(dim: Int, index: Int, size: Int): Tensor[T] = ???

  override def copy(other: Tensor[T]): Tensor[T] = ???

  override def applyFun[A](t: Tensor[A], func: A => T)(implicit evidence$2: ClassTag[A]): Tensor[T] = ???

  override def apply1(func: T => T): Tensor[T] = ???

  override def zipWith[A, B](t1: Tensor[A], t2: Tensor[B], func: (A, B) => T)(implicit evidence$3: ClassTag[A], evidence$4: ClassTag[B]): Tensor[T] = ???

  override def map(other: Tensor[T], func: (T, T) => T): Tensor[T] = ???

  override def squeeze(): Tensor[T] = ???

  override def squeeze(dim: Int): Tensor[T] = ???

  override def squeezeNewTensor(): Tensor[T] = ???

  override def view(sizes: Array[Int]): Tensor[T] = ???

  override def unfold(dim: Int, size: Int, step: Int): Tensor[T] = ???

  override def repeatTensor(sizes: Array[Int]): Tensor[T] = ???

  override def expandAs(template: Tensor[T]): Tensor[T] = ???

  override def expand(sizes: Array[Int]): Tensor[T] = ???

  override def split(size: Int, dim: Int): Array[Tensor[T]] = ???

  override def split(dim: Int): Array[Tensor[T]] = ???

  override def toBreezeVector(): DenseVector[T] = ???

  override def toMLlibVector(): linalg.Vector = ???

  override def toBreezeMatrix(): DenseMatrix[T] = ???

  override def toMLlibMatrix(): Matrix = ???

  override def getType(): TensorDataType = ???

  override def diff(other: Tensor[T], count: Int, reverse: Boolean): Boolean = ???

  override def addSingletonDimension(t: Tensor[T], dim: Int): Tensor[T] = ???

  override def reshape(sizes: Array[Int]): Tensor[T] = ???

  override def save(path: String, overWrite: Boolean): PSSparseRowTensor.this.type = ???

  override def getTensorNumeric(): TensorNumeric[T] = ???

  override def getTensorType: TensorType = ???

  override def toArray(): Array[T] = ???

  override def +(s: T): Tensor[T] = ???

  override def +(t: Tensor[T]): Tensor[T] = ???

  override def -(s: T): Tensor[T] = ???

  override def -(t: Tensor[T]): Tensor[T] = ???

  override def unary_-(): Tensor[T] = ???

  override def /(s: T): Tensor[T] = ???

  override def /(t: Tensor[T]): Tensor[T] = ???

  override def *(s: T): Tensor[T] = ???

  override def *(t: Tensor[T]): Tensor[T] = ???

  override def sum(): T = ???

  override def prod(): T = ???

  override def prod(x: Tensor[T], dim: Int): Tensor[T] = ???

  override def sum(dim: Int): Tensor[T] = ???

  override def sum(x: Tensor[T], dim: Int): Tensor[T] = ???

  override def mean(): T = ???

  override def mean(dim: Int): Tensor[T] = ???

  override def max(): T = ???

  override def max(dim: Int): (Tensor[T], Tensor[T]) = ???

  override def max(values: Tensor[T], indices: Tensor[T], dim: Int): (Tensor[T], Tensor[T]) = ???

  override def min(): T = ???

  override def min(dim: Int): (Tensor[T], Tensor[T]) = ???

  override def min(values: Tensor[T], indices: Tensor[T], dim: Int): (Tensor[T], Tensor[T]) = ???

  override def scatter(dim: Int, index: Tensor[T], src: Tensor[T]): Tensor[T] = ???

  override def gather(dim: Int, index: Tensor[T], src: Tensor[T]): Tensor[T] = ???

  override def conv2(kernel: Tensor[T], vf: Char): Tensor[T] = ???

  override def xcorr2(kernel: Tensor[T], vf: Char): Tensor[T] = ???

  override def sqrt(): Tensor[T] = ???

  override def tanh(): Tensor[T] = ???

  override def abs(): Tensor[T] = ???

  override def add(value: T, y: Tensor[T]): Tensor[T] = ???

  override def add(y: Tensor[T]): Tensor[T] = ???

  override def add(x: Tensor[T], value: T, y: Tensor[T]): Tensor[T] = ???

  override def add(value: T): Tensor[T] = ???

  override def add(x: Tensor[T], y: Tensor[T]): Tensor[T] = ???

  override def dot(y: Tensor[T]): T = ???

  override def cmax(value: T): Tensor[T] = ???

  override def dist(y: Tensor[T], norm: Int): T = ???

  override def addcmul(value: T, tensor1: Tensor[T], tensor2: Tensor[T]): Tensor[T] = ???

  override def addcmul(tensor1: Tensor[T], tensor2: Tensor[T]): Tensor[T] = ???

  override def addcdiv(value: T, tensor1: Tensor[T], tensor2: Tensor[T]): Tensor[T] = ???

  override def sub(value: T, y: Tensor[T]): Tensor[T] = ???

  override def sub(x: Tensor[T], value: T, y: Tensor[T]): Tensor[T] = ???

  override def sub(y: Tensor[T]): Tensor[T] = ???

  override def sub(x: Tensor[T], y: Tensor[T]): Tensor[T] = ???

  override def sub(value: T): Tensor[T] = ???

  override def cmul(y: Tensor[T]): Tensor[T] = ???

  override def cmul(x: Tensor[T], y: Tensor[T]): Tensor[T] = ???

  override def cdiv(y: Tensor[T]): Tensor[T] = ???

  override def cdiv(x: Tensor[T], y: Tensor[T]): Tensor[T] = ???

  override def mul(value: T): Tensor[T] = ???

  override def div(value: T): Tensor[T] = ???

  override def div(y: Tensor[T]): Tensor[T] = ???

  override def mul(x: Tensor[T], value: T): Tensor[T] = ???

  override def addmm(v1: T, M: Tensor[T], v2: T, mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] = ???

  override def addmm(M: Tensor[T], mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] = ???

  override def addmm(mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] = ???

  override def addmm(v2: T, mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] = ???

  override def addmm(v1: T, v2: T, mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] = ???

  override def mm(mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] = ???

  override def addr(t1: Tensor[T], t2: Tensor[T]): Tensor[T] = ???

  override def addr(v1: T, t1: Tensor[T], t2: Tensor[T]): Tensor[T] = ???

  override def addr(v1: T, t1: Tensor[T], v2: T, t2: Tensor[T]): Tensor[T] = ???

  override def addr(v1: T, t1: Tensor[T], v2: T, t2: Tensor[T], t3: Tensor[T]): Tensor[T] = ???

  override def uniform(args: T*): T = ???

  override def addmv(beta: T, vec1: Tensor[T], alpha: T, mat: Tensor[T], vec2: Tensor[T]): Tensor[T] = ???

  override def addmv(beta: T, alpha: T, mat: Tensor[T], vec2: Tensor[T]): Tensor[T] = ???

  override def addmv(alpha: T, mat: Tensor[T], vec2: Tensor[T]): Tensor[T] = ???

  override def mv(mat: Tensor[T], vec2: Tensor[T]): Tensor[T] = ???

  override def baddbmm(beta: T, M: Tensor[T], alpha: T, batch1: Tensor[T], batch2: Tensor[T]): Tensor[T] = ???

  override def baddbmm(beta: T, alpha: T, batch1: Tensor[T], batch2: Tensor[T]): Tensor[T] = ???

  override def baddbmm(alpha: T, batch1: Tensor[T], batch2: Tensor[T]): Tensor[T] = ???

  override def bmm(batch1: Tensor[T], batch2: Tensor[T]): Tensor[T] = ???

  override def pow(y: Tensor[T], n: T): Tensor[T] = ???

  override def pow(n: T): Tensor[T] = ???

  override def square(): Tensor[T] = ???

  override def floor(y: Tensor[T]): Tensor[T] = ???

  override def floor(): Tensor[T] = ???

  override def ceil(): Tensor[T] = ???

  override def inv(): Tensor[T] = ???

  override def erf(): Tensor[T] = ???

  override def erfc(): Tensor[T] = ???

  override def logGamma(): Tensor[T] = ???

  override def digamma(): Tensor[T] = ???

  override def topk(k: Int, dim: Int, increase: Boolean, result: Tensor[T], indices: Tensor[T], sortedResult: Boolean): (Tensor[T], Tensor[T]) = ???

  override def log(y: Tensor[T]): Tensor[T] = ???

  override def exp(y: Tensor[T]): Tensor[T] = ???

  override def sqrt(y: Tensor[T]): Tensor[T] = ???

  override def tanh(y: Tensor[T]): Tensor[T] = ???

  override def log1p(y: Tensor[T]): Tensor[T] = ???

  override def log(): Tensor[T] = ???

  override def exp(): Tensor[T] = ???

  override def log1p(): Tensor[T] = ???

  override def abs(x: Tensor[T]): Tensor[T] = ???

  override def norm(y: Tensor[T], value: Int, dim: Int): Tensor[T] = ???

  override def gt(x: Tensor[T], y: Tensor[T]): Tensor[T] = ???

  override def lt(x: Tensor[T], y: Tensor[T]): Tensor[T] = ???

  override def le(x: Tensor[T], y: Tensor[T]): Tensor[T] = ???

  override def eq(x: Tensor[T], y: T): Tensor[T] = ???

  override def maskedFill(mask: Tensor[T], e: T): Tensor[T] = ???

  override def maskedCopy(mask: Tensor[T], y: Tensor[T]): Tensor[T] = ???

  override def maskedSelect(mask: Tensor[T], y: Tensor[T]): Tensor[T] = ???

  override def norm(value: Int): T = ???

  override def sign(): Tensor[T] = ???

  override def ge(x: Tensor[T], value: Double): Tensor[T] = ???

  override def indexAdd(dim: Int, index: Tensor[T], y: Tensor[T]): Tensor[T] = ???

  override def index(dim: Int, index: Tensor[T], y: Tensor[T]): Tensor[T] = ???

  override def cmax(y: Tensor[T]): Tensor[T] = ???

  override def cmin(y: Tensor[T]): Tensor[T] = ???

  override def cmax(x: Tensor[T], y: Tensor[T]): Tensor[T] = ???

  override def cmin(x: Tensor[T], y: Tensor[T]): Tensor[T] = ???

  override def range(xmin: Double, xmax: Double, step: Int): Tensor[T] = ???

  override def negative(x: Tensor[T]): Tensor[T] = ???

  override def reduce(dim: Int, result: Tensor[T], reducer: (T, T) => T): Tensor[T] = ???

  override def sumSquare(): T = ???

  override def clamp(min: Double, max: Double): Tensor[T] = ???

  override def toTensor[D](implicit ev: TensorNumeric[D]): Tensor[D] = ???

  override private[bigdl] def toQuantizedTensor: QuantizedTensor[T] =
    throw new IllegalArgumentException("SparseTensor cannot be cast to QuantizedTensor")
}

object PSSparseRowTensor {
  def apply[@specialized(Float, Double) T: ClassTag](vectors: Map[Long, Vector], nVector: Int, vectorSize: Int)(
    implicit ev: TensorNumeric[T], psEv: PSTensorNumeric[T]): PSSparseRowTensor[T] = new PSSparseRowTensor[T](vectors, Array(nVector, vectorSize))
}