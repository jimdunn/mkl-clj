(ns mkl-clj.blas.real.general
  (:import org.bytedeco.javacpp.mkl_rt))

;;;;
;;;; BLAS Level 2 and 3 - real, double precision, general matrices
;;;;                      (no band matrices yet)
;;;;

;;;; BLAS Level 2

(defn cblas-dgemv
  "Matrix vector product with general matrix
  y = alpha*A*x + beta*y,
  or
  y = alpha*A'*x + beta*y,
  where:
  alpha and beta are scalars, x and y are vectors,
  A is an m-by-n matrix.

  layout - storage format (mkl_rt/CblasRowMajor) or (mkl_rt/CblasColMajor).
  trans - if trans=mkl_rt/CblasNoTrans, then y = alpha*A*x + beta*y;
          if trans=mkl_rt/CblasTrans, then y = alpha*A'*x + beta*y;
  m - number of rows of A. must be non-negative
  n - number of columns of A. must be non-negative
  alpha - scalar
  a - double array, size lda*k.
      layout = mkl_rt/CblasColMajor, k is n.
      of the array a must contain the matrix A.
      layout = mkl_rt/CblasRowMajor, k is m.
      of the array a must contain the matrix A.
  lda - the leading dimension of a
        layout = mkl_rt/CblasColMajor, lda must be at least max(1, m).
        layout = mkl_rt/CblasRowMajor, lda must be at least max(1, n).
  x - double array, size at least (1+(n-1)*abs(incx)) when
      trans=mkl_rt/CblasNoTrans and at least (1+(m - 1)*abs(incx)) otherwise
  incx - step size between elements of x. cannot be zero
  beta - scalar. When beta is zero, then y need not be set on input.
  y - double array, size at least (1 +(m - 1)*abs(incy)) when
      trans=mkl_rt/CblasNoTrans and at least (1 +(n - 1)*abs(incy)) otherwise.
  incy - step size between elements of x. cannot be zero"
  [layout trans m n alpha a lda x incx beta y incy]
  (mkl_rt/cblas_dgemv layout trans m n alpha a lda x incx beta y incy))

(defn cblas-dger
  "Rank-1 update of a general matrix.
  A = alpha*x*y'+ A,
  where:
  alpha is a scalar, x is an m-element vector, y is an n-element vector,
  A is an m-by-n general matrix.

  layout - storage format (mkl_rt/CblasRowMajor) or (mkl_rt/CblasColMajor).
  m - number of rows of A. must be non-negative
  n - number of columns of A. must be non-negative
  alpha - scalar
  x - double array, size at least (1 + (m - 1)*abs(incx)).
  incx - step size between elements of x. cannot be zero
  y - double array, size at least (1 + (n - 1)*abs(incy)).
  incy - step size between elements of y. cannot be zero
  a - double array, size lda*k.
      For layout = mkl_rt/CblasColMajor, k is n.
      For layout = mkl_rt/CblasRowMajor, k is m.
  lda - leading dimension of a
        layout = mkl_rt/CblasColMajor, the value of lda must be at least max(1, m).
        layout = mkl_rt/CblasRowMajor, the value of lda must be at least max(1, n)."
  [layout m n alpha x incx y incy a lda]
  (mkl_rt/cblas_dger layout m n alpha x incx y incy a lda))


;;;; BLAS Level 3
(defn cblas-dgemm
  "Product of general matrices.
  C = alpha*op(A)*op(B) + beta*C,
  where:
  op(X) is op(X) = X or op(X) = X^T
  alpha and beta are scalars,
  A, B and C are matrices:
  op(A) is an m-by-k matrix,
  op(B) is a k-by-n matrix,
  C is an m-by-n matrix.

  layout - storage format is (mkl_rt/CblasRowMajor) or (mkl_rt/CblasColMajor).
  transa - if transa=mkl_rt/CblasNoTrans, then op(A) = A;
           if transa=mkl_rt/CblasTrans, then op(A) = A^T;
  transb - if transb=mkl_rt/CblasNoTrans, then op(B) = B;
           if transb=mkl_rt/CblasTrans, then op(B) = B^T;
  m - number of rows of op(A) and C. Must be non-negative.
  n - number of columns of op(B) C. Must be non-negative.
  k - number of columns of op(A) and the number of rows op(B).
      Must be non-negative.
  alpha - scalar
  a - layout = mkl_rt/CblasColMajor, transa=mkl_rt/CblasNoTrans
      or
      layout = mkl_rt/CblasRowMajor, transa=mkl_rt/CblasTrans
      double array, size lda*k.

      layout = mkl_rt/CblasColMajor, transa=mkl_rt/CblasTrans
      or
      layout = mkl_rt/CblasRowMajor, transa=mkl_rt/CblasNoTrans
      double array, size lda*m.
  lda - leading dimension of a
        layout = mkl_rt/CblasColMajor, transa=mkl_rt/CblasNoTrans
        or
        layout = mkl_rt/CblasRowMajor, transa=mkl_rt/CblasTrans
        lda must be at least max(1, m).

        layout = mkl_rt/CblasColMajor, transa=mkl_rt/CblasTrans
        or
        layout = mkl_rt/CblasRowMajor, transa=mkl_rt/CblasNoTrans
        lda must be at least max(1, k)
  b - layout = mkl_rt/CblasColMajor, transb=mkl_rt/CblasNoTrans
      or
      layout = mkl_rt/CblasRowMajor, transb=mkl_rt/CblasTrans
      double array, size ldb by n.

      layout = mkl_rt/CblasColMajor, transb=mkl_rt/CblasTrans
      or
      layout = mkl_rt/CblasRowMajor, transb=mkl_rt/CblasNoTrans
      double array, size ldb by k.
  ldb - leading dimension of b
        layout = mkl_rt/CblasColMajor, transb=mkl_rt/CblasNoTrans
        or
        layout = mkl_rt/CblasRowMajor, transb=mkl_rt/CblasTrans
        ldb must be at least max(1, k).

        layout = mkl_rt/CblasColMajor, transb=mkl_rt/CblasTrans
        or
        layout = mkl_rt/CblasRowMajor, transb=mkl_rt/CblasNoTrans
        ldb must be at least max(1, n).
   beta - scalar. When beta is zero, then c need not be set on input.
   c - layout = mkl_rt/CblasColMajor
       double array, size ldc by n.
       layout = mkl_rt/CblasRowMajor
       double array, size ldc by m.
  ldc - the leading dimension of c
        layout = mkl_rt/CblasColMajor
        ldc must be at least max(1, m).

        layout = mkl_rt/CblasRowMajor
        ldc must be at least max(1, n)."
  [layout trans-a trans-b m n k alpha a lda b ldb beta c ldc]
  (mkl_rt/cblas_dgemm layout trans-a trans-b m n k alpha a lda b ldb beta c ldc))
