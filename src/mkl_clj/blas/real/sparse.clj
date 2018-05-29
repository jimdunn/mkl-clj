(ns mkl-clj.blas.real.sparse
  (:import org.bytedeco.javacpp.mkl_rt
           org.bytedeco.javacpp.mkl_rt$sparse_matrix
           org.bytedeco.javacpp.mkl_rt$matrix_descr
           [org.bytedeco.javacpp IntPointer DoublePointer]))

;;;;
;;;; Sparse BLAS Level 1 - real, double precision
;;;;

(defn cblas-daxpyi
  "Adds scalar multiple of sparse vector, x, to dense vector y:
  y = a*x + y

  nz - number of elements in x and indx
  a - scalar
  x - double array, size at least nz
  indx - int array of indices for elements of x, values must be distinct,
         size at least nz
  y - double array, size at least max(indx[i])"
  [nz a x indx y]
  (mkl_rt/cblas_daxpyi nz a x indx y))

(defn cblas-ddoti
  "Dot product of a compressed sparse vector and a full-storage.
  Result is zero if nz is not positive.

  nz - number of elements in x and indx
  x - double array, size at least nz.
  indx - int array of indices for elements of x, values must be distinct,
         size at least nz
  y - double array, size at least max(indx[i])"
  [nz x indx y]
  (mkl_rt/cblas_ddoti nz x indx y))

(defn cblas-dgthr
  "Gathers a full-storage sparse vector's elements into compressed form.

  nz - number of elements of y to be gathered
  y - double array, size at least max(indx[i])
  x - double array for non-zero values of sparse vector
  indx - int array of indices for elements to be gathered, size at least nz."
  [nz y x indx]
  (mkl_rt/cblas_dgthr nz y x indx))

(defn cblas-dgthrz
  "Gathers a full-storage sparse vector's elements into compressed form,
  replacing them by zeros.

  nz - number of elements of y to be gathered
  y - double array, size at least max(indx[i])
  x - double array for non-zero values of sparse vector
  indx - int array of indices for elements to be gathered, size at least nz."
  [nz y x indx]
  (mkl_rt/cblas_dgthrz nz y x indx))

(defn cblas-droti
  "Applies the Givens rotation to elements of two real vectors, x (in compressed
  form nz, x, indx) and y (in full storage form):

  x[i] = c*x[i] + s*y[indx[i]]
  y[indx[i]] = c*y[indx[i]]- s*x[i]

  nz - number of elements in x and indx
  x - double array, size at least nz
  indx - int array of indices for elements of x, values must be distinct,
         size at least nz
  y - double array, size at least max(indx[i]).
  c - a scalar, the cosine of the angle of rotation
  s - a scalar, the sine of the angle of rotation"
  [nz x indx y c s]
  (mkl_rt/cblas_droti nz x indx y c s))

(defn cblas-dsctr
  "Converts compressed sparse vector, x, into full storage vector, y.

  nz - number of elements of x to be scattered
  x - double array, size at least nz
  indx - int array of indices for elements of x to be scattered,
         size at least nz
  y - double array, size at least max(indx[i])"
  [nz x indx y]
  (mkl_rt/cblas_dsctr nz x indx y))


;;;;
;;;; Inspector-executor Sparse BLAS Routines
;;;;

(def SPARSE-INDEX-BASE-ZERO mkl_rt/SPARSE_INDEX_BASE_ZERO)
(def SPARSE-MATRIX-TYPE-GENERAL mkl_rt/SPARSE_MATRIX_TYPE_GENERAL)
(def SPARSE-OPERATION-NON-TRANSPOSE mkl_rt/SPARSE_OPERATION_NON_TRANSPOSE)
(def SPARSE-STATUS-SUCCESS mkl_rt/SPARSE_STATUS_SUCCESS)
(def SPARSE-MEMORY-AGGRESSIVE mkl_rt/SPARSE_MEMORY_AGGRESSIVE)

(defn sparse-matrix
  "Creates a handle which will contain the data for sparse matrices to be
  used in inspector-executor routines."
  []
  (mkl_rt$sparse_matrix.))

(defn matrix-descr
  "Returns a matrix_desc struct for Inspector-executor Sparse BLAS Routines."
  [& {:keys [type mode diag]}]
  (let [desc (mkl_rt$matrix_descr.)]
    (when-not (nil? type)
      (.type desc type))
    (when-not (nil? mode)
      (.type desc mode))
    (when-not (nil? diag)
      (.type desc diag))
    desc))

;;;; Matrix Manipulation Routines

(defn sparse-d-create-csr
  "Creates a sparse matrix in CSR format."
  [A indexing rows cols rows-start rows-end col-indx values]
  {:post [(= mkl_rt/SPARSE_STATUS_SUCCESS %)]}
  (mkl_rt/mkl_sparse_d_create_csr
   A indexing rows cols rows-start rows-end col-indx values))

(defn sparse-d-create-coo
  "Creates a sparse matrix in COO format."
  [A indexing rows cols nnz row-indx col-indx values]
  {:post [(= mkl_rt/SPARSE_STATUS_SUCCESS %)]}
  (mkl_rt/mkl_sparse_d_create_csr
   A indexing rows cols nnz row-indx col-indx values))

(defn sparse-copy
  "Copies the sparse matrix, source, into dest."
  [source descr dest]
  (mkl_rt/mkl_sparse_copy source descr dest))

(defn sparse-destroy
  "Frees the memory associated with the sparse matrix handle, A."
  [A]
  (mkl_rt/mkl_sparse_destroy A))

(defn sparse-convert-csr
  "Applies the operation to the sparse matrix in source and stores the
  results in dest."
  [source operation dest]
  (mkl_rt/mkl_sparse_convert_csr source operation dest))

(defn sparse-d-export-csr
  "Exports the internal data for the matrix in A into CSR format."
  [A indexing rows cols rows-start rows-end col-indx values]
  (mkl_rt/mkl_sparse_d_export_csr
   A indexing rows cols rows-start rows-end col-indx values))

(defn sparse-d-set-value
  "Set a specific value in the sparse matrix A."
  [A row col value]
  (mkl_rt/mkl_sparse_d_set_value A row col value))

;;;; Inspector-executor Sparse BLAS Analysis Routines

(defn sparse-set-mv-hint
  [A operation descr expected-calls]
  {:post [(= mkl_rt/SPARSE_STATUS_SUCCESS %)]}
  (mkl_rt/mkl_sparse_set_mv_hint A operation descr expected-calls))

(defn sparse-set-mm-hint
  [A operation descr layout dense-matrix-size expected-calls]
  {:post [(= mkl_rt/SPARSE_STATUS_SUCCESS %)]}
  (mkl_rt/mkl_sparse_set_mm_hint
   A operation descr layout dense-matrix-size expected-calls))

(defn sparse-set-dotmv-hint
  [A operation descr expected-calls]
  {:post [(= mkl_rt/SPARSE_STATUS_SUCCESS %)]}
  (mkl_rt/mkl_sparse_set_dotmv_hint A operation descr expected-calls))

(defn sparse-set-memory-hint
  [A policy]
  (mkl_rt/mkl_sparse_set_memory_hint A policy))

(defn sparse-optimize
  [A]
  {:post [(= mkl_rt/SPARSE_STATUS_SUCCESS %)]}
  (mkl_rt/mkl_sparse_optimize A))

;;;; Inspector-executor Sparse BLAS Execution Routines

(defn sparse-d-mv
  [operation alpha A descr x beta y]
  {:post [(= mkl_rt/SPARSE_STATUS_SUCCESS %)]}
  (mkl_rt/mkl_sparse_d_mv operation alpha A descr x beta y))

(defn sparse-d-mm
  [operation alpha A descr layout x columns ldx beta y ldy]
  {:post [(= mkl_rt/SPARSE_STATUS_SUCCESS %)]}
  (mkl_rt/mkl_sparse_d_mm
   operation alpha A descr layout x columns ldx beta y ldy))

(defn sparse-d-add
  "Adds sparse matrices in A and B after applying operation to A and
  multiplying A by the scalar alpha, and stores results in C.
  C = alpha*operation(A)*B"
  [operation A alpha B C]
  {:post [(= mkl_rt/SPARSE_STATUS_SUCCESS %)]}
  (mkl_rt/mkl_sparse_d_add operation A alpha B C))

(defn sparse-spmm
  [operation A B C]
  {:post [(= mkl_rt/SPARSE_STATUS_SUCCESS %)]}
  (mkl_rt/mkl_sparse_spmm
   operation A B C))

(defn sparse-d-dotmv
  [operation alpha A descr x beta y d]
  {:post [(= mkl_rt/SPARSE_STATUS_SUCCESS %)]}
  (mkl_rt/mkl_sparse_d_dotmv operation alpha A descr x beta y d))

;;;;
;;;; Sparse BLAS Level 2
;;;;

(defn sparse-d-mv
  "y = alpha*op(A)*x + beta*y"
  [operation alpha A descr x beta y]
  (mkl_rt/mkl_sparse_d_mv operation alpha A descr x beta y))

(defn sparse-d-trsv
  "Solves
  op(A)*y = alpha * x
  where A is a triangular sparse matrix, alpha is a scalar, and x and y are vectors."
  [operation alpha A descr x y]
  (mkl_rt/mkl_sparse_d_trsv operation alpha A descr x y))

;;;;
;;;; Sparse BLAS Level 3
;;;;

(defn sparse-d-mm
  "y = alpha*op(A)*x + beta*y
  where alpha and beta are scalars, A is a sparse matrix, and x and y are dense
  matrices."
  [operation alpha A descr layout x columns ldx beta y ldy]
  (mkl_rt/mkl_sparse_d_mm operation alpha A descr layout x columns ldx beta y ldy))

(defn sparse-d-trsm
  "Solves a system of linear equations with multiple right hand sides for a
  triangular sparse matrix.
  y = alpha*inv(op(A))*x
  where alpha is a scalar, x and y are dense matrices, and A is a sparse matrix."
  [operation alpha A descr layout x columns ldx y ldy]
  (mkl_rt/mkl_sparse_d_trsm operation alpha A descr layout x columns ldx y ldy))
