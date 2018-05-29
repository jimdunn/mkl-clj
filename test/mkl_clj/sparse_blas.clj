(ns mkl-clj.sparse-blas
  (:require [mkl-clj.blas.real.sparse :as sblas])
  (:import [org.bytedeco.javacpp IntPointer DoublePointer])
  (:use midje.sweet))

(facts
 "Sparse BLAS Level 1"
 (fact "daxpyi"
       (let [nz 2
             n 5
             a 2.0
             x (double-array [1.0 2.0])
             indx (int-array [0 4])
             y (double-array (repeat n 1.0))]
         (sblas/cblas-daxpyi nz a x indx y)
         (into [] y) => [3.0 1.0 1.0 1.0 5.0]))
 (fact "ddoti"
       (let [nz 2
             n 5
             x (double-array [1.0 2.0])
             indx (int-array [0 4])
             y (double-array (repeat n 1.0))]
         (sblas/cblas-ddoti nz x indx y) => 3.0))
 (fact "dgthr"
       (let [nz 2
             n 5
             x (double-array nz)
             indx (int-array [0 4])
             y (double-array [1.0 0.0 0.0 0.0 2.0])]
         (sblas/cblas-dgthr nz y x indx)
         [(into [] x)
          (into [] y)] => [[1.0 2.0]
                           [1.0 0.0 0.0 0.0 2.0]]))
 (fact "dgthrz"
       (let [nz 2
             n 5
             x (double-array nz)
             indx (int-array [0 4])
             y (double-array [1.0 0.0 0.0 0.0 2.0])]
         (sblas/cblas-dgthrz nz y x indx)
         [(into [] x)
          (into [] y)] => [[1.0 2.0]
                           [0.0 0.0 0.0 0.0 0.0]]))
 (fact
  "droti"
  (let [nz 2
        n 5
        x (double-array [1.0 1.0])
        indx (int-array [0 4])
        y (double-array [1.0 0.0 0.0 0.0 -1.0])
        ;; 45 degree clockwise rotation
        c (/ 1.0 (Math/sqrt 2.0))
        s (/ 1.0 (Math/sqrt 2.0))]
    (sblas/cblas-droti nz x indx y c s)
    [(into [] x)
     (into [] y)] =>
    (just
     (just
      [(roughly 1.41421) (roughly 0.0)])
     (just
      [(roughly 0.0) (roughly 0.0) (roughly 0.0) (roughly 0.0) (roughly -1.41421)]))))
 (fact "dsctr"
       (let [nz 2
             n 5
             x (double-array [1.0 2.0])
             indx (int-array [0 4])
             y (double-array n)]
         (sblas/cblas-dsctr nz x indx y)
         (into [] y) => [1.0 0.0 0.0 0.0 2.0])))

(facts
 "Sparse BLAS Level 2"
 (fact
  "sparse-d-mv"
  (let [A (sblas/sparse-matrix)
        indexing sblas/SPARSE-INDEX-BASE-ZERO
        rows 5 cols 5
        ;; wasn't able to get this working with the Pointers
        ;; CSR Storage
        values (DoublePointer. (double-array [1 -1 -3 -2 5 4 6 4 -4 2 7 8 -5]))
        rows-start (IntPointer. (int-array [0 3 5 8 11]))
        rows-end (IntPointer. (int-array [3 5 8 11 13]))
        col-indx (IntPointer. (int-array [0 1 3 0 1 2 3 4 0 2 3 1 4]))
        create-stat (sblas/sparse-d-create-csr
                     A indexing rows cols rows-start rows-end col-indx values)
        ;; inspector routines
        type sblas/SPARSE-MATRIX-TYPE-GENERAL
        descr (sblas/matrix-descr :type type)
        op sblas/SPARSE-OPERATION-NON-TRANSPOSE
        repeat 1
        hint-stat (sblas/sparse-set-mv-hint A op descr repeat)
        policy sblas/SPARSE-MEMORY-AGGRESSIVE
        memory-stat (sblas/sparse-set-memory-hint A policy)
        optimize-stat (sblas/sparse-optimize A)
        ;; y = alpha*op(A)*x + beta*y
        alpha 2.0
        beta 3.0
        x (double-array [1.0 2.0 1.0 2.0 1.0])
        y (double-array [2.0 1.0 2.0 1.0 2.0])
        ;; executor
        mv-stat (sblas/sparse-d-mv op alpha A descr x beta y)]
    (into [] y) => [-8.0 19.0 46.0 27.0 28.0])))
