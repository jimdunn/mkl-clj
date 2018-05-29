(ns mkl-clj.core
  (:import
   org.bytedeco.javacpp.mkl_rt
   [org.bytedeco.javacpp DoublePointer]
   [org.bytedeco.javacpp.indexer DoubleIndexer]))

;; //==============================================================
;; //
;; // SAMPLE SOURCE CODE - SUBJECT TO THE TERMS OF SAMPLE CODE LICENSE AGREEMENT,
;; // http://software.intel.com/en-us/articles/intel-sample-source-code-license-agreement/
;; //
;; // Copyright 2016-2017 Intel Corporation
;; //
;; // THIS FILE IS PROVIDED "AS IS" WITH NO WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT
;; // NOT LIMITED TO ANY IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
;; // PURPOSE, NON-INFRINGEMENT OF INTELLECTUAL PROPERTY RIGHTS.
;; //
;; // =============================================================
;; /*******************************************************************************
;; *   This example computes real matrix C=alpha*A*B+beta*C using Intel(R) MKL
;; *   function dgemm, where A, B, and C are matrices and alpha and beta are
;; *   scalars in double precision.
;; *
;; *   In this simple example, practices such as memory management, data alignment,
;; *   and I/O that are necessary for good programming style and high MKL
;; *   performance are omitted to improve readability.
;; ********************************************************************************/
(defn dgemm-example []
  (println "\n This example computes real matrix C=alpha*A*B+beta*C using \n"
           "Intel(R) MKL function dgemm, where A, B, and  C are matrices and \n"
           "alpha and beta are double precision scalars\n")
  (let [m 2000 p 200 n 1000
        _ (println
           (format
            (str
             " Initializing data for matrix multiplication C=A*B for matrix \n"
             " A(%dx%d) and matrix B(%dx%d)\n")
            m p p n))
        alpha 1.0 beta 0.0
        _ (println
           " Allocating memory for matrices aligned on 64-byte boundary for better \n"
           "performance \n")
        A (DoublePointer. (mkl_rt/MKL_malloc (* m p Double/BYTES) 64))
        B (DoublePointer. (mkl_rt/MKL_malloc (* p n Double/BYTES) 64))
        C (DoublePointer. (mkl_rt/MKL_malloc (* m n Double/BYTES) 64))
        _ (when (every? nil? [A B C])
            (println "\n ERROR: Can't allocate memory for matrices. Aborting... \n")
            (mkl_rt/MKL_free A)
            (mkl_rt/MKL_free B)
            (mkl_rt/MKL_free C))
        A-idx (DoubleIndexer/create (.capacity A (* m p)))
        B-idx (DoubleIndexer/create (.capacity B (* p n)))
        C-idx (DoubleIndexer/create (.capacity C (* m n)))]

    (dotimes [i (* m p)] (.put A i (double (inc i))))
    (dotimes [i (* p n)] (.put B i (double (dec (* -1.0 i)))))
    (dotimes [i (* m n)] (.put C i 0.0))

    (println
     " Computing matrix product using Intel(R) MKL dgemm function via CBLAS interface")
    (mkl_rt/cblas_dgemm
     mkl_rt/CblasRowMajor mkl_rt/CblasNoTrans mkl_rt/CblasNoTrans
     m n p alpha A p B n beta C n)

    (doseq [i (range 6) j (range 6)]
      (println (format "%12.0f" (.get A-idx ^long (+ j (* i p))))))

    (doseq [i (range 6) j (range 6)]
      (println (format "%12.0f" (.get B-idx ^long (+ j (* i n))))))

    (doseq [i (range 6) j (range 6)]
      (println (format "%12.5G" (.get C-idx ^long (+ j (* i n))))))

    (println "\n Deallocating memory")
    (mkl_rt/MKL_free A)
    (mkl_rt/MKL_free B)
    (mkl_rt/MKL_free C)
    (println " Example completed.")))

;; Here's my simpler version. It doesn't appear that the performance of dgemm is
;; affected by using a DoublePointer vs double-array except that accessing the
;; elements of the double-array appears to be faster than accessing the elements
;; of the DoublePointer by a factor of two.
;; mkl_rt/MKL_free can be called on the Pointers or the deallocate method can
;; be used to free memory, though, which is handy.

(defn dgemm-example-1 []
  (let [m 2000 p 200 n 1000
        alpha 1.0 beta 0.0

        A (double-array (* m p))
        B (double-array (* p n))
        C (double-array (* m n))]

    (dotimes [i (* m p)] (aset ^doubles A i (double (inc i))))
    (dotimes [i (* p n)] (aset ^doubles B i (double (dec (* -1.0 i)))))
    (dotimes [i (* m n)] (aset ^doubles C i 0.0))

    (mkl_rt/cblas_dgemm
     mkl_rt/CblasRowMajor mkl_rt/CblasNoTrans mkl_rt/CblasNoTrans
     m n p alpha A p B n beta C n)

    (doseq [i (range 6) j (range 6)]
      (let [i (long i) j (long j)]
        (println (format "%12.0f" (aget ^doubles A ^long (+ j (* i p)))))))

    (doseq [i (range 6) j (range 6)]
      (let [i (long i) j (long j)]
        (println (format "%12.0f" (aget ^doubles B ^long (+ j (* i n)))))))

    (doseq [i (range 6) j (range 6)]
      (let [i (long i) j (long j)]
        (println (format "%12.5G" (aget ^doubles C ^long (+ j (* i n)))))))))
