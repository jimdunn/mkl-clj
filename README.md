# mkl-clj

A Clojure library for using the [Intel Math Kernel Library](https://software.intel.com/en-us/mkl).
This is written primarily for my own convenience and edification. This code may serve as an
example of how to use MKL or other C and C++ libraries through [Javacpp Presets](https://github.com/bytedeco/javacpp-presets),
but otherwise it would be better to use something like [Neanderthal](https://neanderthal.uncomplicate.org/) instead.

It is necessary to install the [MKL](https://software.intel.com/en-us/mkl) to run this code.

## Example

``` clojure
(import
   'org.bytedeco.javacpp.mkl_rt
   '[org.bytedeco.javacpp DoublePointer]
   '[org.bytedeco.javacpp.indexer DoubleIndexer])

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
  (println " Example completed."))
```

## License

Copyright © 2018 James Dunn

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
