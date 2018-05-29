(ns mkl-clj.blas.real.vector
  (:import org.bytedeco.javacpp.mkl_rt))

;;;;
;;;; BLAS Level 1 - real, double precision
;;;;

(defn cblas-dasum
  "Returns the sum of the magnitudes of the elements of the vector x.

  n - number of elements in x
  x - double array, size at least (1 + (n-1)*abs(incx))
  incx - step size between elements of x"
  [n x incx]
  (mkl_rt/cblas_dasum n x incx))

(defn cblas-daxpy
  "Product of scalar and vector plus another vector.
  y = a*x + y

  n - number of elements in x and y
  a - the scalar
  x - double array, size at least (1 + (n-1)*abs(incx))
  incx - step size between elements of x
  y - double array, size at least (1 + (n-1)*abs(incy))
  incy - step size between elements of y"
  [n alpha x incx y incy]
  (mkl_rt/cblas_daxpy n alpha x incx y incy))

(defn cblas-dcopy
  "Copies the elements of a vector x to the vector y. If n isn't positive
  y is unchanged.

  n - number of elements in x and y.
  x - double array, size at least (1 + (n-1)*abs(incx)).
  incx - step size between elements of x
  y - double array, size at least (1 + (n-1)*abs(incy)).
  incy - step size between elements of y"
  [n x incx y incy]
  (mkl_rt/cblas_dcopy n x incx y incy))

(defn cblas-ddot
  "Returns the dot product of vectors x and y. If n isn't positive,
  the return value is 0.

  n - number of elements in x and y.
  x - double array, size at least (1 + (n-1)*abs(incx))
  incx - step size between elements of x
  y - double array, size at least (1 + (n-1)*abs(incy))
  incy - step size between elements of y"
  [n x incx y incy]
  (mkl_rt/cblas_ddot n x incx y incy))

(defn cblas-dnrm2
  "Returns the Euclidean norm of a vector x.

  n - number of elements in x
  x - double array, size at least (1 + (n-1)*abs(incx))
  incx - step size between elements of x"
  [n x incx]
  (mkl_rt/cblas_dnrm2 n x incx))

(defn cblas-drot
  "Rotation of points in the plane. The vector x contains the x-coordinates
  of the points and the y-coordinates are in y.
  The elements of x are replaced by c*x + s*y and the elements of y are
  replaced by c*y - s*x. c and s are the cosine and sine of the angle of
  rotation. The rotations are clockwise for some reason.

  n - number of elements in x and y
  x - double array, size at least (1 + (n-1)*abs(incx))
  incx - step size between elements of x
  y - double array, size at least (1 + (n-1)*abs(incy))
  incy - step size between elements of y
  c - cosine of angle of rotation
  s - sine of angle of rotation"
  [n x incx y incy c s]
  (mkl_rt/cblas_drot n x incx y incy c s))

(defn cblas-drotg
  "Computes the parameters for a Givens rotation of a point with
  cartesian coordinates (a, b).
  Given a and b, this routine produces c and s for the transformation:
   ⌈c  s⌉ ⌈a⌉ =  ⌈r⌉
   ⌊-s c⌋ ⌊b⌋    ⌊0⌋
  The parameter z is defined such that if |a| > |b|, z is s; otherwise
  if c is not 0 z is 1/c; otherwise z is 1.
  Arguments are all double arrays with length 1.
  a - contains the parameter r
  b - contains the parameter z - can be used to construct c and s
  c - cosine of the angle of rotation
  s - sine of the angle of rotation"
  [a b c s]
  (mkl_rt/cblas_drotg a b c s))

(defn cblas-drotm
  "Performs modified Givens rotation of points in the plane represented by the
  vectors x and y which respectively hold the x and y coordinates of the points.
  Each element x[i[ is replaced by h11*x[i] + h12*y[i] and each element y[i] is
  replaced by h21*x[i] + h22*y[i].

   ⌈x_i⌉ = H ⌈x_i⌉
   ⌊y_i⌋     ⌊y_i⌋

  n - number of elements in x and y.
  x - double array, size at least (1 + (n-1)*abs(incx)).
  incx - step size between elements of x
  y - double array, size at least (1 + (n-1)*abs(incy)).
  incy - step size between elements of y
  param - double array - [flag h11 h21 h12 h22]

  flag = -1.0:
  H =  ⌈h11 h12⌉
       ⌊h21 h22⌋

  flag = 0.0:
  H =  ⌈1.0 h12⌉
       ⌊h21 1.0⌋

  flag = 1.0:
  H =  ⌈h11 1.0⌉
       ⌊-1.0 h22⌋

  flag = -2.0:
  H =  ⌈1.0 0.0⌉
       ⌊0.0 1.0⌋
  When the flag is not -1.0, the components of H that are always 1.0, -1.0,
  or 0.0 are just assumed based on the flag. The are not required to be set in
  params."
  [n x incx y incy param]
  (mkl_rt/cblas_drotm n x incx y incy param))

(defn cblas-drotmg
  "Computes elements of a modified Givens rotation H for a vector [x1 y1] that
  transforms it to [x1 0].

   ⌈x1⌉ = H ⌈x1 sqrt(d1)⌉
   ⌊0 ⌋     ⌊y1 sqrt(d2)⌋

  The vector must be scaled when rotated to keep the x component the same
  magnitude.

  d1 - scaling factor for the x-coordinate of the vector
  d2 - scaling factor for the y-coordinate of the vector
  x1 - x-coordinate of the vector
  y1 - y-coordinate of the vector
  param - double array - [flag h11 h21 h12 h22]

  if flag = -1.0 then
  H =  ⌈h11 h12⌉
       ⌊h21 h22⌋

  flag = 0.0:
  H =  ⌈1.0 h12⌉
       ⌊h21 1.0⌋

  flag = 1.0:
  H =  ⌈h11 1.0⌉
       ⌊-1.0 h22⌋

  flag = -2.0:
  H =  ⌈1.0 0.0⌉
       ⌊0.0 1.0⌋
  When the flag is not -1.0, the components of H that are always 1.0, -1.0,
  or 0.0 are just assumed based on the flag. The are not required to be set in
  params."
  [d1 d2 x1 x2 param]
  (mkl_rt/cblas_drotmg d1 d2 x1 x2 param))

(defn cblas-dscal
  "Scales a vector x by a.
  x = a*x

  n - number of elements in x
  a - the scalar
  x - double array, size at least (1 + (n-1)*abs(incx))
  incx - step size between elements of x"
  [n a x incx]
  (mkl_rt/cblas_dscal n a x incx))

(defn cblas-dswap
  "Swaps the values of two vectors x and y.

  n - number of elements in x and y
  x - double array, size at least (1 + (n-1)*abs(incx))
  incx - step size between elements of x
  y - double array, size at least (1 + (n-1)*abs(incy))
  incy - step size between elements of y"
  [n x incx y incy]
  (mkl_rt/cblas_dswap n x incx y incy))

(defn cblas-idamax
  "Returns the index of the first element with the maximum absolute value or
  0 if n or incx aren't positive.

  n - number of elements in x
  x - double array, size at least (1 + (n-1)*abs(incx))
  incx - step size between elements of x"
  [n x incx]
  (mkl_rt/cblas_idamax n x incx))

(defn cblas-idamin
  "Returns the index of the first element with the minimum absolute value or
  0 if n or incx aren't positive.

  n - number of elements in x
  x - double array, size at least (1 + (n-1)*abs(incx))
  incx - step size between elements of x"
  [n x incx]
  (mkl_rt/cblas_idamin n x incx))
