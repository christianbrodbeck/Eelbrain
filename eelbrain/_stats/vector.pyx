# optimized statistics functions
#cython: boundscheck=False, wraparound=False

cimport cython
from libc.math cimport sin, cos
import numpy as np
cimport numpy as cnp

ctypedef cnp.int8_t INT8
ctypedef cnp.int64_t INT64
ctypedef cnp.float64_t FLOAT64


def rotation_matrices(cnp.ndarray[FLOAT64, ndim=1] phis,
                      cnp.ndarray[FLOAT64, ndim=1] thetas,
                      cnp.ndarray[FLOAT64, ndim=1] xis,
                      cnp.ndarray[FLOAT64, ndim=3] out):
    cdef unsigned long i, v, case
    cdef FLOAT64 theta, phi, xi
    cdef unsigned long n_cases = phis.shape[0]
    cdef double raxis_x, raxis_y, raxis_z
    cdef double tmp1, tmp2

    cdef double c, s, t

    for case in range(n_cases):
        phi = phis[case]
        theta = thetas[case]
        xi = xis[case]

        raxis_x = sin(phi) * cos(theta)
        raxis_y = sin(phi) * sin(theta)
        raxis_z = cos(phi)
        c = cos(xi)
        s = sin(xi)
        t = 1.0 - c

        out[case, 0, 0] = c + t * raxis_x ** 2
        out[case, 1, 1] = c + t * raxis_y ** 2
        out[case, 2, 2] = c + t * raxis_z ** 2

        tmp1 = raxis_x * raxis_y * t
        tmp2 = raxis_z * s
        out[case, 1, 0] = tmp1 + tmp2
        out[case, 0, 1] = tmp1 - tmp2

        tmp1 = raxis_x * raxis_z * t
        tmp2 = raxis_y * s
        out[case, 2, 0] = tmp1 - tmp2
        out[case, 0, 2] = tmp1 + tmp2

        tmp1 = raxis_y * raxis_z * t
        tmp2 = raxis_x * s
        out[case, 2, 1] = tmp1  + tmp2
        out[case, 1, 2] = tmp1  - tmp2
    return out


@cython.cdivision(True)
def mean_norm_rotated(cnp.ndarray[FLOAT64, ndim=3] y,
                      cnp.ndarray[FLOAT64, ndim=3] rotation,
                      cnp.ndarray[FLOAT64, ndim=1] out):
    cdef unsigned long i, v, case, vi
    cdef double norm, mean

    cdef unsigned long n_cases = y.shape[0]
    cdef unsigned long n_dims = y.shape[1]
    cdef unsigned long n_tests = y.shape[2]

    for i in range(n_tests):
        norm = 0
        for v in range(n_dims):
            mean = 0
            for case in range(n_cases):
                for vi in range(n_dims):
                    mean += rotation[case, v, vi] * y[case, vi, i]
            norm += (mean / n_cases) ** 2
        out[i] = norm ** 0.5
    return out


@cython.cdivision(True)
def t2_stat_rotated(cnp.ndarray[FLOAT64, ndim=3] y,
                    cnp.ndarray[FLOAT64, ndim=3] rotation,
                    cnp.ndarray[FLOAT64, ndim=1] out):
    cdef unsigned long i, v, case, vi
    cdef double norm, mean, var, temp

    cdef unsigned long n_cases = y.shape[0]
    cdef unsigned long n_dims = y.shape[1]
    cdef unsigned long n_tests = y.shape[2]

    for i in range(n_tests):
        norm = 0
        for v in range(n_dims):
            mean = 0
            var = 0
            for case in range(n_cases):
                temp = 0
                for vi in range(n_dims):
                    temp += rotation[case, v, vi] * y[case, vi, i]
                mean += temp
                var += temp ** 2
            temp = mean ** 2
            var -= temp / n_cases
            if var <= 0:
                norm += 0
            else:
                norm += (temp / var)
        out[i] = norm ** 0.5
    return out


@cython.cdivision(True)
def t2_stat(cnp.ndarray[FLOAT64, ndim=3] y,
            cnp.ndarray[FLOAT64, ndim=1] out):
    cdef unsigned long i, v, case
    cdef double norm, mean, var, temp

    cdef unsigned long n_cases = y.shape[0]
    cdef unsigned long n_dims = y.shape[1]
    cdef unsigned long n_tests = y.shape[2]

    for i in range(n_tests):
        norm = 0
        for v in range(n_dims):
            mean = 0
            var = 0
            for case in range(n_cases):
                temp = y[case, v, i]
                mean += temp
                var += temp ** 2
            temp = mean ** 2
            var -= temp / n_cases
            if var <= 0:
                norm += 0
            else:
                norm += (temp / var)
        out[i] = norm ** 0.5
    return out
