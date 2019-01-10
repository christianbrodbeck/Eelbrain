# optimized statistics functions
#cython: boundscheck=False, wraparound=False

cimport cython
from libc.math cimport sin, cos
import numpy as np
cimport numpy as cnp

ctypedef cnp.int8_t INT8
ctypedef cnp.int64_t INT64
ctypedef cnp.float64_t FLOAT64


# Chistian
# def rotation_matrices(cnp.ndarray[FLOAT64, ndim=1] phis,
#                       cnp.ndarray[FLOAT64, ndim=1] thetas,
#                       cnp.ndarray[FLOAT64, ndim=3] out):
#     cdef unsigned long i, v, case
#     cdef FLOAT64 theta, phi
#     cdef unsigned long n_cases = phis.shape[0]
#
#     for case in range(n_cases):
#         phi = phis[case]
#         theta = thetas[case]
#         out[case, 0, 0] = cos(theta)
#         out[case, 1, 0] = sin(theta)
#         out[case, 2, 0] = 0
#         out[case, 0, 1] = -sin(theta) * cos(phi)
#         out[case, 1, 1] = cos(theta) * cos(phi)
#         out[case, 2, 1] = sin(phi)
#         out[case, 0, 2] = sin(theta) * sin(phi)
#         out[case, 1, 2] = cos(theta) * -sin(phi)
#         out[case, 2, 2] = cos(phi)
#     return out

# Proloy
def rotation_matrices(cnp.ndarray[FLOAT64, ndim=1] phis,
                      cnp.ndarray[FLOAT64, ndim=1] thetas,
                      cnp.ndarray[FLOAT64, ndim=3] out):
    cdef unsigned long i, v, case
    cdef FLOAT64 theta, phi
    cdef unsigned long n_cases = phis.shape[0]

    for case in range(n_cases):
        phi = phis[case]
        theta = thetas[case]
        out[case, 0, 0] = cos(phi)
        out[case, 1, 0] = - sin(phi)
        out[case, 2, 0] = 0
        out[case, 0, 1] = sin(phi) * cos(theta)
        out[case, 1, 1] = cos(phi) * cos(theta)
        out[case, 2, 1] = -sin(theta)
        out[case, 0, 2] = sin(phi) * sin(theta)
        out[case, 1, 2] = cos(phi) * sin(theta)
        out[case, 2, 2] = cos(theta)
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
