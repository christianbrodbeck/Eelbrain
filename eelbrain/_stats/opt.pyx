# optimized statistics functions
# cython: language_level=3, boundscheck=False, wraparound=False
from cython.view cimport array as cvarray
from libc.stdlib cimport malloc, free
cimport numpy as np


def anova_full_fmaps(
        const np.npy_float64[:,:] y,
        const np.npy_float64[:,:] x,
        const np.npy_float64[:,:] xsinv,
        np.npy_float64[:,:] f_map,
        const np.npy_int64[:,:] effects,
        const np.npy_int8[:, :] e_ms,
):
    """Compute f-maps for a balanced, fully specified ANOVA model
    
    Parameters
    ----------
    y : array (n_cases, n_tests)
        Dependent Measurement.
    x : array (n_cases, n_betas)
        model matrix.
    xsinv : array (n_betas, n_cases)
        xsinv for regression.
    f_map : array (n_fs, n_tests)
        container for output.
    effects : array (n_effects, 2)
        For each effect, indicating the first index in betas and df.
    e_ms : array (n_effects, n_effects)
        Each row represents the expected MS of one effect.
    """
    cdef unsigned long i
    cdef size_t df, i_beta, i_effect, i_effect_ms, i_start, i_stop, i_fmap, case
    cdef double v, ss, ms_denom

    cdef unsigned long n_tests = y.shape[1]
    cdef unsigned int n_cases = y.shape[0]
    cdef unsigned int n_betas = x.shape[1]
    cdef unsigned int n_effects = effects.shape[0]
    cdef double *betas = <double *>malloc(sizeof(double) * n_betas)
    cdef double *mss = <double *>malloc(sizeof(double) * n_effects)

    for i in range(n_tests):
        if zero_variance(y, i) == 1:
            for i_fmap in range(f_map.shape[0]):
                f_map[i_fmap, i] = 0
            continue

        _lm_betas(y, i, xsinv, betas)

        # find MS of effects
        for i_effect in range(n_effects):
            i_start = effects[i_effect, 0]
            df = effects[i_effect, 1]
            i_stop = i_start + df
            ss = 0
            for case in range(n_cases):
                v = 0
                for i_beta in range(i_start, i_stop):
                    v += x[case, i_beta] * betas[i_beta]
                ss += v ** 2
            mss[i_effect] = ss / df

        # compute F maps
        i_fmap = 0
        for i_effect in range(n_effects):
            ms_denom = 0
            for i_effect_ms in range(n_effects):
                if e_ms[i_effect, i_effect_ms] > 0:
                    ms_denom += mss[i_effect_ms]

            if ms_denom > 0:
                f_map[i_fmap, i] = mss[i_effect] / ms_denom
                i_fmap += 1

    free(betas)
    free(mss)


def anova_fmaps(const np.npy_float64[:,:] y,
                const np.npy_float64[:,:] x,
                const np.npy_float64[:,:] xsinv,
                np.npy_float64[:,:] f_map,
                const np.npy_int64[:,:] effects,
                int df_res):
    """Compute f-maps for a balanced ANOVA model with residuals

    Parameters
    ----------
    y : array (n_cases, n_tests)
        Dependent Measurement.
    x : array (n_cases, n_betas)
        model matrix.
    xsinv : array (n_betas, n_cases)
        xsinv for regression.
    f_map : array (n_fs, n_tests)
        container for output.
    effects : array (n_effects, 2)
        For each effect, indicating the first index in betas and df.
    df_res : int
        Df of the residuals.
    """
    cdef unsigned long i
    cdef unsigned int df, i_beta, i_effect, i_effect_ms, i_effect_beta, i_fmap, case
    cdef double v, SS, MS, MS_res, MS_den

    cdef unsigned long n_tests = y.shape[1]
    cdef unsigned int n_cases = y.shape[0]
    cdef unsigned int n_betas = x.shape[1]
    cdef unsigned int n_effects = effects.shape[0]
    cdef double *betas = <double *>malloc(sizeof(double) * n_betas)
    cdef double [:,:] values = cvarray((n_cases, n_betas), sizeof(double), "d")
    cdef double *predicted_y = <double *>malloc(sizeof(double) * n_cases)

    for i in range(n_tests):
        if zero_variance(y, i) == 1:
            for i_fmap in range(f_map.shape[0]):
                f_map[i_fmap, i] = 0
            continue

        _lm_betas(y, i, xsinv, betas)

        # expand accounted variance
        for case in range(n_cases):
            predicted_y[case] = 0
            for i_beta in range(n_betas):
                v = x[case, i_beta] * betas[i_beta]
                predicted_y[case] += v
                values[case, i_beta] = v

        # residuals
        SS = 0
        for case in range(n_cases):
            SS += (y[case, i] - predicted_y[case]) ** 2
        MS_res = SS / df_res

        # find MS of effects
        for i_effect in range(n_effects):
            i_effect_beta = effects[i_effect, 0]
            df = effects[i_effect, 1]
            SS = 0
            for case in range(n_cases):
                v = 0
                for i_beta in range(i_effect_beta, i_effect_beta + df):
                    v += values[case, i_beta]
                SS += v ** 2
            MS = SS / df
            f_map[i_effect, i] = MS / MS_res

    free(betas)


def sum_square(
        const np.npy_float64[:,:] y,
        np.npy_float64[:] out,
):
    """Compute the Sum Square of the data

    Parameters
    ----------
    y : array (n_cases, n_tests)
        Dependent Measurement.
    out : array (n_tests,)
        container for output.
    """
    cdef unsigned long i
    cdef unsigned int case
    cdef double ss

    cdef unsigned long n_tests = y.shape[1]
    cdef unsigned int n_cases = y.shape[0]

    for i in range(n_tests):
        ss = 0
        for case in range(n_cases):
            ss += y[case, i] ** 2

        out[i] = ss


def ss(
        const np.npy_float64[:,:] y,
        np.npy_float64[:] out,
):
    """Compute sum squares in the data (after subtracting the intercept)

    Parameters
    ----------
    y : array (n_cases, n_tests)
        Dependent Measurement.
    out : array (n_tests,)
        container for output.
    """
    cdef unsigned long i
    cdef unsigned int case
    cdef double mean, ss_

    cdef unsigned long n_tests = y.shape[1]
    cdef unsigned int n_cases = y.shape[0]

    for i in range(n_tests):
        # find mean
        mean = 0
        for case in range(n_cases):
            mean += y[case, i]
        mean /= n_cases

        # find SS of residuals
        ss_ = 0
        for case in range(n_cases):
            ss_ += (y[case, i] - mean) ** 2

        out[i] = ss_


cdef int zero_variance(
        const np.npy_float64[:,:] y,
        unsigned long i,
):
    """Check whether a column of y has zero variance"""
    cdef unsigned int case
    cdef double ref_value = y[0, i]

    for case in range(1, y.shape[0]):
        if y[case, i] != ref_value:
            return 0
    return 1


cdef void _lm_betas(
        const np.npy_float64[:,:] y,
        unsigned long i,
        const np.npy_float64[:,:] xsinv,
        double *betas,
):
    """Fit a linear model

    Parameters
    ----------
    y : array (n_cases, n_tests)
        Dependent Measurement.
    i : int
        Index of the test for which to calculate betas.
    xsinv : array (n_betas, n_cases)
        xsinv for x.
    betas : array (n_betas,)
        Output container.
    """
    cdef unsigned int i_beta, case
    cdef double beta

    cdef unsigned int n_cases = y.shape[0]
    cdef unsigned int df_x = xsinv.shape[0]

    # betas = xsinv * y
    for i_beta in range(df_x):
        beta = 0
        for case in range(n_cases):
            beta += xsinv[i_beta, case] * y[case, i]
        betas[i_beta] = beta


cdef double _lm_res_ss(
        const np.npy_float64[:,:] y,
        int i,
        const np.npy_float64[:,:] x,
        int df_x,
        double *betas,
):
    """Residual sum squares

    Parameters
    ----------
    y : array (n_cases, n_tests)
        Dependent Measurement.
    i : int
        Index of the test for which to calculate betas.
    x : array (n_cases, df_model)
        Model matrix.
    betas : array (n_betas,)
        Fitted regression coefficients.
    """
    cdef unsigned int case, i_beta
    cdef double predicted_y

    cdef double ss = 0
    cdef unsigned int n_cases = y.shape[0]

    for case in range(n_cases):
        predicted_y = 0
        for i_beta in range(df_x):
            predicted_y += x[case, i_beta] * betas[i_beta]
        ss += (y[case, i] - predicted_y) ** 2

    return ss


def lm_res(
        const np.npy_float64[:,:] y,
        const np.npy_float64[:,:] x,
        const np.npy_float64[:,:] xsinv,
        np.npy_float64[:,:] res,
):
    """Fit a linear model and compute the residuals

    Parameters
    ----------
    y : array (n_cases, n_tests)
        Dependent Measurement.
    x : array (n_cases, n_betas)
        Model matrix for the model.
    xsinv : array (n_betas, n_cases)
        xsinv for x.
    res : array (n_cases, n_tests)
        Container for output.
    """
    cdef unsigned long i
    cdef unsigned int i_beta, case
    cdef double predicted_y, SS_res

    cdef unsigned long n_tests = y.shape[1]
    cdef unsigned int n_cases = y.shape[0]
    cdef unsigned int df_x = xsinv.shape[0]
    cdef double *betas = <double *>malloc(sizeof(double) * df_x)

    for i in range(n_tests):
        _lm_betas(y, i, xsinv, betas)

        # predict y and find residuals
        for case in range(n_cases):
            predicted_y = 0
            for i_beta in range(df_x):
                predicted_y += x[case, i_beta] * betas[i_beta]
            res[case,i] = y[case, i] - predicted_y

    free(betas)


def lm_res_ss(
        const np.npy_float64[:,:] y,
        const np.npy_float64[:,:] x,
        const np.npy_float64[:,:] xsinv,
        np.npy_float64[:] ss,
):
    """Fit a linear model and compute the residual sum squares

    Parameters
    ----------
    y : array (n_cases, n_tests)
        Dependent Measurement.
    x : array (n_cases, n_betas)
        Model matrix for the model.
    xsinv : array (n_betas, n_cases)
        xsinv for x.
    ss : array (n_tests,)
        Container for output.
    """
    cdef unsigned long i

    cdef unsigned long n_tests = y.shape[1]
    cdef unsigned int n_cases = y.shape[0]
    cdef unsigned int df_x = xsinv.shape[0]
    cdef double *betas = <double *>malloc(sizeof(double) * df_x)

    for i in range(n_tests):
        _lm_betas(y, i, xsinv, betas)
        ss[i] = _lm_res_ss(y, i, x, df_x, betas)

    free(betas)


def t_1samp(
        const np.npy_float64[:,:] y,
        np.npy_float64[:] out,
):
    """T-values for 1-sample t-test

    Parameters
    ----------
    y : array (n_cases, n_tests)
        Dependent Measurement.
    out : array (n_tests,)
        Container for output.
    """
    cdef unsigned long i, case
    cdef double mean, denom

    cdef unsigned long n_tests = y.shape[1]
    cdef unsigned long n_cases = y.shape[0]
    cdef double div = (n_cases - 1) * n_cases

    for i in range(n_tests):
        # mean
        mean = 0
        for case in range(n_cases):
            mean += y[case, i]
        mean /= n_cases

        # variance
        denom = 0
        for case in range(n_cases):
            denom += (y[case, i] - mean) ** 2

        denom /= div
        denom **= 0.5
        if denom > 0:
            out[i] = mean / denom
        else:
            out[i] = 0


def t_1samp_perm(
        const np.npy_float64[:,:] y,
        np.npy_float64[:] out,
        const np.npy_int8[:] sign,
):
    """T-values for 1-sample t-test

    Parameters
    ----------
    y : array (n_cases, n_tests)
        Dependent Measurement.
    out : array (n_tests,)
        Container for output.
    sign : array (n_cases,)
        The randomly asigned sign for each case.
    """
    cdef unsigned long i, case
    cdef double mean, denom

    cdef unsigned long n_tests = y.shape[1]
    cdef unsigned int n_cases = y.shape[0]
    cdef double div = (n_cases - 1) * n_cases
    cdef double *case_buffer = <double *>malloc(sizeof(double) * n_cases)

    for i in range(n_tests):
        for case in range(n_cases):
            case_buffer[case] = y[case, i] * sign[case]

        # mean
        mean = 0
        for case in range(n_cases):
            mean += case_buffer[case]
        mean /= n_cases

        # variance
        denom = 0
        for case in range(n_cases):
            denom += (case_buffer[case] - mean) ** 2

        denom /= div
        denom **= 0.5
        if denom > 0:
            out[i] = mean / denom
        else:
            out[i] = 0


def t_ind(
        const np.npy_float64[:,:] y,
        np.npy_float64[:] out,
        const np.npy_int8[:] group,
):
    "Indpendent-samples t-test, assuming equal variance"
    cdef unsigned long i, case
    cdef double mean0, mean1, var

    cdef unsigned long n_tests = y.shape[1]
    cdef unsigned long n_cases = y.shape[0]
    cdef unsigned long df = n_cases - 2
    cdef unsigned long n0 = 0
    cdef unsigned long n1 = 0

    if group.shape[0] != n_cases:
        raise ValueError("length of group does not match n_cases in y")

    for case in range(n_cases):
        if group[case]:
            n1 += 1
    n0 = n_cases - n1

    cdef double var_mult = (1. / n0 + 1. / n1) / df

    for i in range(n_tests):
        mean0 = 0.
        mean1 = 0.
        var = 0.

        # means
        for case in range(n_cases):
            if group[case]:
                mean1 += y[case, i]
            else:
                mean0 += y[case, i]
        mean0 /= n0
        mean1 /= n1

        # variance
        for case in range(n_cases):
            if group[case]:
                var += (y[case, i] - mean1) ** 2
            else:
                var += (y[case, i] - mean0) ** 2
        if var == 0:
            out[i] = 0
            continue
        out[i] = (mean1 - mean0) / (var * var_mult) ** 0.5


def has_zero_variance(const np.npy_float64[:,:] y):
    "True if any data-columns have zero variance"
    cdef double value
    cdef unsigned long case, i

    for i in range(y.shape[1]):
        value = y[0, i]
        for case in range(1, y.shape[0]):
            if y[case, i] != value:
                break
        else:
            return True
    return False
