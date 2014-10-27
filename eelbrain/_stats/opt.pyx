# optimized statistics functions
#cython: boundscheck=False, wraparound=False

cimport cython
from cython.view cimport array as cvarray
from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as np

ctypedef np.uint32_t NP_UINT32
ctypedef fused scalar:
    cython.int
    cython.long
    cython.longlong
    cython.float
    cython.double


def merge_labels(unsigned int [:,:] cmap, int n_labels_in,
                 unsigned int [:,:] edges):
    """Merge adjacent labels with non-standard connectivity

    Parameters
    ----------
    cmap : array of int, ndim=2
        Array with labels, labels are merged along the second axis; cmap is
        modified in-place.
    n_labels_in : int
        Number of labels in cmap.
    edges : array of int (n_edges, 2)
        Edges of the connectivity graph.
    """
    n_labels_in += 1

    cdef unsigned int slice_i, i, dst_i
    cdef unsigned int n_vert = cmap.shape[0]
    cdef unsigned int n_slices = cmap.shape[1]
    cdef unsigned int n_edges = edges.shape[0]
    cdef unsigned int label, connected_label, src, dst
    cdef unsigned int relabel_src, relabel_dst

    cdef unsigned int* relabel = <unsigned int*> malloc(sizeof(unsigned int) * n_labels_in)
    for i in range(n_labels_in):
        relabel[i] = i

    # find targets for relabeling
    for slice_i in range(n_slices):
        for i in range(n_edges):
            src = edges[i, 0]
            label = cmap[src, slice_i]
            if label == 0:
                continue
            dst = edges[i, 1]
            connected_label = cmap[dst, slice_i]
            if connected_label == 0:
                continue

            while relabel[label] < label:
                label = relabel[label]

            while relabel[connected_label] < connected_label:
                connected_label = relabel[connected_label]

            if label > connected_label:
                relabel_src = label
                relabel_dst = connected_label
            else:
                relabel_src = connected_label
                relabel_dst = label

            relabel[relabel_src] = relabel_dst

    # find lowest labels
    cdef int n_labels_out = -1
    for i in range(n_labels_in):
        relabel_dst = relabel[i]
        if relabel_dst == i:
            n_labels_out += 1
        else:
            while relabel[relabel_dst] < relabel_dst:
                relabel_dst = relabel[relabel_dst]
            relabel[i] = relabel_dst

    # relabel cmap
    for i in range(n_vert):
        for slice_i in range(n_slices):
            label = cmap[i, slice_i]
            if label != 0:
                relabel_dst = relabel[label]
                if relabel_dst != label:
                    cmap[i, slice_i] = relabel_dst

    # find all label ids in cmap
    out = np.empty(n_labels_out, dtype=np.uint32)
    cdef unsigned int [:] label_ids = out
    dst_i = 0
    for i in range(1, n_labels_in):
        if i == relabel[i]:
            label_ids[dst_i] = i
            dst_i += 1

    free(relabel)
    return out


def anova_full_fmaps(scalar[:, :] y, double[:, :] x, double[:, :] xsinv,
                     double[:, :] f_map, np.int16_t[:, :] effects, 
                     np.int8_t[:, :] e_ms):
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
    cdef unsigned int df, i_beta, i_effect, i_effect_ms, i_start, i_stop, i_fmap, case
    cdef double v, ss, ms_denom

    cdef unsigned long n_tests = y.shape[1]
    cdef unsigned int n_cases = y.shape[0]
    cdef unsigned int n_betas = x.shape[1]
    cdef unsigned int n_effects = effects.shape[0]
    cdef double *betas = <double *>malloc(sizeof(double) * n_betas)
    cdef double *mss = <double *>malloc(sizeof(double) * n_effects)

    for i in range(n_tests):
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


def anova_fmaps(scalar[:, :] y, double[:, :] x, double[:, :] xsinv,
                double[:, :] f_map, np.int16_t[:, :] effects, int df_res):
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


def sum_square(scalar[:,:] y, double[:] out):
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


def ss(scalar[:,:] y, double[:] out):
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


cdef void _lm_betas(scalar[:,:] y, unsigned long i, double[:,:] xsinv,
                    double *betas) nogil:
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
    df_x : int
        Degrees of freedom of the model (n_betas).
    n_cases : int
        Number of cases in y.
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


cdef double _lm_res_ss(scalar[:,:] y, int i, double[:,:] x, int df_x,
                       double *betas) nogil:
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


def lm_betas(scalar[:,:] y, double[:,:] x, double[:,:] xsinv, double[:,:] out):
    """Fit a linear model

    Parameters
    ----------
    y : array (n_cases, n_tests)
        Dependent Measurement.
    x : array (n_cases, n_betas)
        Model matrix for the model.
    xsinv : array (n_betas, n_cases)
        xsinv for x.
    out : array (n_coefficients, n_tests)
        Container for output.
    """
    cdef unsigned long i
    cdef unsigned int i_beta

    cdef unsigned long n_tests = y.shape[1]
    cdef unsigned int df_x = xsinv.shape[0]
    cdef double *betas = <double *>malloc(sizeof(double) * df_x)

    for i in range(n_tests):
        _lm_betas(y, i, xsinv, betas)
        for i_beta in range(df_x):
            out[i_beta, i] = betas[i_beta]

    free(betas)


def lm_res(scalar[:,:] y, double[:,:] x, double[:, :] xsinv, double[:,:] res):
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


def lm_res_ss(scalar[:,:] y, double[:,:] x, double[:,:] xsinv, double[:] ss):
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


def lm_t(scalar[:,:] y, double[:,:] x, double[:,:] xsinv, double[:] a, double[:,:] out):
    """T-values for linear multiple regression

    Parameters
    ----------
    y : array (n_cases, n_tests)
        Dependent Measurement.
    x : array (n_cases, df_model)
        Model matrix.
    xsinv : array (df_model, n_cases)
        xsinv for x.
    out : array (df_model, n_tests)
        Container for output.
    """
    cdef unsigned long i, i_beta
    cdef double ss_res, ms_res, se_res

    cdef unsigned long n_tests = y.shape[1]
    cdef unsigned int n_cases = y.shape[0]
    cdef unsigned int df_x = xsinv.shape[0]
    cdef double df_res = n_cases - df_x
    cdef double *betas = <double *>malloc(sizeof(double) * df_x)

    for i in range(n_tests):
        _lm_betas(y, i, xsinv, betas)
        ss_res = _lm_res_ss(y, i, x, df_x, betas)
        ms_res = ss_res / df_res
        se_res = ms_res ** 0.5
        for i_beta in range(df_x):
            out[i_beta, i] = betas[i_beta] * a[i_beta] / se_res

    free(betas)


def t_1samp(scalar[:,:] y, double[:] out):
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
    cdef unsigned int n_cases = y.shape[0]
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
        out[i] = mean / denom


def t_1samp_perm(scalar[:,:] y, double[:] out, np.int8_t[:] sign):
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
        out[i] = mean / denom
