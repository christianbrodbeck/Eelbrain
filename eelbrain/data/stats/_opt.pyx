# optimized statistics functions
cimport cython
from cython.view cimport array as cvarray
import numpy as np
cimport numpy as np

ctypedef np.uint32_t NP_UINT32


@cython.boundscheck(False)
def merge_labels(np.ndarray[NP_UINT32, ndim=2] cmap, int n_labels_in, dict conn):
    """Merge adjacent labels with non-standard connectivity

    Parameters
    ----------
    cmap : array of int, ndim=2
        Array with labels, labels are merged along the second axis; cmap is
        modified in-place.
    n_labels_in : int
        Number of labels in cmap.
    conn : dict
        Connectivity, as a {src_i: [dst_i_1, dst_i_2, ...]} dict.
    """
    n_labels_in += 1

    cdef unsigned int slice_i, i, dst_i
    cdef unsigned int n_vert = cmap.shape[0]
    cdef unsigned int n_slices = cmap.shape[1]
    cdef NP_UINT32 label, connected_label
    cdef NP_UINT32 relabel_src, relabel_dst
    cdef np.ndarray[NP_UINT32, ndim = 1] relabel = np.arange(n_labels_in,
                                                           dtype=np.uint32)

    # find targets for relabeling
    for slice_i in range(n_slices):
        for i in conn:
            label = cmap[i, slice_i]
            if label == 0:
                continue

            for dst_i in conn[i]:
                connected_label = cmap[dst_i, slice_i]
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
    for i in range(n_labels_in):
        relabel_dst = relabel[i]
        if relabel_dst < i:
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
    cdef np.ndarray[NP_UINT32, ndim = 1] label_ids = np.unique(cmap)
    if label_ids[0] == 0:
        label_ids = label_ids[1:]
    return label_ids


@cython.boundscheck(False)
def _anova_full_fmaps(double[:, :] y, double[:, :] x, double[:, :] xsinv,
                      double[:, :] f_map, np.int16_t[:, :] effects, 
                      np.int8_t[:, :] e_ms):
    """Compute f-maps for a balanced, fully specified ANOVA model
    
    Parameters
    ----------
    y : array (n_cases, n_tests)
        Dependent Measurement.
    x : array (n_cases, n_betas)
        model matrix.
    xsinv : array
        xsinv for regression.
    f_map : array (n_fs, n_tests)
        container for output.
    effects : array (n_effects, 2)
        For each effect, indicating the first index in betas and df.
    e_ms : array (n_effects, n_effects)
        Each row represents the expected MS of one effect.
    """
    cdef int i, i_beta, i_effect, i_effect_ms, i_effect_beta, i_fmap, case
    cdef int df
    cdef double v, SS, MS_den

    cdef int n_tests = y.shape[1]
    cdef int n_cases = y.shape[0]
    cdef int n_betas = x.shape[1]
    cdef int n_effects = effects.shape[0]
    cdef double [:] betas = cvarray((n_betas,), sizeof(double), 'd')
    cdef double [:] MSs = cvarray((n_effects,), sizeof(double), 'd')

    for i in range(n_tests):
        # betas (xsinv * y)
        for i_beta in range(n_betas):
            betas[i_beta] = 0
            for case in range(n_cases):
                betas[i_beta] += xsinv[i_beta, case] * y[case, i]
        
        # find MS of effects
        for i_effect in range(n_effects):
            i_effect_beta = effects[i_effect, 0]
            df = effects[i_effect, 1]
            SS = 0
            for case in range(n_cases):
                v = 0
                for i_beta in range(i_effect_beta, i_effect_beta + df):
                    v += x[case, i_beta] * betas[i_beta]
                SS += v ** 2
            MSs[i_effect] = SS / df

        # compute F maps
        i_fmap = 0
        for i_effect in range(n_effects):
            MS_den = 0
            for i_effect_ms in range(n_effects):
                if e_ms[i_effect, i_effect_ms] > 0:
                    MS_den += MSs[i_effect_ms]

            if MS_den > 0:
                f_map[i_fmap, i] = MSs[i_effect] / MS_den
                i_fmap += 1
                    
                    
@cython.boundscheck(False)
def _anova_fmaps(double[:, :] y, double[:, :] x, double[:, :] xsinv,
                 double[:, :] f_map, np.int16_t[:, :] effects, int df_res):
    """Compute f-maps for a balanced ANOVA model with residuals

    Parameters
    ----------
    y : array (n_cases, n_tests)
        Dependent Measurement.
    x : array (n_cases, n_betas)
        model matrix.
    xsinv : array
        xsinv for regression.
    f_map : array (n_fs, n_tests)
        container for output.
    effects : array (n_effects, 2)
        For each effect, indicating the first index in betas and df.
    df_res : int
        Df of the residuals.
    """
    cdef int i, i_beta, i_effect, i_effect_ms, i_effect_beta, i_fmap, case
    cdef int df
    cdef double v, SS, MS, MS_res, MS_den

    cdef int n_tests = y.shape[1]
    cdef int n_cases = y.shape[0]
    cdef int n_betas = x.shape[1]
    cdef int n_effects = effects.shape[0]
    cdef double [:] betas = cvarray((n_betas,), sizeof(double), 'd')
    cdef double [:,:] values = cvarray((n_cases, n_betas), sizeof(double), "d")
    cdef double [:] predicted_y = cvarray((n_cases,), sizeof(double), 'd')

    for i in range(n_tests):
        # betas (xsinv * y)
        for i_beta in range(n_betas):
            betas[i_beta] = 0
            for case in range(n_cases):
                betas[i_beta] += xsinv[i_beta, case] * y[case, i]
        
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
