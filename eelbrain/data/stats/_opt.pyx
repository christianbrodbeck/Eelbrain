# optimized statistics functions
cimport cython
import numpy as np
cimport numpy as np

ctypedef np.uint32_t NP_UINT32


@cython.boundscheck(False)
def merge_labels(np.ndarray[NP_UINT32, ndim=2] cmap,  int n_labels_in, dict conn):
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
    cdef np.ndarray[NP_UINT32, ndim=1] relabel = np.arange(n_labels_in, 
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
    cdef np.ndarray[NP_UINT32, ndim=1] label_ids = np.unique(cmap)
    if label_ids[0] == 0:
        label_ids = label_ids[1:]
    return label_ids 
