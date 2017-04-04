# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import numpy as np
from scipy.ndimage import generate_binary_structure, label


VALID_TYPES = {'none', 'grid', 'custom'}


class Connectivity(object):
    """N-dimensional connectivity"""
    __slots__ = ('struct', 'custom')

    def __init__(self, dims, parc=None):
        types = tuple(getattr(dim, '_connectivity_type', 'grid') for dim in dims)
        invalid = set(types).difference(VALID_TYPES)
        if invalid:
            raise RuntimeError("Invalid connectivity type: %s" %
                               (', '.join(invalid),))

        # custom connectivity
        self.custom = {}
        n_custom = types.count('custom')
        if n_custom > 1:
            raise NotImplementedError("More than one axis with custom connectivity")
        elif n_custom:
            axis = types.index('custom')
            if axis > 0:
                raise NotImplementedError(
                    "Custom connectivity on axis other than first")
            custom_dim = dims[axis]
            if custom_dim.name == parc:
                edges = custom_dim.connectivity(disconnect_parc=True)
            else:
                edges = custom_dim.connectivity()
            dim_length = len(custom_dim)
            src = edges[:, 0]
            n_edges = np.bincount(src, minlength=dim_length)
            edge_stop = np.cumsum(n_edges)
            edge_start = edge_stop - n_edges
            self.custom[axis] = (edges, edge_start, edge_stop)

        # prepare struct for grid connectivity
        self.struct = generate_binary_structure(len(dims), 1)
        for i, ctype in enumerate(types):
            if ctype != 'grid':
                self.struct[(slice(None),) * i + (slice(None, None, 2),)] = False

    def __getstate__(self):
        return {k: getattr(self, k) for k in self.__slots__}

    def __setstate__(self, state):
        for k, v in state.iteritems():
            setattr(self, k, v)
