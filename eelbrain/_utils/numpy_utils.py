# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from distutils.version import LooseVersion
import numpy as np


full_slice = slice(None)


def slice_to_arange(s, length):
    """Convert a slice into a numerical index

    Parameters
    ----------
    s : slice
        Slice.
    length : int
        Length of the target for the index (only needed if ``slice.stop`` is
        None).

    Returns
    -------
    arange : array of int
        Numerical index equivalent to slice.
    """
    if s.start is None:
        start = 0
    else:
        start = s.start

    if s.stop is None:
        stop = length
    else:
        stop = s.stop

    return np.arange(start, stop, s.step)


# pre numpy 0.10, digitize requires 1d-array
if LooseVersion(np.__version__) < LooseVersion('1.10'):
    def digitize(x, bins, right=False):
        if np.isscalar(x):
            return np.digitize(np.atleast_1d(x), bins, right)[0]
        elif x.ndim != 1:
            raise NotImplementedError("digitize for pre 1.10 numpy with ndim > "
                                      "1 array")
        return np.digitize(x, bins, right)
else:
    digitize = np.digitize
