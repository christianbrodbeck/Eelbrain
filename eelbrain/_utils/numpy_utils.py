# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>

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
