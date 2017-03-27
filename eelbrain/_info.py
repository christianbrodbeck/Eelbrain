# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Info dictionary

Internally used entries:

cmap ticks
    Non-standard cmap-ticks; used by plot.brain.p_map to transmit proper tick
    labels for remapped p-values
"""
from itertools import izip

import numpy as np


# Key constants for info dictionaries
BAD_CHANNELS = 'bad_channels'


def _values_equal(a, b):
    "Test equality, taking into account array values"
    if a is b:
        return True
    elif type(a) is not type(b):
        return False
    a_iterable = np.iterable(a)
    b_iterable = np.iterable(b)
    if a_iterable != b_iterable:
        return False
    elif not a_iterable:
        return a == b
    elif len(a) != len(b):
        return False
    elif isinstance(a, np.ndarray):
        if a.shape == b.shape:
            return (a == b).all()
        else:
            return False
    elif isinstance(a, (tuple, list)):
        return all(_values_equal(a_, b_) for a_, b_ in izip(a, b))
    elif isinstance(a, dict):
        if a.viewkeys() == b.viewkeys():
            return all(_values_equal(a[k], b[k]) for k in a)
        else:
            return False
    else:
        return a == b


def merge_info(items):
    "Merge info dicts from several objects"
    info0 = items[0].info
    other_infos = [i.info for i in items[1:]]
    # find shared keys
    info_keys = set(info0.keys())
    for info in other_infos:
        info_keys.intersection_update(info.keys())
    # find shared values
    out = {}
    for key in info_keys:
        v0 = info0[key]
        if all(_values_equal(info[key], v0) for info in other_infos):
            out[key] = v0
    return out
