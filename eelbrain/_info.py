# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Info dictionary

Main entries
------------

meas
    Measurement:

    B
        Magnetic field strength (MEG).
    V
        Voltage (EEG).
    p
        Probability (statistics).
    r, t, f
        Statistic (correlation, t- and f- values).

contours
    Default plotting argument.
cmap
    Default plotting argument.
vmin, vmax
    Default plotting argument.


Used by GUIs
------------

bad_channels
    Bad channels.


Internally used
---------------

cmap ticks
    Non-standard cmap-ticks; used by plot.brain.p_map to transmit proper tick
    labels for remapped p-values
"""
import numpy as np


# Key constants for info dictionaries
BAD_CHANNELS = 'bad_channels'


###########
# Constants
###########

_unit_fmt = {
    1: "%s",
    1e-3: "m%s",
    1e-6: r"$\mu$%s",
    1e-9: "n%s",
    1e-12: "p%s",
    1e-15: "f%s",
}


#####################
# Generate info dicts
#####################

def default_info(meas, **kwargs):
    "Default colorspace info"
    kwargs['meas'] = meas
    return kwargs


def cluster_pmap_info():
    return {
        'meas': 'p',
        'cmap': 'sig',
        'vmax': 0.05,
        'contours': {0.05: (0., 0., 0.)},
    }


def sig_info(p=.05, contours={.01: '.5', .001: '0'}):
    "Info dict for significance map"
    return {
        'meas': 'p',
        'cmap': 'sig',
        'vmax': p,
        'contours': contours,
    }


def stat_info(meas, c0=None, c1=None, c2=None, tail=0, **kwargs):
    if 'contours' not in kwargs:
        contours = kwargs['contours'] = {}
        if c0 is not None:
            if tail >= 0:
                contours[c0] = (1.0, 0.5, 0.1)
            if tail <= 0:
                contours[-c0] = (0.5, 0.1, 1.0)
        if c1 is not None:
            if tail >= 0:
                contours[c1] = (1.0, 0.9, 0.2)
            if tail <= 0:
                contours[-c1] = (0.9, 0.2, 1.0)
        if c2 is not None:
            if tail >= 0:
                contours[c2] = (1.0, 1.0, 0.8)
            if tail <= 0:
                contours[-c2] = (1.0, 0.8, 1.0)

    if meas == 'r':
        info = {'meas': meas, 'cmap': 'RdBu_r'}
    elif meas == 't':
        info = {'meas': meas, 'cmap': 'RdBu_r'}
    elif meas == 'f':
        info = {'meas': meas, 'cmap': 'BuPu_r', 'vmin': 0}
    else:
        info = default_info(meas)
    info.update(kwargs)
    return info


def eeg_info(vmax=None, mult=1, unit='V', meas="V"):
    unit = _unit_fmt[1 / mult] % unit
    out = dict(cmap='xpolar', meas=meas, unit=unit)
    if vmax is not None:
        out['vmax'] = vmax
    return out


def meg_info(vmax=None, mult=1, unit='T', meas="B"):
    unit = _unit_fmt[1 / mult] % unit
    out = dict(cmap='xpolar', meas=meas, unit=unit)
    if vmax is not None:
        out['vmax'] = vmax
    return out


###################
# Update info dicts
###################

def set_plot_args(info, args={'cmap': 'jet'}, copy=True):
    """Update the plotting arguments in info to reflect a new colorspace

    Parameters
    ----------
    info : dict
        The previous info dictionary.
    args : dict
        The new colorspace info dictionary.
    copy : bool
        Make a copy of the dictionary before modifying it.

    Returns
    -------
    info : dict
        The updated dictionary.
    """
    if copy:
        info = info.copy()
    # remove
    for key in ('meas', 'unit', 'cmap', 'vmin', 'vmax', 'contours'):
        if key in info and key not in args:
            info.pop(key)
    info.update(args)
    return info


#######################
# operate on info dicts
#######################

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
        return all(_values_equal(a_, b_) for a_, b_ in zip(a, b))
    elif isinstance(a, dict):
        if a.keys() == b.keys():
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
