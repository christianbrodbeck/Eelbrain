"""NDVar operations"""
from collections import defaultdict

import numpy as np
import mne

from ._data_obj import NDVar, UTS, Ordered
from . import _colorspaces as cs


def concatenate(ndvars, dim='time', name=None):
    """Concatenate multiple NDVars

    Parameters
    ----------
    ndvars : NDVar | sequence of NDVar
        NDVars to be concatenated. Can also be a single NDVar with ``case``
        dimension to concatenate the different cases.
    dim : str
        Dimension along which to concatenate (only 'time' and 'case' are
        implemented).
    name : str (optional)
        Name the NDVar holding the result.

    Returns
    -------
    ndvar : NDVar
        NDVar with concatenated data. For ``dim='time'``, the output time axis
        starts at t=0.
    """
    ndvar = ndvars[0]
    axis = ndvar.get_axis(dim)
    dim_names = ndvar.get_dimnames((None,) * axis + (dim,) +
                                   (None,) * (ndvar.ndim - axis - 1))
    x = np.concatenate([v.get_data(dim_names) for v in ndvars], axis)
    if dim == 'time':
        out_dim = UTS(0, ndvar.time.tstep, x.shape[axis])
    elif dim == 'case':
        out_dim = 'case'
    else:
        raise NotImplementedError("dim=%s is not implemented; only 'time' and "
                                  "'case' are implemented" % repr(dim))
    dims = ndvar.dims[:axis] + (out_dim,) + ndvar.dims[axis + 1:]
    return NDVar(x, dims, {}, name or ndvar.name)


def cwt_morlet(y, freqs, use_fft=True, n_cycles=3.0, zero_mean=False,
               out='magnitude'):
    """Time frequency decomposition with Morlet wavelets (mne-python)

    Parameters
    ----------
    y : NDVar with time dimension
        Signal.
    freqs : scalar | array
        Frequency/ies of interest. For a scalar, the output will not contain a
        frequency dimension.
    use_fft : bool
        Compute convolution with FFT or temporal convolution.
    n_cycles: float | array of float
        Number of cycles. Fixed number or one per frequency.
    zero_mean : bool
        Make sure the wavelets are zero mean.
    out : 'complex' | 'magnitude' | 'phase'
        Format of the data in the returned NDVar.

    Returns
    -------
    tfr : NDVar
        Time frequency decompositions.
    """
    from mne.time_frequency.tfr import cwt_morlet

    if not y.get_axis('time') == y.ndim - 1:
        raise NotImplementedError
    x = y.x
    x = x.reshape((np.prod(x.shape[:-1]), x.shape[-1]))
    Fs = 1. / y.time.tstep
    if np.isscalar(freqs):
        freqs = [freqs]
        fdim = None
    else:
        fdim = Ordered("frequency", freqs, 'Hz')
        freqs = fdim.values
    x = cwt_morlet(x, Fs, freqs, use_fft, n_cycles, zero_mean)
    if out == 'magnitude':
        x = np.abs(x)
    elif out == 'complex':
        pass
    else:
        raise ValueError("out = %r" % out)

    new_shape = y.x.shape[:-1]
    dims = y.dims[:-1]
    if fdim is not None:
        new_shape += (len(freqs),)
        dims += (fdim,)
    new_shape += y.x.shape[-1:]
    dims += y.dims[-1:]

    x = x.reshape(new_shape)
    info = cs.set_info_cs(y.info, cs.default_info('A'))
    return NDVar(x, dims, info, y.name)


def neighbor_correlation(x, dim='sensor', obs='time', name=None):
    """Calculate Neighbor correlation

    Parameters
    ----------
    x : NDVar
        The data.
    dim : str
        Dimension over which to correlate neighbors (default 'sensor').
    obs : str
        Dimension which provides observations over which to compute the
        correlation (default 'time').
    name : str
        Name for the new NDVar.

    Returns
    -------
    correlation : NDVar
        NDVar that contains for each element in ``dim`` the with average
        correlation coefficient with its neighbors.
    """
    dim_obj = x.get_dim(dim)

    # find neighbors
    neighbors = defaultdict(list)
    for a, b in dim_obj.connectivity():
        neighbors[a].append(b)
        neighbors[b].append(a)

    # for each point, find the average correlation with its neighbors
    data = x.get_data((dim, obs))
    cc = np.corrcoef(data)
    y = np.empty(len(dim_obj))
    for i in xrange(len(dim_obj)):
        y[i] = np.mean(cc[i, neighbors[i]])

    info = cs.set_info_cs(x.info, cs.stat_info('r'))
    return NDVar(y, (dim_obj,), info, name)


def resample(data, sfreq, npad=100, window='boxcar'):
    """Resample an NDVar along the 'time' dimension with appropriate filter

    Parameters
    ----------
    data : NDVar
        Ndvar which should be resampled.
    sfreq : scalar
        New sampling frequency.
    npad : int
        Number of samples to use at the beginning and end for padding.
    window : string | tuple
        See :func:`scipy.signal.resample` for description.

    Notes
    -----
    Uses :func:`mne.filter.resample`.
    """
    axis = data.get_axis('time')
    old_sfreq = 1.0 / data.time.tstep
    x = mne.filter.resample(data.x, sfreq, old_sfreq, npad, axis, window)
    tstep = 1. / sfreq
    time = UTS(data.time.tmin, tstep, x.shape[axis])
    dims = data.dims[:axis] + (time,) + data.dims[axis + 1:]
    return NDVar(x, dims=dims, info=data.info, name=data.name)
