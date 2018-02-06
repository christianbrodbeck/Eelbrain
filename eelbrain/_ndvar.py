# -*- coding: utf-8 -*-
# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""NDVar operations

NDVar methods should work on all NDVars; This module defines operations that
depend on the presence of specific dimensions as functions, as well as
operations that operate on more than one NDVar.
"""
from collections import defaultdict
from copy import copy
from itertools import izip, repeat
from math import floor
from numbers import Real

import mne
import numpy as np
from scipy import linalg, signal

from . import mne_fixes
from . import _colorspaces as cs
from ._data_obj import (
    NDVar, Case, Categorial, Dimension, Scalar, SourceSpace, UTS,
    asndvar, combine)
from ._exceptions import DimensionMismatchError
from ._info import merge_info
from ._stats.connectivity import Connectivity
from ._stats.connectivity import find_peaks as _find_peaks
from ._trf._boosting_opt import l1


def concatenate(ndvars, dim='time', name=None, tmin=0):
    """Concatenate multiple NDVars

    Parameters
    ----------
    ndvars : NDVar | sequence of NDVar
        NDVars to be concatenated. Can also be a single NDVar with ``case``
        dimension to concatenate the different cases.
    dim : str | Dimension
        Either a string specifying an existsing dimension along which to
        concatenate, or a Dimension object to create a new dimension.
    name : str (optional)
        Name the NDVar holding the result.
    tmin : scalar | 'first'
        Time axis start, only applies when ``dim == 'time'``; default is 0.
        Set ``tmin='first'`` to use ``tmin`` of ``ndvars[0]``.

    Returns
    -------
    ndvar : NDVar
        NDVar with concatenated data.
    """
    try:
        ndvar = ndvars[0]
    except TypeError:
        ndvars = tuple(ndvars)
        ndvar = ndvars[0]

    if isinstance(ndvars, NDVar):
        info = ndvars.info
    else:
        info = merge_info(ndvars)

    if dim is Case or (isinstance(dim, basestring) and dim == 'case'):
        n = sum(1 if not v.has_case else len(v) for v in ndvars)
        dim = Case(n)

    if isinstance(dim, Dimension):
        dim_names = (ndvar.dimnames[:ndvar.has_case] + (np.newaxis,) +
                     ndvar.dimnames[ndvar.has_case:])
        x = np.concatenate([v.get_data(dim_names) for v in ndvars],
                           int(ndvar.has_case))
        dims = ndvar.dims[:ndvar.has_case] + (dim,) + ndvar.dims[ndvar.has_case:]
    else:
        axis = ndvar.get_axis(dim)
        dim_names = ndvar.get_dimnames((None,) * axis + (dim,) +
                                       (None,) * (ndvar.ndim - axis - 1))
        x = np.concatenate([v.get_data(dim_names) for v in ndvars], axis)
        dim_obj = ndvar.dims[axis]
        if isinstance(dim_obj, UTS):
            if isinstance(tmin, basestring):
                if tmin == 'first':
                    tmin = ndvar.time.tmin
                else:
                    raise ValueError("tmin=%r" % (tmin,))
            out_dim = UTS(tmin, ndvar.time.tstep, x.shape[axis])
        elif isinstance(dim_obj, Case):
            out_dim = Case
        elif isinstance(dim_obj, SourceSpace):
            if (len(ndvars) != 2 or
                        ndvars[0].source.rh_n != 0 or
                        ndvars[1].source.lh_n != 0):
                raise NotImplementedError(
                    "Can only concatenate NDVars along source space with "
                    "exactly two NDVars, one for lh and one for rh (in this "
                    "order)")
            lh = ndvars[0].source
            rh = ndvars[1].source
            if lh.subject != rh.subject:
                raise ValueError("NDVars not from the same subject (%s/%s)" %
                                 (lh.subject, rh.subject))
            elif lh.src != rh.src:
                raise ValueError("NDVars have different source-spaces (%s/%s)" %
                                 (lh.src, rh.src))
            elif lh.subjects_dir != rh.subjects_dir:
                raise ValueError("NDVars have different subjects_dirs (%s/%s)" %
                                 (lh.subjects_dir, rh.subjects_dir))
            # parc
            if lh.parc is None or rh.parc is None:
                parc = None
            else:
                parc = combine((lh.parc, rh.parc))

            out_dim = SourceSpace([lh.lh_vertices, rh.rh_vertices], lh.subject,
                                  lh.src, lh.subjects_dir, parc)
        elif isinstance(dim_obj, Scalar):
            out_dim = Scalar._concatenate(v.get_dim(dim) for v in ndvars)
        else:
            raise NotImplementedError(
                "concatenate() is not implemented for concatenating along %s "
                "dimensions" % (dim_obj.__class__.__name__,))
        dims = ndvar.dims[:axis] + (out_dim,) + ndvar.dims[axis + 1:]
    return NDVar(x, dims, info, name or ndvar.name)


def convolve(h, x):
    """Convolve ``h`` and ``x`` along the time dimension

    Parameters
    ----------
    h : NDVar | sequence of NDVar
        Kernel.
    x : NDVar | sequence of NDVar
        Data to convolve, corresponding to ``h``.

    Returns
    -------
    y : NDVar
        Convolution, with same time dimension as ``x``.
    """
    if isinstance(h, NDVar):
        if not isinstance(x, NDVar):
            raise TypeError("If h is an NDVar, x also needs to be an NDVar "
                            "(got x=%r)" % (x,))

        if h.ndim == 1:
            ht = h.get_dim('time')
            hdim = None
        elif h.ndim == 2:
            hdim, ht = h.get_dims((None, 'time'))
        else:
            raise NotImplementedError("h must be 1 or 2 dimensional, got h=%r" % h)

        if x.ndim == 1:
            xt = x.get_dim('time')
            xdim = None
        elif x.ndim == 2:
            xdim, xt = x.get_dims((None, 'time'))
            if xdim != hdim:
                raise ValueError("h %s dimension and x %s dimension do not "
                                 "match" % (hdim.name, xdim.name))
        else:
            raise NotImplementedError("x must be 1 or 2 dimensional, got x=%r" % h)

        if ht.tstep != xt.tstep:
            raise ValueError(
                "h and x need to have same time-step (got h.time.tstep=%s, "
                "x.time.tstep=%s)" % (ht.tstep, xt.tstep))

        if h.ndim == 1:
            data = np.convolve(h.x, x.x)
            dims = (xt,)
        elif x.ndim == 1:
            xdata = x.get_data('time')
            data = np.array(tuple(np.convolve(h_i, xdata) for h_i in
                                  h.get_data((hdim.name, 'time'))))
            dims = (hdim, xt)
        else:
            data = None
            for h_i, x_i in izip(h.get_data((hdim.name, 'time')),
                                 x.get_data((xdim.name, 'time'))):
                if data is None:
                    data = np.convolve(h_i, x_i)
                else:
                    data += np.convolve(h_i, x_i)
            dims = (xt,)

        # full convolution -> decide which slice of data corresponds to x
        # i.e., if kernel tmin is positive, add zero-padding
        i_start = -int(round(ht.tmin / ht.tstep))
        i_stop = i_start + xt.nsamples
        if i_start < 0:
            if data.ndim == 2:
                pad = np.zeros((len(hdim), -i_start))
            else:
                pad = np.zeros(-i_start)
            data = np.concatenate((pad, data[..., :i_stop]), -1)
        elif i_stop > data.shape[-1]:
            if data.ndim == 2:
                pad = np.zeros((len(hdim), i_stop - data.shape[-1]))
            else:
                pad = np.zeros(i_stop - data.shape[-1])
            data = np.concatenate((data[..., i_start:], pad), -1)
        else:
            data = data[..., i_start: i_stop]

        return NDVar(data, dims, x.info.copy(), x.name)
    else:
        out = None
        for h_, x_ in izip(h, x):
            if out is None:
                out = convolve(h_, x_)
            else:
                out += convolve(h_, x_)
        return out


def cross_correlation(in1, in2, name="{in1} * {in2}"):
    """Cross-correlation between two NDVars along the time axis
    
    Parameters
    ----------
    in1 : NDVar  (time,)
        First NDVar.
    in2 : NDVar  (time,)
        Second NDVar.
    name : str  
        Name for the new NDVar.
        
    Returns
    -------
    cross_correlation : NDVar  (time,)
        Cross-correlation between ``in1`` and ``in2``, with a time axis 
        reflecting time shift.
    """
    x1 = in1.get_data(('time',))
    x2 = in2.get_data(('time',))
    in1_time = in1.get_dim('time')
    in2_time = in2.get_dim('time')
    tstep = in1_time.tstep
    if in2_time.tstep != tstep:
        raise ValueError("in1 and in2 need to have the same tstep, got %s and "
                         "%s" % (tstep, in2_time.tstep))
    nsamples = in1_time.nsamples + in2_time.nsamples - 1
    in1_i0 = -(in1_time.tmin / tstep)
    in2_i0 = -(in2_time.tmin / tstep)
    in2_rel_i0 = in2_i0 - in2_time.nsamples
    out_i0 = in1_i0 - in2_rel_i0 - 1
    tmin = -out_i0 * tstep
    time = UTS(tmin, tstep, nsamples)
    x_corr = signal.correlate(x1, x2)
    return NDVar(x_corr, (time,), merge_info((in1, in2)),
                 name.format(in1=in1.name, in2=in2.name))


def cwt_morlet(y, freqs, use_fft=True, n_cycles=3.0, zero_mean=False,
               out='magnitude', decim=1):
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
    out : 'complex' | 'magnitude'
        Format of the data in the returned NDVar.

    Returns
    -------
    tfr : NDVar
        Time frequency decompositions.
    """
    if out == 'magnitude':
        magnitude_out = True
        out = 'power'
    elif out not in ('complex', 'phase', 'power'):
        raise ValueError("out=%r" % (out,))
    else:
        magnitude_out = False
    dimnames = y.get_dimnames((None,) * (y.ndim - 1) + ('time',))
    data = y.get_data(dimnames)
    dims = y.get_dims(dimnames)
    data_flat = data.reshape((1, np.prod(data.shape[:-1]), data.shape[-1]))
    time_dim = dims[-1]
    sfreq = 1. / time_dim.tstep
    if np.isscalar(freqs):
        freqs = [freqs]
        fdim = None
    else:
        fdim = Scalar("frequency", freqs, 'Hz')
        freqs = fdim.values

    x_flat = mne.time_frequency.tfr_array_morlet(
        data_flat, sfreq, freqs, n_cycles, zero_mean, use_fft, decim, out)

    out_shape = list(data.shape)
    out_dims = list(dims)
    if fdim is not None:
        out_shape.insert(-1, len(fdim))
        out_dims.insert(-1, fdim)
    if decim != 1:
        n_times = x_flat.shape[-1]
        out_shape[-1] = n_times
        out_dims[-1] = UTS(time_dim.tmin, time_dim.tstep * decim, n_times)
    x = x_flat.reshape(out_shape)
    if magnitude_out:
        x **= 0.5
    info = cs.set_info_cs(y.info, cs.default_info('A'))
    return NDVar(x, out_dims, info, y.name)


def dss(ndvar):
    u"""Denoising source separation (DSS)

    Parameters
    ----------
    ndvar : NDVar (case, dim, time)
        Data to decompose. DSS is performed over the case and time dimensions.

    Returns
    -------
    to_dss : NDVar (dss, dim)
        Transform data to DSS.
    from_dss : NDVar (dim, dss)
        Reconstruct data form DSS.

    Notes
    -----
    the method is described in  [1]_. This function uses the implementation from
    the `mne-sandbox <https://github.com/mne-tools/mne-sandbox>`_.
    
    References
    ----------
    .. [1] de Cheveigné, A., & Simon, J. Z. (2008). Denoising based on spatial 
        filtering. Journal of Neuroscience Methods, 171(2), 331–339. 
        `10.1016/j.jneumeth.2008.03.015 
        <https://doi.org/10.1016/j.jneumeth.2008.03.015>`_

    """
    dim_names = ndvar.get_dimnames(('case', None, 'time'))
    x = ndvar.get_data(dim_names)
    data_dim = ndvar.get_dim(dim_names[1])
    dss_mat = mne_fixes.dss(x, return_data=False)

    # if data is rank-deficient, dss_mat is not square
    a, b = dss_mat.shape
    if a != b:
        raise ValueError("Data for DSS is rank-deficient. Did you exclude all "
                         "bad channels?")

    n_comp = len(dss_mat)
    dss_dim = Scalar('dss', np.arange(n_comp))
    to_dss = NDVar(dss_mat, (dss_dim, data_dim), {}, 'to dss')
    from_dss = NDVar(linalg.inv(dss_mat), (data_dim, dss_dim), {}, 'from dss')
    return to_dss, from_dss


def filter_data(ndvar, l_freq, h_freq, filter_length='auto',
                l_trans_bandwidth='auto', h_trans_bandwidth='auto',
                method='fir', iir_params=None, phase='zero',
                fir_window='hamming', fir_design=None):
    """Apply :func:`mne.filter.filter_data` to an NDVar

    Returns
    -------
    filtered_ndvar : NDVar
        NDVar with same dimensions as ``ndvar`` and filtered data.
    """
    axis = ndvar.get_axis('time')
    if axis == ndvar.ndim:
        axis = None
        data = ndvar.x
    else:
        data = ndvar.x.swapaxes(axis, -1)
    sfreq = 1. / ndvar.time.tstep

    x = mne.filter.filter_data(
        data, sfreq, l_freq, h_freq, None, filter_length, l_trans_bandwidth,
        h_trans_bandwidth, 1, method, iir_params, True, phase, fir_window,
        fir_design
    )

    if axis is not None:
        x = x.swapaxes(axis, -1)
    return NDVar(x, ndvar.dims, ndvar.info.copy(), ndvar.name)


def find_intervals(ndvar, interpolate=False):
    """Find intervals from a boolean NDVar

    Parameters
    ----------
    ndvar : boolean NDVar (time,)
        Data which to convert to intervals.
    interpolate : bool
        By default, ``start`` values reflect the first sample that is ``True``
        and ``stop`` values reflect the first sample that is ``False``. With
        ``interpolate=True``, time points are shifted half a sample to the
        left. This is desirable for example when marking regions in a plot.

    Returns
    -------
    intervals : iterator over tuples
        Intervals represented as ``(start, stop)`` tuples
    """
    if ndvar.dimnames != ('time',):
        raise DimensionMismatchError("Requires NDVar with time dimension only,"
                                     "got %r" % (ndvar,))
    # make sure we have a boolean NDVar
    if ndvar.x.dtype.kind != 'b':
        ndvar = ndvar != 0
    ndvar = ndvar.astype(int)

    diff = np.diff(ndvar.x)
    if ndvar.x[0]:
        diff = np.append(1, diff)
    else:
        diff = np.append(0, diff)

    onsets = ndvar.time.times[diff == 1]
    offsets = ndvar.time.times[diff == -1]
    if ndvar.x[-1]:
        offsets = np.append(offsets, ndvar.time.tstop)

    if interpolate:
        shift = -ndvar.time.tstep / 2.
        onsets += shift
        offsets += shift

    return tuple(izip(onsets, offsets))


def find_peaks(ndvar):
    """Find local maxima in an NDVar
    
    Parameters
    ----------
    ndvar : NDVar
        Data in which to find peaks.
    
    Returns
    -------
    peaks : NDVar of bool
        NDVar that is ``True`` at local maxima.
    """
    for custom_ax, dim in enumerate(ndvar.dims):
        if getattr(dim, '_connectivity_type', None) == 'custom':
            break
    else:
        custom_ax = 0

    if custom_ax:
        x = ndvar.x.swapaxes(custom_ax, 0)
        dims = list(ndvar.dims)
        dims[custom_ax], dims[0] = dims[0], dims[custom_ax]
    else:
        x = ndvar.x
        dims = ndvar.dims

    connectivity = Connectivity(dims)
    peak_map = _find_peaks(x, connectivity)
    if custom_ax:
        peak_map = peak_map.swapaxes(custom_ax, 0)
    return NDVar(peak_map, ndvar.dims, {}, ndvar.name)


def frequency_response(b, frequencies=None):
    """Frequency response for a FIR filter

    Parameters
    ----------
    b : NDVar  (..., time, ...)
        FIR Filter.
    frequencies : int | array_like
        Number of frequencies at which to compute the response, or array with
        exact frequencies in Hz.

    Returns
    -------
    frequency_response : NDVar  (..., frequency)
        Frequency response for each filter in ``b``.
    """
    time = b.get_dim('time')
    dimnames = b.get_dimnames(last='time')
    data = b.get_data(dimnames)
    if frequencies is None or isinstance(frequencies, int):
        wor_n = None
    else:
        if isinstance(frequencies, Scalar):
            freqs = frequencies.values
        else:
            freqs = np.asarray(frequencies)
        wor_n = freqs * (2 * np.pi * time.tstep)
    orig_shape = data.shape[:-1]
    data_flat = data.reshape((-1, len(time)))
    fresps = []
    for h in data_flat:
        freqs, fresp = signal.freqz(h, worN=wor_n)
        fresps.append(fresp)
    freqs_hz = freqs / (2 * np.pi * time.tstep)
    frequency = Scalar('frequency', freqs_hz, 'Hz')
    fresps = np.array(fresps).reshape(orig_shape + (-1,))
    dims = b.get_dims(dimnames[:-1]) + (frequency,)
    return NDVar(fresps, dims, b.info.copy(), b.name)


def label_operator(labels, operation='mean', exclude=None, weights=None,
                   dim_name='label', dim_values=None):
    """Convert labeled NDVar into a matrix operation to extract label values
    
    Parameters
    ----------
    labels : NDVar of int
        NDVar in which each label corresponds to a unique integer.
    operation : 'mean' | 'sum'
        Whether to extract the label mean or sum.
    exclude : array_like
        Values to exclude (i.e., use ``exclude=0`` to ignore the area where 
        ``labels==0``.
    weights : NDVar
        NDVar with same dimension as ``labels`` to assign weights to label 
        elements.
    dim_name : str
        Name for the dimension characterized by labels (default ``"label"``).
    dim_values : dict
        Dictionary mapping label ids (i.e., values in ``labels``) to values on 
        the dimension characterized by labels. If values are strings the new 
        dimension will be categorical, if values are scalar it will be Scalar.
        The default values are the integers in ``labels``.
    
    Returns
    -------
    m : NDVar
        Label operator, ``m.dot(data)`` extracts label mean/sum.
    """
    if operation not in ('mean', 'sum'):
        raise ValueError("operation=%r" % (operation,))
    dimname = labels.get_dimnames((None,))[0]
    dim = labels.get_dim(dimname)
    if weights is not None:
        if weights.get_dim(dimname) != dim:
            raise DimensionMismatchError("weights.{0} does not correspond to "
                                         "labels.{0}".format(dimname))
        weights = weights.get_data((dimname,))
    label_data = labels.get_data((dimname,))
    label_values = np.unique(label_data)
    if exclude is not None:
        label_values = np.setdiff1d(label_values, exclude)
    # out-dim
    if dim_values is None:
        label_dim = Scalar(dim_name, label_values)
    else:
        values = tuple(dim_values[i] for i in label_values)
        if all(isinstance(v, basestring) for v in values):
            label_dim = Categorial(dim_name, values)
        elif all(isinstance(v, Real) for v in values):
            label_dim = Scalar(dim_name, values)
        else:
            raise TypeError("If dim_values is specified, values mus be either "
                            "all strings or all real numbers; got %r" %
                            (dim_values,))
    # construct operator
    x = np.empty((len(label_values), len(dim)))
    for v, xs in izip(label_values, x):
        np.equal(label_data, v, xs)
        if weights is not None:
            xs *= weights
        if operation == 'mean':
            xs /= l1(xs, np.array(((0, len(xs)),), np.int64))
    return NDVar(x, (label_dim, dim), {}, labels.name)


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
    x = asndvar(x)
    low_var = x.std(obs).x < 1e-25
    dim_obj = x.get_dim(dim)
    if np.any(low_var):
        raise ValueError("Low variance at %s = %s" %
                         (dim, dim_obj._dim_index(low_var)))

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
    return NDVar(y, (dim_obj,), info, name or x.name)


def psd_welch(ndvar, fmin=0, fmax=np.inf, n_fft=256, n_overlap=0, n_per_seg=None):
    """Power spectral density with Welch's method

    Parameters
    ----------
    ndvar : NDVar  (..., time, ...)
        Data with time dimension.
    fmin : scalar
        Lower bound of the frequencies of interest.
    fmax : scalar
        Upper bound of the frequencies of interest.
    n_fft : int
        Length of the FFT in samples (default 256).
    n_overlap : int
        Overlap between segments in samples (default 0).
    n_per_seg : int | None
        Length of each Welch segment in samples. Smaller ``n_per_seg`` result
        in smoother PSD estimates (default ``n_fft``).

    Returns
    -------
    psd : NDVar  (..., frequency)
        Power spectral density.

    Notes
    -----
    Uses :func:`mne.time_frequency.psd_array_welch` implementation.
    """
    time_ax = ndvar.get_axis('time')
    dims = list(ndvar.dims)
    del dims[time_ax]
    last_ax = ndvar.ndim - 1
    data = ndvar.x
    if time_ax != last_ax:
        data = data.rollaxis(time_ax, last_ax + 1)
    sfreq = 1. / ndvar.time.tstep
    psds, freqs = mne.time_frequency.psd_array_welch(
        data, sfreq, fmin, fmax, n_fft=n_fft, n_overlap=n_overlap,
        n_per_seg=n_per_seg)
    dims.append(Scalar("frequency", freqs, 'Hz'))
    return NDVar(psds, dims, ndvar.info.copy(), ndvar.name)


def rename_dim(ndvar, old_name, new_name):
    """Rename an NDVar dimension

    Parameters
    ----------
    ndvar : NDVar
        NDVar on which a dimension should be renamed.
    old_name : str
        Current name of the dimension.
    new_name : str
        New name for the dimension.

    Returns
    -------
    ndvar_out : NDVar
        Shallow copy of ``ndvar`` with the dimension ``old_name`` renamed to
        ``new_name``.

    Notes
    -----
    This is needed when creating an NDVar with two instances of the same
    dimension (for example, point spread functions, mapping :class:`SourceSpace`
    to :class:`SourceSpace`).
    """
    axis = ndvar.get_axis(old_name)
    old_dim = ndvar.dims[axis]
    new_dim = copy(old_dim)
    dims = list(ndvar.dims)
    new_dim.name = new_name
    dims[axis] = new_dim
    return NDVar(ndvar.x, dims, ndvar.info.copy(), ndvar.name)


def resample(ndvar, sfreq, npad=100, window='none', name=None):
    """Resample an NDVar along the 'time' dimension with appropriate filter

    Parameters
    ----------
    ndvar : NDVar
        Ndvar which should be resampled.
    sfreq : scalar
        New sampling frequency.
    npad : int
        Number of samples to use at the beginning and end for padding.
    window : string | tuple
        See :func:`scipy.signal.resample` for description.
    name : str
        Name for the new NDVar (default is ``ndvar.name``).

    Notes
    -----
    By default (``window='none'``) this function uses
    :func:`scipy.signal.resample` directly. If ``window`` is set to a different
    value, the more sophisticated but slower :func:`mne.filter.resample` is
    used.

    This function can be very slow when the number of time samples is uneven
    (see :func:`scipy.signal.resample`).
    """
    if name is None:
        name = ndvar.name
    axis = ndvar.get_axis('time')
    if window == 'none':
        new_tstep = 1. / sfreq
        new_num = int(floor((ndvar.time.tstop - ndvar.time.tmin) / new_tstep))
        # crop input data
        new_duration = new_tstep * new_num
        old_num = int(round(new_duration / ndvar.time.tstep))
        if old_num == ndvar.time.nsamples:
            x = ndvar.x
        else:
            idx = (slice(None),) * axis + (slice(None, old_num),)
            x = ndvar.x[idx]
        # resamples
        x = signal.resample(x, new_num, axis=axis)
        dims = (ndvar.dims[:axis] +
                (UTS(ndvar.time.tmin, new_tstep, new_num),) +
                ndvar.dims[axis + 1:])
        return NDVar(x, dims, ndvar.info.copy(), name)
    old_sfreq = 1.0 / ndvar.time.tstep
    x = mne.filter.resample(ndvar.x, sfreq, old_sfreq, npad, axis, window)
    new_tstep = 1.0 / sfreq
    time = UTS(ndvar.time.tmin, new_tstep, x.shape[axis])
    dims = ndvar.dims[:axis] + (time,) + ndvar.dims[axis + 1:]
    return NDVar(x, dims, ndvar.info, name)


class Filter(object):
    "Filter and downsample"
    def __init__(self, sfreq=None):
        self.sfreq = sfreq

    def __repr__(self):
        args = self._repr_args()
        if self.sfreq:
            args += ', sfreq=%i' % self.sfreq
        return '%s(%s)' % (self.__class__.__name__, args)

    def _repr_args(self):
        raise NotImplementedError

    def _get_b_a(self, tstep):
        raise NotImplementedError

    def __eq__(self, other):
        return self.sfreq == other.sfreq

    def filter(self, ndvar):
        """Filter an NDVar"""
        b, a = self._get_b_a(ndvar.time.tstep)
        if not np.all(np.abs(np.roots(a)) < 1):
            raise ValueError("Filter unstable")
        x = signal.lfilter(b, a, ndvar.x, ndvar.get_axis('time'))
        out = NDVar(x, ndvar.dims, ndvar.info.copy(), ndvar.name)
        if self.sfreq:
            return resample(out, self.sfreq)
        else:
            return out

    def filtfilt(self, ndvar):
        """Forward-backward filter an NDVar"""
        b, a = self._get_b_a(ndvar.time.tstep)
        if not np.all(np.abs(np.roots(a)) < 1):
            raise ValueError("Filter unstable")
        x = signal.filtfilt(b, a, ndvar.x, ndvar.get_axis('time'))
        out = NDVar(x, ndvar.dims, ndvar.info.copy(), ndvar.name)
        if self.sfreq:
            return resample(out, self.sfreq)
        else:
            return out


class Butterworth(Filter):
    """Butterworth filter

    Parameters
    ----------
    low : scalar
        Low cutoff frequency.
    high : scalar | None
        High cutoff frequency.
    order : int
        Filter order.
    sfreq : scalar
        Downsample filtered signal to this sampling frequency.

    Notes
    -----
    Uses :func:`scipy.signal.butter`.
    """
    def __init__(self, low, high, order, sfreq=None):
        if not low and not high:
            raise ValueError("Neither low nor high set")
        self.low = low
        self.high = high
        self.order = int(order)
        Filter.__init__(self, sfreq)

    def _repr_args(self):
        return '%s, %s, %i' % (self.low, self.high, self.order)

    def __eq__(self, other):
        return (Filter.__eq__(self, other) and self.low == other.low and
                self.high == other.high and self.order == other.order)

    def _get_b_a(self, tstep):
        nyq = 1. / tstep / 2.
        if self.low and self.high:
            return signal.butter(self.order, (self.low / nyq, self.high / nyq),
                                 'bandpass')
        elif self.low:
            return signal.butter(self.order, self.low / nyq, 'highpass')
        elif self.high:
            return signal.butter(self.order, self.high / nyq, 'lowpass')
        else:
            raise ValueError("Neither low nor high set")


def segment(continuous, times, tstart, tstop, decim=1):
    """Segment a continuous NDVar

    Parameters
    ----------
    continuous : NDVar
        NDVar with a continuous time axis.
    times : sequence of scalar
        Times for which to extract segments.
    tstart : scalar
        Start time for segments.
    tstop : scalar
        Stop time for segments.
    decim : int
        Decimate data after segmenting by factor ``decim`` (the default is
        ``1``, i.e. no decimation).

    Returns
    -------
    segmented_data : NDVar
        NDVar with all data segments corresponding to ``times``, stacked along
        the ``case`` axis.
    """
    if continuous.has_case:
        raise ValueError("Continuous data can't have case dimension")
    axis = continuous.get_axis('time')
    tstep = None if decim == 1 else continuous.time.tstep * decim
    segments = [continuous.sub(time=(t + tstart, t + tstop, tstep)) for
                t in times]

    s0_time = segments[0].time
    dims = (('case',) +
            continuous.dims[:axis] +
            (UTS(tstart, s0_time.tstep, s0_time.nsamples),) +
            continuous.dims[axis + 1:])
    return NDVar(np.array(tuple(s.x for s in segments)), dims,
                 continuous.info.copy(), continuous.name)


def set_parc(ndvar, parc, dim='source'):
    """Change the parcellation of an :class:`NDVar` with SourceSpace dimension

    Parameters
    ----------
    ndvar : NDVar
        NDVar with SourceSpace dimension.
    parc : None | str | Factor
        Parcellation to identify source space vertex locations.
        Can be specified as Factor assigning a label to each source, or a
        string specifying a FreeSurfer parcellation (stored as ``*.annot``
        files with the MRI).
    dim : str
        Name of the dimension to operate on (usually 'source', the default).

    Returns
    -------
    out_ndvar : NDVar
        Shallow copy of ``ndvar`` with the source space parcellation set to
        ``parc``.
    """
    axis = ndvar.get_axis(dim)
    old = ndvar.dims[axis]
    new = SourceSpace(old.vertices, old.subject, old.src, old.subjects_dir,
                      parc, old._subgraph(), dim)
    dims = list(ndvar.dims)
    dims[axis] = new
    return NDVar(ndvar.x, dims, ndvar.info.copy(), ndvar.name)


def set_tmin(ndvar, tmin=0.):
    """Change the time axis of an :class:`NDVar`

    Parameters
    ----------
    tmin : scalar
        New ``tmin`` value (default 0).

    Returns
    -------
    out_ndvar : NDVar
        Shallow copy of ``ndvar`` with updated time axis.
    """
    axis = ndvar.get_axis('time')
    old = ndvar.dims[axis]
    dims = list(ndvar.dims)
    dims[axis] = UTS(tmin, old.tstep, old.nsamples)
    return NDVar(ndvar.x, dims, ndvar.info.copy(), ndvar.name)
