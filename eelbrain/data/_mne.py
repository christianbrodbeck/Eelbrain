from itertools import izip
from math import ceil

import numpy as np

from mne.label import _n_colors, Label
from mne.source_space import label_src_vertno_sel
from mne import minimum_norm as mn

from .data_obj import (ascategorial, asepochs, isfactor, isinteraction,
                       Dataset, Factor, NDVar, Ordered, SourceSpace, UTS)


def labels_from_clusters(clusters, names=None):
    """Create Labels from source space clusters

    Parameters
    ----------
    clusters : NDVar
        NDVar which is non-zero on the cluster. Can have a case dimension.
    names : None | list of str | str
        Label names corresponding to clusters (default is "clusterX").

    Returns
    -------
    labels : list of mne.Label
        One label for each cluster.
    """
    source = clusters.source
    source_space = clusters.source.get_source_space()
    subject = source.subject

    if clusters.has_case:
        n_clusters = len(clusters)
        x = clusters.x
    else:
        n_clusters = 1
        x = clusters.x[None, :]

    if isinstance(names, basestring) and n_clusters == 1:
        names = [names]
    elif names is None:
        names = ("cluster%i" % i for i in xrange(n_clusters))
    elif len(names) != n_clusters:
        err = "Number of names difference from number of clusters."
        raise ValueError(err)

    colors = _n_colors(n_clusters)
    labels = []
    for i, color, name in izip(xrange(n_clusters), colors, names):
        idx = (x[i] != 0)
        where = np.nonzero(idx)[0]
        src_in_lh = (where < source.lh_n)
        if np.all(src_in_lh):
            hemi = 'lh'
            hemi_vertices = source.lh_vertno
        elif np.any(src_in_lh):
            raise ValueError("Can't have clusters spanning both hemispheres")
        else:
            hemi = 'rh'
            hemi_vertices = source.rh_vertno
            where -= source.lh_n
        vertices = hemi_vertices[where]

        label = Label(vertices, hemi=hemi, name=name, subject=subject,
                      color=color, src=source_space)
        labels.append(label)

    return labels


_default_frequencies = list(np.e ** np.arange(2, 3.8, .1))

def source_induced_power(epochs='epochs', x=None, ds=None, src='ico-4',
                         label=None, sub=None, inv=None, subjects_dir=None,
                         frequencies=_default_frequencies, *args, **kwargs):
    """Compute source induced power and phase locking from mne Epochs

    Parameters
    ----------
    epochs : str | mne.Epochs
        Epochs with sensor space data.
    x : None | str | categorial
        Categories for which to compute power and phase locking (if None the
        grand average is used).
    ds : None | Dataset
        Dataset containing the relevant data objects.
    src : str
        How to handle the source dimension: either a source space (the one on
        which the inverse operator is based, e.g. 'ico-4') or the name of a
        numpy function that reduces the dimensionality (e.g., 'mean').
    label : Label
        Restricts the source estimates to a given label.
    sub : str | index
        Subset of Dataset rows to use.
    inv : None | dict
        The inverse operator (or None if the inverse operator is in
        ``ds.info['inv']``.
    subjects_dir : str
        subjects_dir.
    frequencies : array
        Array of frequencies of interest.
    lambda2 : float
        The regularization parameter of the minimum norm.
    method : "MNE" | "dSPM" | "sLORETA"
        Use mininum norm, dSPM or sLORETA.
    nave : int
        The number of averages used to scale the noise covariance matrix.
    n_cycles : float | array of float
        Number of cycles. Fixed number or one per frequency.
    decim : int
        Temporal decimation factor.
    use_fft : bool
        Do convolutions in time or frequency domain with FFT.
    pick_ori : None | "normal"
        If "normal", rather than pooling the orientations by taking the norm,
        only the radial component is kept. This is only implemented
        when working with loose orientations.
    baseline : None (default) or tuple of length 2
        The time interval to apply baseline correction.
        If None do not apply it. If baseline is (a, b)
        the interval is between "a (s)" and "b (s)".
        If a is None the beginning of the data is used
        and if b is None then b is set to the end of the interval.
        If baseline is equal ot (None, None) all the time
        interval is used.
    baseline_mode : None | 'logratio' | 'zscore'
        Do baseline correction with ratio (power is divided by mean
        power during baseline) or zscore (power is divided by standard
        deviation of power during baseline after subtracting the mean,
        power = [power - mean(power_baseline)] / std(power_baseline)).
    pca : bool
        If True, the true dimension of data is estimated before running
        the time frequency transforms. It reduces the computation times
        e.g. with a dataset that was maxfiltered (true dim is 64). Default is
        False.
    n_jobs : int
        Number of jobs to run in parallel.
    zero_mean : bool
        Make sure the wavelets are zero mean.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
    """
    epochs = asepochs(epochs, sub, ds)
    if x is not None:
        x = ascategorial(x, sub, ds)
    if inv is None:
        inv = ds.info.get('inv', None)
    if inv is None:
        msg = ("No inverse operator specified. Either specify the inv "
               "parameter or provide it in ds.info['inv']")
        raise ValueError(msg)

    # set pca to False
    if len(args) < 10 and 'pca' not in kwargs:
        kwargs['pca'] = False

    subject = inv['src'][0]['subject_his_id']
    if label is None:
        vertices = [inv['src'][0]['vertno'], inv['src'][1]['vertno']]
    else:
        vertices, _ = label_src_vertno_sel(label, inv['src'])

    # prepare output dimensions
    frequency = Ordered('frequency', frequencies, 'Hz')
    if len(args) >= 5:
        decim = args[4]
    else:
        decim = kwargs.get('decim', 1)
    tmin = epochs.tmin
    tstep = 1. / epochs.info['sfreq'] / decim
    nsamples = int(ceil(float(len(epochs.times)) / decim))
    time = UTS(tmin, tstep, nsamples)
    src_fun = getattr(np, src, None)
    if src_fun is None:
        source = SourceSpace(vertices, subject, src, subjects_dir)
        dims = (source, frequency, time)
    else:
        dims = (frequency, time)

    if x is None:
        p, pl = mn.source_induced_power(epochs, inv, frequencies, label,
                                        *args, **kwargs)
        if src_fun is not None:
            p = src_fun(p, axis=0)
            pl = src_fun(pl, axis=0)
    else:
        shape = (len(x.cells),) + tuple(len(dim) for dim in dims)
        dims = ('case',) + dims
        p = np.empty(shape)
        pl = np.empty(shape)
        x_ = []
        for i, cell in enumerate(x.cells):
            idx = (x == cell)
            epochs_ = epochs[idx]
            p_, pl_ = mn.source_induced_power(epochs_, inv, frequencies,
                                              label, *args, **kwargs)
            if src_fun is not None:
                p_ = src_fun(p_, axis=0)
                pl_ = src_fun(pl_, axis=0)
            p[i] = p_
            pl[i] = pl_
            x_.append(cell)

    out = Dataset()
    out['power'] = NDVar(p, dims)
    out['phase_locking'] = NDVar(pl, dims)
    if x is None:
        pass
    elif isfactor(x):
        out[x.name] = Factor(x_)
    elif isinteraction(x):
        for i, name in x.cell_header:
            out[name] = Factor((cell[i] for cell in x_))
    else:
        raise TypeError("x=%s" % repr(x))
    return out
