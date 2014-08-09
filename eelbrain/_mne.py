from itertools import izip
from math import ceil, log
import os
import re

import numpy as np
import scipy as sp
from scipy.spatial.distance import cdist

import mne
from mne import minimum_norm as mn
from mne.label import Label
from mne.source_space import label_src_vertno_sel
from mne.utils import get_subjects_dir

from ._data_obj import (ascategorial, asepochs, isfactor, isinteraction,
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
    from mne.label import _n_colors

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
                      color=color).fill(source_space)
        labels.append(label)

    return labels


def morph_source_space(ndvar, subject_to, morph_mat=None, vertices_to=None):
    """Morph source estimate between subjects using a precomputed morph-matrix

    Parameters
    ----------
    ndvar : NDVar
        NDVar with SourceSpace dimension.
    subject_to : string
        Name of the subject on which to morph.
    morph_mat : None | sparse matrix
        The morphing matrix. If ndvar contains a whole source space, the morph
        matrix can be automatically loaded, although providing a cached matrix
        can speed up processing by a second or two.
    vertices_to : None | list of array of int
        The vertices on the destination subject's brain. If ndvar contains a
        whole source space, vertices_to can be automatically loaded, although
        providing them as argument can speed up processing by a second or two.

    Returns
    -------
    morphed_ndvar : NDVar
        NDVar morphed to the destination subject.
    """
    src = ndvar.source.src
    subject_from = ndvar.source.subject
    subjects_dir = ndvar.source.subjects_dir
    vertices_from = ndvar.source.vertno
    if vertices_to is None:
        path = SourceSpace._src_pattern.format(subjects_dir=subjects_dir,
                                               subject=subject_to, src=src)
        src_to = mne.read_source_spaces(path)
        vertices_to = [src_to[0]['vertno'], src_to[1]['vertno']]
    elif not isinstance(vertices_to, list) or not len(vertices_to) == 2:
        raise ValueError('vertices_to must be a list of length 2')

    if morph_mat is None:
        morph_mat = mne.compute_morph_matrix(subject_from, subject_to,
                                             vertices_from, vertices_to, None,
                                             subjects_dir)
    elif not sp.sparse.issparse(morph_mat):
        raise ValueError('morph_mat must be a sparse matrix')
    elif not sum(len(v) for v in vertices_to) == morph_mat.shape[0]:
        raise ValueError('morph_mat.shape[0] must match number of vertices in '
                         'vertices_to')

    # flatten data
    axis = ndvar.get_axis('source')
    x = ndvar.x
    if axis != 0:
        x = x.swapaxes(0, axis)
    n_sources = len(x)
    if not n_sources == morph_mat.shape[1]:
        raise ValueError('ndvar source dimension length must be the same as '
                         'morph_mat.shape[0]')
    if ndvar.ndim > 2:
        shape = x.shape
        x = x.reshape((n_sources, -1))

    # apply morph matrix
    x_ = morph_mat * x

    # restore data shape
    if ndvar.ndim > 2:
        shape_ = (len(x_),) + shape[1:]
        x_ = x_.reshape(shape_)
    if axis != 0:
        x_ = x_.swapaxes(axis, 0)

    # package output NDVar
    if ndvar.source.parc is None:
        parc = None
    else:
        parc = ndvar.source.parc.name
    source = SourceSpace(vertices_to, subject_to, src, subjects_dir, parc)
    dims = ndvar.dims[:axis] + (source,) + ndvar.dims[axis + 1:]
    info = ndvar.info.copy()
    out = NDVar(x_, dims, info, ndvar.name)
    return out


def source_induced_power(epochs='epochs', x=None, ds=None, src='ico-4',
                         label=None, sub=None, inv=None, subjects_dir=None,
                         frequencies='4:40:0.1', *args, **kwargs):
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
    frequencies : str | array_like
        Array of frequencies of interest. A 'low:high' string is interpreted as
        logarithmically increasing range.
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

    # find frequencies
    if isinstance(frequencies, basestring):
        m = re.match("(\d+):(\d+):([\d.]+)", frequencies)
        if not m:
            raise ValueError("Invalid frequencies parameter: %r" % frequencies)
        low = log(float(m.group(1)))
        high = log(float(m.group(2)))
        step = float(m.group(3))
        frequencies = np.e ** np.arange(low, high, step)
    else:
        frequencies = np.asarray(frequencies)

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
        source = SourceSpace(vertices, subject, src, subjects_dir, None)
        dims = (source, frequency, time)
    else:
        dims = (frequency, time)

    if x is None:
        cells = (None,)
    else:
        cells = x.cells
    shape = (len(cells),) + tuple(len(dim) for dim in dims)
    dims = ('case',) + dims
    p = np.empty(shape)
    pl = np.empty(shape)
    for i, cell in enumerate(cells):
        if cell is None:
            epochs_ = epochs
        else:
            idx = (x == cell)
            epochs_ = epochs[idx]

        p_, pl_ = mn.source_induced_power(epochs_, inv, frequencies, label,
                                          *args, **kwargs)
        if src_fun is None:
            p[i] = p_
            pl[i] = pl_
        else:
            src_fun(p_, axis=0, out=p[i])
            src_fun(pl_, axis=0, out=pl[i])

    out = Dataset()
    out['power'] = NDVar(p, dims)
    out['phase_locking'] = NDVar(pl, dims)
    if x is None:
        pass
    elif isfactor(x):
        out[x.name] = Factor(cells)
    elif isinteraction(x):
        for i, name in enumerate(x.cell_header):
            out[name] = Factor((cell[i] for cell in cells))
    else:
        raise TypeError("x=%s" % repr(x))
    return out


# label operations ---

def dissolve_label(labels, source, targets, subjects_dir=None,
                   hemis=('lh', 'rh')):
    """
    Assign every point from source to the target that is closest to it.

    Parameters
    ----------
    labels : list
        List of labels (as returned by mne.read_annot).
    source : str
        Name of the source label (without hemi affix).
    targets : list of str
        List of target label names (without hemi affix).

    Notes
    -----
    Modifies ``labels`` in-place, returns None.
    """
    subjects_dir = get_subjects_dir(subjects_dir)
    subject = labels[0].subject

    idx = {l.name: i for i, l in enumerate(labels)}

    rm = set()
    for hemi in hemis:
        fpath = os.path.join(subjects_dir, subject, 'surf', hemi + '.inflated')
        points, _ = mne.read_surface(fpath)

        src_name = '-'.join((source, hemi))
        src_idx = idx[src_name]
        rm.add(src_idx)
        src_label = labels[src_idx]
        tgt_names = ['-'.join((name, hemi)) for name in targets]
        tgt_idxs = [idx[name] for name in tgt_names]
        tgt_labels = [labels[i] for i in tgt_idxs]
        tgt_points = [points[label.vertices] for label in tgt_labels]

        vert_by_tgt = {i: [] for i in xrange(len(targets))}
        for src_vert in src_label.vertices:
            point = points[src_vert:src_vert + 1]
            dist = [cdist(point, pts).min() for pts in tgt_points]
            tgt = np.argmin(dist)
            vert_by_tgt[tgt].append(src_vert)

        for i, label in enumerate(tgt_labels):
            new_vertices = vert_by_tgt[i]
            label.vertices = np.union1d(label.vertices, new_vertices)

    for i in sorted(rm, reverse=True):
        del labels[i]


def rename_label(labels, old, new):
    """Rename a label in a parcellation

    Parameters
    ----------
    labels : list of Label
        The labels
    old, new : str
        Old and new names without hemi affix.

    Notes
    -----
    Modifies ``labels`` in-place, returns None.
    """
    delim = '-'
    for hemi in ('lh', 'rh'):
        old_ = delim.join((old, hemi))
        for label in labels:
            if label.name == old_:
                new_ = delim.join((new, hemi))
                label.name = new_
