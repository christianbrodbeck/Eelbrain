from itertools import izip
from math import ceil, floor, log
import os
from operator import add, sub
import re

import numpy as np
import scipy as sp
from scipy.spatial.distance import cdist

import mne
from mne import minimum_norm as mn
from mne.label import Label, BiHemiLabel
from mne.source_space import label_src_vertno_sel
from mne.utils import get_subjects_dir

from ._data_obj import (ascategorial, asepochs, isfactor, isinteraction,
                        Dataset, Factor, NDVar, Ordered, SourceSpace, UTS)


def _vertices_equal(v1, v0):
    "Test whether v1 and v0 are equal"
    return np.array_equal(v1[0], v0[0]) and np.array_equal(v1[1], v0[1])


def shift_mne_epoch_trigger(epochs, trigger_shift, min_shift=None, max_shift=None):
    """Shift the trigger in an MNE Epochs object

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object.
    trigger_shift : scalar sequence
        For each event in ``epochs`` the amount of time by which to shift the
        trigger (in seconds).
    min_shift : scalar (optional)
        Minimum time shift, used to crop data (default is
        ``min(trigger_shift)``).
    max_shift : scalar (optional)
        Maximum time shift, used to crop data (default is
        ``max(trigger_shift)``).

    Returns
    -------
    shifted_epochs : mne.EpochsArray
        Epochs object in which data and timing information is shifted to set
        the time point t=0 to the new trigger position. Data is temporally
        cropped so that only time points with information on all the epochs
        are contained in ``shifted_epochs``.
    """
    data = epochs.get_data()
    tstep = 1. / epochs.info['sfreq']
    shifts = [int(round(x / tstep)) for x in trigger_shift]
    if min_shift is None:
        min_shift = min(shifts)
    else:
        min_shift = int(floor(min_shift / tstep))

    if max_shift is None:
        max_shift = max(shifts)
    else:
        max_shift = int(ceil(max_shift / tstep))

    x, y, z = data.shape
    start_offset = -min_shift
    stop_offset = z - max_shift
    new_shape = (x, y, z - (max_shift - min_shift))
    new_data = np.empty(new_shape, data.dtype)
    for i, shift in enumerate(shifts):
        new_data[i] = data[i, :, start_offset + shift:stop_offset + shift]
    tmin = epochs.tmin + (start_offset / float(epochs.info['sfreq']))
    return mne.EpochsArray(new_data, epochs.info, epochs.events, tmin,
                           epochs.event_id)


def labels_from_clusters(clusters, names=None):
    """Create Labels from source space clusters

    Parameters
    ----------
    clusters : NDVar
        NDVar which is non-zero on the cluster. Can have a case dimension, but
        does not have to.
    names : None | list of str | str
        Label names corresponding to clusters (default is "cluster%i").

    Returns
    -------
    labels : list of mne.Label
        One label for each cluster.
    """
    from mne.label import _n_colors

    if isinstance(names, basestring):
        names = [names]

    source = clusters.source
    source_space = clusters.source.get_source_space()
    subject = source.subject
    collapse = tuple(dim for dim in clusters.dimnames if dim not in ('case', 'source'))
    clusters_index = clusters.any(collapse)
    if clusters_index.has_case:
        n_clusters = len(clusters)
    else:
        n_clusters = 1
        clusters_index = (clusters_index,)

    if names is None:
        names = ("cluster%i" % i for i in xrange(n_clusters))
    elif len(names) != n_clusters:
        err = "Number of names difference from number of clusters."
        raise ValueError(err)

    colors = _n_colors(n_clusters)
    labels = []
    for cluster, color, name in izip(clusters_index, colors, names):
        lh_vertices = source.lh_vertno[cluster.x[:source.lh_n]]
        rh_vertices = source.rh_vertno[cluster.x[source.lh_n:]]
        if len(lh_vertices) and len(rh_vertices):
            lh = Label(lh_vertices, hemi='lh', name=name + '-lh',
                       subject=subject, color=color).fill(source_space)
            rh = Label(rh_vertices, hemi='rh', name=name + '-rh',
                       subject=subject, color=color).fill(source_space)
            label = BiHemiLabel(lh, rh, name, color)
        elif len(lh_vertices):
            label = Label(lh_vertices, hemi='lh', name=name + '-lh',
                          subject=subject, color=color).fill(source_space)
        elif len(rh_vertices):
            label = Label(rh_vertices, hemi='rh', name=name + '-lh',
                          subject=subject, color=color).fill(source_space)
        else:
            raise ValueError("Empty Cluster")
        labels.append(label)

    return labels


def labels_from_mni_coords(seeds, extent=30., subject='fsaverage',
                           surface='white', mask=None, subjects_dir=None,
                           parc=None):
    """Create a parcellation from seed coordinates in MNI space

    Parameters
    ----------
    seeds : dict
        Seed coordinates. Keys are label names, including -hemi tags. values
        are seeds (array_like of shape (3,) or (3, n_seeds)).
    extent : scalar
        Extent of the label in millimeters (maximum distance from the seed).
    subject : str
        MRI-subject to use (default 'fsaverage').
    surface : str
        Surface to use (default 'white').
    mask : None | str
        A parcellation used to mask the parcellation under construction.
    subjects_dir : str
        SUBJECTS_DIR.
    parc : None | str
        Name of the parcellation under construction (only used for error
        messages).
    """
    name_re = re.compile("\w+-(lh|rh)$")
    if not all(name.endswith(('lh', 'rh')) for name in seeds):
        err = ("Names need to end in 'lh' or 'rh' so that the proper "
               "hemisphere can be selected")
        raise ValueError(err)

    # load surfaces
    subjects_dir = get_subjects_dir(subjects_dir)
    fpath = os.path.join(subjects_dir, subject, 'surf', '.'.join(('%s', surface)))
    surfs = {hemi: mne.read_surface(fpath % hemi) for hemi in ('lh', 'rh')}

    # prepare seed properties for mne.grow_labels
    vertices = []
    names = []
    hemis = []
    for name, coords_ in seeds.iteritems():
        m = name_re.match(name)
        if not m:
            raise ValueError("Invalid seed name in %r parc: %r. Names must "
                             "conform to the 'xxx-lh' or 'xxx-rh' scheme."
                             % (parc, name))
        coords = np.atleast_2d(coords_)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("Invalid coordinate specification for seed %r in "
                             "parc %r: %r. Seeds need to be specified as "
                             "arrays with shape (3,) or (n_seeds, 3)."
                             % (name, parc, coords_))
        hemi = m.group(1)
        seed_verts = []
        for coord in coords:
            dist = np.sqrt(np.sum((surfs[hemi][0] - coord) ** 2, axis=1))
            seed_verts.append(np.argmin(dist))
        vertices.append(seed_verts)
        names.append(name)
        hemis.append(hemi == 'rh')

    # grow labels
    labels = mne.grow_labels(subject, vertices, extent, hemis, subjects_dir,
                             1, False, names, surface)

    # apply mask
    if mask is not None:
        mlabels = mne.read_labels_from_annot(subject, mask,
                                             subjects_dir=subjects_dir)
        unknown = {l.hemi: l for l in mlabels if l.name.startswith('unknown-')}

        for label in labels:
            rm = unknown[label.hemi]
            if np.any(np.in1d(label.vertices, rm.vertices)):
                label.vertices = np.setdiff1d(label.vertices, rm.vertices, True)

    return labels


def morph_source_space(ndvar, subject_to, vertices_to=None, morph_mat=None,
                       copy=False):
    """Morph source estimate to a different MRI subject

    Parameters
    ----------
    ndvar : NDVar
        NDVar with SourceSpace dimension.
    subject_to : string
        Name of the subject on which to morph.
    vertices_to : None | list of array of int
        The vertices on the destination subject's brain. If ndvar contains a
        whole source space, vertices_to can be automatically loaded, although
        providing them as argument can speed up processing by a second or two.
    morph_mat : None | sparse matrix
        The morphing matrix. If ndvar contains a whole source space, the morph
        matrix can be automatically loaded, although providing a cached matrix
        can speed up processing by a second or two.
    copy : bool
        Make sure that the data of ``morphed_ndvar`` is separate from
        ``ndvar`` (default False).

    Returns
    -------
    morphed_ndvar : NDVar
        NDVar morphed to the destination subject.

    Notes
    -----
    This function is used to make sure a number of different NDVars are defined
    on the same MRI subject and handles scaled MRIs efficiently. If the MRI
    subject on which ``ndvar`` is defined is a scaled copy of ``subject_to``,
    by default a shallow copy of ``ndvar`` is returned. That means that it is
    not safe to assume that ``morphed_ndvar`` can be modified in place without
    altering ``ndvar``. To make sure the date of the output is independent from
    the data of the input, set the argument ``copy=True``.
    """
    subjects_dir = ndvar.source.subjects_dir
    subject_from = ndvar.source.subject
    src = ndvar.source.src
    if vertices_to is None:
        path = SourceSpace._src_pattern.format(subjects_dir=subjects_dir,
                                               subject=subject_to, src=src)
        src_to = mne.read_source_spaces(path)
        vertices_to = [src_to[0]['vertno'], src_to[1]['vertno']]
    elif not isinstance(vertices_to, list) or not len(vertices_to) == 2:
        raise ValueError('vertices_to must be a list of length 2')

    if subject_from == subject_to and _vertices_equal(ndvar.source.vertno,
                                                      vertices_to):
        if copy:
            return ndvar.copy()
        else:
            return ndvar

    axis = ndvar.get_axis('source')
    x = ndvar.x

    # check whether it is a scaled brain
    do_morph = True
    cfg_path = os.path.join(subjects_dir, subject_from,
                            'MRI scaling parameters.cfg')
    if os.path.exists(cfg_path):
        cfg = mne.coreg.read_mri_cfg(subject_from, subjects_dir)
        subject_from = cfg['subject_from']
        if subject_to == subject_from and _vertices_equal(ndvar.source.vertno,
                                                          vertices_to):
            if copy:
                x_ = x.copy()
            else:
                x_ = x
            vertices_to = ndvar.source.vertno
            do_morph = False

    if do_morph:
        vertices_from = ndvar.source.vertno
        if morph_mat is None:
            morph_mat = mne.compute_morph_matrix(subject_from, subject_to,
                                                 vertices_from, vertices_to,
                                                 None, subjects_dir)
        elif not sp.sparse.issparse(morph_mat):
            raise ValueError('morph_mat must be a sparse matrix')
        elif not sum(len(v) for v in vertices_to) == morph_mat.shape[0]:
            raise ValueError('morph_mat.shape[0] must match number of '
                             'vertices in vertices_to')

        # flatten data
        if axis != 0:
            x = x.swapaxes(0, axis)
        n_sources = len(x)
        if not n_sources == morph_mat.shape[1]:
            raise ValueError('ndvar source dimension length must be the same '
                             'as morph_mat.shape[0]')
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
                   hemi='both'):
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
    subjects_dir : str
        subjects_dir.
    hemi : 'both', 'lh', 'rh'
        Hemisphere(s) for which to dissolve the label.

    Notes
    -----
    Modifies ``labels`` in-place, returns None.
    """
    subjects_dir = get_subjects_dir(subjects_dir)
    subject = labels[0].subject
    if hemi == 'both':
        hemis = ('lh', 'rh')
    elif hemi == 'lh' or hemi == 'rh':
        hemis = (hemi,)
    else:
        raise ValueError("hemi=%r" % hemi)

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


def combination_label(name, exp, labels):
    """Create labels based on combination of existing labels

    Parameters
    ----------
    name : str
        Name for the new label (without -hemi tag to create labels for both
        hemispheres).
    exp : str
        Boolean expression containing label names, + and - (all without -hemi
        tags).
    labels : dict
        {name: label} dictionary.

    Returns
    -------
    labels : list
        List of labels, one or two depending on what hemispheres are included.
    """
    m = re.match("([\w.]+)-([lr]h)", name)
    if m:
        name = m.group(1)
        hemis = (m.group(2),)
    else:
        hemis = ('lh', 'rh')

    out = None
    operator = None
    i = 0
    pattern = re.compile("\s*([\w.]+)\s*([+-]|$)")
    while True:
        m = pattern.match(exp, i)
        if not m:
            raise ValueError("Error parsing label expression:\n%s\n%s"
                             % (exp, ' ' * i + '^'))
        name_, next_op = m.groups()
        if out:
            for i, hemi in enumerate(hemis):
                out[i] = operator(out[i], labels['%s-%s' % (name_, hemi)])
        else:
            out = [labels['%s-%s' % (name_, hemi)] for hemi in hemis]

        if next_op == '+':
            operator = add
        elif next_op == '-':
            operator = sub
        elif next_op == '':
            break
        i = m.end()

    for i, hemi in enumerate(hemis):
        out[i].name = '%s-%s' % (name, hemi)
    return out
