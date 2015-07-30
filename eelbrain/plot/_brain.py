# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from __future__ import division
from distutils.version import LooseVersion
from itertools import izip
import os
from tempfile import mkdtemp
from warnings import warn

from nibabel.freesurfer import read_annot
import numpy as np
import mne

from .._data_obj import asndvar, NDVar
from ..fmtxt import Image, im_table, ms
from ._colors import ColorList


# defaults
FOREGROUND = (0, 0, 0)
BACKGROUND = (1, 1, 1)


def _idx(i):
    return int(round(i))


def assert_can_save_movies():
    import surfer
    if LooseVersion(surfer.__version__) <= LooseVersion('0.5'):
        raise ImportError("Saving movies requires PySurfer 0.6")


def annot(annot, subject='fsaverage', surf='smoothwm', borders=False, alpha=0.7,
          hemi=None, views=('lat', 'med'), w=None, h=None, axw=None, axh=None,
          foreground=None, background=None, parallel=True, subjects_dir=None):
    """Plot the parcellation in an annotation file

    Parameters
    ----------
    annot : str
        Name of the annotation (e.g., "PALS_B12_LOBES").
    subject : str
        Name of the subject (default 'fsaverage').
    surf : 'inflated' | 'pial' | 'smoothwm' | 'sphere' | 'white'
        Freesurfer surface to use as brain geometry.
    borders : bool | int
        Show only label borders (PySurfer Brain.add_annotation() argument).
    alpha : scalar
        Alpha of the annotation (1=opaque, 0=transparent, default 0.7).
    hemi : 'lh' | 'rh' | 'both' | 'split'
        Which hemispheres to plot (default includes hemisphere with more than one
        label in the annot file).
    views : str | iterator of str
        View or views to show in the figure. Options are: 'rostral', 'parietal',
        'frontal', 'ventral', 'lateral', 'caudal', 'medial', 'dorsal'.
    w, h, axw, axh : scalar
        Layout parameters (figure width/height, subplot width/height).
    foreground : mayavi color
        Figure foreground color (i.e., the text color).
    background : mayavi color
        Figure background color.
    parallel : bool
        Set views to parallel projection (default ``True``).
    smoothing_steps : None | int
        Number of smoothing steps if data is spatially undersampled (pysurfer
        ``Brain.add_data()`` argument).
    subjects_dir : None | str
        Override the subjects_dir associated with the source space dimension.

    Returns
    -------
    brain : surfer.Brain
        PySurfer Brain instance.
    """
    if hemi is None:
        annot_lh = mne.read_labels_from_annot(subject, annot, 'lh',
                                              subjects_dir=subjects_dir)
        use_lh = len(annot_lh) > 1
        annot_rh = mne.read_labels_from_annot(subject, annot, 'rh',
                                              subjects_dir=subjects_dir)
        use_rh = len(annot_rh) > 1
        if use_lh and use_rh:
            hemi = 'split'
        elif use_lh:
            hemi = 'lh'
        elif use_rh:
            hemi = 'rh'
        else:
            raise ValueError("Neither hemisphere contains more than one label")

    brain = _surfer_brain(None, subject, surf, hemi, views, w, h, axw, axh,
                          foreground, background, subjects_dir)
    brain.add_annotation(annot, borders, alpha)

    if parallel:
        _set_parallel(brain, surf)

    return brain


def annot_legend(lh, rh, *args, **kwargs):
    """Plot a legend for a freesurfer parcellation

    Parameters
    ----------
    lh : str
        Path to the lh annot-file.
    rh : str
        Path to the rh annot-file.

    Returns
    -------
    legend : plot.ColorList
        ColorList figure with legend for the parcellation.
    """
    _, lh_colors, lh_names = read_annot(lh)
    _, rh_colors, rh_names = read_annot(rh)
    lh_colors = dict(izip(lh_names, lh_colors[:, :4] / 255.))
    rh_colors = dict(izip(rh_names, rh_colors[:, :4] / 255.))
    names = set(lh_names)
    names.update(rh_names)
    colors = {}
    seq = []  # sequential order in legend
    seq_lh = []
    seq_rh = []
    for name in names:
        if name in lh_colors and name in rh_colors:
            if np.array_equal(lh_colors[name], rh_colors[name]):
                colors[name] = lh_colors[name]
                seq.append(name)
            else:
                colors[name + '-lh'] = lh_colors[name]
                colors[name + '-rh'] = rh_colors[name]
                seq_lh.append(name + '-lh')
                seq_rh.append(name + '-rh')
        elif name in lh_colors:
            colors[name + '-lh'] = lh_colors[name]
            seq_lh.append(name + '-lh')
        else:
            colors[name + '-rh'] = rh_colors[name]
            seq_rh.append(name + '-rh')
    return ColorList(colors, seq + seq_lh + seq_rh, *args, **kwargs)


def _plot(data, lut, vmin, vmax, *args, **kwargs):
    "Plot depending on source space kind"
    if data.source.kind == 'vol':
        return _voxel_brain(data, lut, vmin, vmax, *args, **kwargs)
    else:
        return surfer_brain(data, lut, vmin, vmax, *args, **kwargs)


def dspm(src, fmin=13, fmax=22, fmid=None, *args, **kwargs):
    """
    Plot a source estimate with coloring for dSPM values (bipolar).

    Parameters
    ----------
    src : NDVar, dims = ([case,] source, [time])
        NDVar with SourceSpace dimension. If stc contains a case dimension,
        the average across cases is taken.
    fmin, fmax : scalar >= 0
        Start- and end-point for the color gradient for positive values. The
        gradient for negative values goes from -fmin to -fmax. Values between
        -fmin and fmin are transparent.
    fmid : None | scalar
        Midpoint for the color gradient. If fmid is None (default) it is set
        half way between fmin and fmax.
    surf : 'inflated' | 'pial' | 'smoothwm' | 'sphere' | 'white'
        Freesurfer surface to use as brain geometry.
    views : str | iterator of str
        View or views to show in the figure. Options are: 'rostral', 'parietal',
        'frontal', 'ventral', 'lateral', 'caudal', 'medial', 'dorsal'.
    hemi : 'lh' | 'rh' | 'both' | 'split'
        Which hemispheres to plot (default based on data).
    colorbar : bool
        Whether to add a colorbar to the figure.
    time_label : str
        Label to show time point. Use ``'ms'`` or ``'s'`` to display time in
        milliseconds or in seconds, or supply a custom format string to format
        time values (in seconds; default is ``'ms'``).
    w, h, axw, axh : scalar
        Layout parameters (figure width/height, subplot width/height).
    foreground : mayavi color
        Figure foreground color (i.e., the text color).
    background : mayavi color
        Figure background color.
    parallel : bool
        Set views to parallel projection (default ``True``).
    smoothing_steps : None | int
        Number of smoothing steps if data is spatially undersampled (pysurfer
        ``Brain.add_data()`` argument).
    subjects_dir : None | str
        Override the subjects_dir associated with the source space dimension.

    Returns
    -------
    brain : surfer.Brain
        PySurfer Brain instance containing the plot.
    """
    if fmid is None:
        fmid = (fmax + fmin) / 2
    lut = _dspm_lut(fmin, fmid, fmax)
    return _plot(src, lut, -fmax, fmax, *args, **kwargs)


def stat(*args, **kwargs):
    warn("plot.brain.stat() has been deprecated, use plot.brain.p_map() "
         "instead", DeprecationWarning)
    return p_map(*args, **kwargs)


def p_map(p_map, param_map=None, p0=0.05, p1=0.01, solid=False, *args,
          **kwargs):
    """Plot a map of p-values in source space.

    Parameters
    ----------
    p_map : NDVar
        Statistic to plot (normally a map of p values).
    param_map : NDVar
        Statistical parameter covering the same data points as p_map. Used
        only for incorporating the directionality of the effect into the plot.
    p0, p1 : scalar
        Threshold p-values for the color map.
    solid : bool
        Use solid color patches between p0 and p1 (default: False - blend
        transparency between p0 and p1).
    surf : 'inflated' | 'pial' | 'smoothwm' | 'sphere' | 'white'
        Freesurfer surface to use as brain geometry.
    views : str | iterator of str
        View or views to show in the figure. Options are: 'rostral', 'parietal',
        'frontal', 'ventral', 'lateral', 'caudal', 'medial', 'dorsal'.
    hemi : 'lh' | 'rh' | 'both' | 'split'
        Which hemispheres to plot (default based on data).
    colorbar : bool
        Whether to add a colorbar to the figure.
    time_label : str
        Label to show time point. Use ``'ms'`` or ``'s'`` to display time in
        milliseconds or in seconds, or supply a custom format string to format
        time values (in seconds; default is ``'ms'``).
    w, h, axw, axh : scalar
        Layout parameters (figure width/height, subplot width/height).
    foreground : mayavi color
        Figure foreground color (i.e., the text color).
    background : mayavi color
        Figure background color.
    parallel : bool
        Set views to parallel projection (default ``True``).
    smoothing_steps : None | int
        Number of smoothing steps if data is spatially undersampled (pysurfer
        ``Brain.add_data()`` argument).
    subjects_dir : None | str
        Override the subjects_dir associated with the source space dimension.

    Returns
    -------
    brain : surfer.Brain
        PySurfer Brain instance containing the plot.
    """
    pmap, lut, vmax = _p_lut(p_map, param_map, p0=p0, p1=p1, solid=solid)
    return _plot(pmap, lut, -vmax, vmax, *args, **kwargs)


def activation(src, threshold=None, vmax=None, *args, **kwargs):
    "This function is deprecated. Use plot.brain.dspm() instead."
    warn("plot.brain.activation() is deprecated. Use plot.brain.dspm() "
         "instead.", DeprecationWarning)

    x = src.mean()
    std = src.std()

    if threshold is None:
        threshold = x + std
    if vmax is None:
        vmax = x + 2 * std
    lut = _activation_lut(threshold, vmax)

    return _plot(src, lut, -vmax, vmax, *args, **kwargs)


def cluster(cluster, vmax=None, *args, **kwargs):
    """Plot a spatio-temporal cluster

    Plots a cluster with the assumption that all non-zero data should be
    visible, while areas that not part of the cluster are 0.

    Parameters
    ----------
    cluster : NDVar
        The cluster.
    vmax : scalar != 0
        Maximum value in the colormap. Default is the maximum value in the
        cluster.
    surf : 'inflated' | 'pial' | 'smoothwm' | 'sphere' | 'white'
        Freesurfer surface to use as brain geometry.
    views : str | iterator of str
        View or views to show in the figure. Options are: 'rostral', 'parietal',
        'frontal', 'ventral', 'lateral', 'caudal', 'medial', 'dorsal'.
    hemi : 'lh' | 'rh' | 'both' | 'split'
        Which hemispheres to plot (default based on data).
    colorbar : bool
        Whether to add a colorbar to the figure.
    time_label : str
        Label to show time point. Use ``'ms'`` or ``'s'`` to display time in
        milliseconds or in seconds, or supply a custom format string to format
        time values (in seconds; default is ``'ms'``).
    w, h, axw, axh : scalar
        Layout parameters (figure width/height, subplot width/height).
    foreground : mayavi color
        Figure foreground color (i.e., the text color).
    background : mayavi color
        Figure background color.
    parallel : bool
        Set views to parallel projection (default ``True``).
    smoothing_steps : None | int
        Number of smoothing steps if data is spatially undersampled (pysurfer
        ``Brain.add_data()`` argument).
    subjects_dir : None | str
        Override the subjects_dir associated with the source space dimension.

    Returns
    -------
    brain : surfer.Brain
        PySurfer Brain instance containing the plot.
    """
    if vmax is None:
        vmax = max(cluster.x.max(), -cluster.x.min())
        if vmax == 0:
            raise ValueError("The cluster's data is all zeros")
    elif vmax == 0:
        raise ValueError("vmax can't be 0")
    elif vmax < 0:
        vmax = -vmax

    lut = _dspm_lut(0, vmax / 10, vmax)
    return _plot(cluster, lut, -vmax, vmax, *args, **kwargs)


def _surfer_brain(unit, subject='fsaverage', surf='smoothwm', hemi='split',
                  views=('lat', 'med'), w=None, h=None, axw=None, axh=None,
                  foreground=None, background=None, subjects_dir=None):
    """Create surfer.Brain instance

    Parameters
    ----------
    subject : str
        Name of the subject (default 'fsaverage').
    surf : 'inflated' | 'pial' | 'smoothwm' | 'sphere' | 'white'
        Freesurfer surface to use as brain geometry.
    hemi : 'lh' | 'rh' | 'both' | 'split'
        Which hemispheres to plot.
    views : str | iterator of str
        View or views to show in the figure.
    colorbar : bool
        Whether to add a colorbar to the figure.
    w, h, axw, axh : scalar
        Layout parameters (figure width/height, subplot width/height).
    foreground : mayavi color
        Figure foreground color (i.e., the text color).
    background : mayavi color
        Figure background color.
    subjects_dir : None | str
        Override the subjects_dir associated with the source space dimension.

    Returns
    -------
    brain : surfer.Brain
        PySurfer Brain instance.
    """
    from ._brain_fix import Brain

    if isinstance(views, basestring):
        views = [views]
    elif not isinstance(views, list):
        views = list(views)

    if hemi == 'split':
        n_views_x = 2
    elif hemi in ('lh', 'rh', 'both'):
        n_views_x = 1
    else:
        raise ValueError("Unknown value for hemi parameter: %s" % repr(hemi))

    title = None
    config_opts = {}
    if w is not None:
        config_opts['width'] = w
    elif axw is not None:
        config_opts['width'] = axw * n_views_x
    else:
        config_opts['width'] = 500 * n_views_x

    if h is not None:
        config_opts['height'] = h
    elif axh is not None:
        config_opts['height'] = axh * len(views)
    else:
        config_opts['height'] = 400 * len(views)

    if foreground is None:
        config_opts['foreground'] = FOREGROUND
    else:
        config_opts['foreground'] = foreground

    if background is None:
        config_opts['background'] = BACKGROUND
    else:
        config_opts['background'] = background

    brain = Brain(unit, subject, hemi, surf, True, title,
                  config_opts=config_opts, views=views,
                  subjects_dir=subjects_dir)

    return brain


def surfer_brain(src, colormap='hot', vmin=0, vmax=9, surf='smoothwm',
                 views=('lat', 'med'), hemi=None, colorbar=True,
                 time_label='ms', w=None, h=None, axw=None, axh=None,
                 foreground=None, background=None, parallel=True,
                 smoothing_steps=None, mask=True, subjects_dir=None):
    """Create a PySurfer Brain object with a data layer

    Parameters
    ----------
    src : NDVar, dims = ([case,] source, [time])
        NDVar with SourceSpace dimension. If stc contains a case dimension,
        the average across cases is taken.
    colormap :
        Colormap for PySurfer.
    vmin, vmax : scalar
        Endpoints for the colormap.
    surf : 'inflated' | 'pial' | 'smoothwm' | 'sphere' | 'white'
        Freesurfer surface to use as brain geometry.
    views : str | iterator of str
        View or views to show in the figure. Options are: 'rostral', 'parietal',
        'frontal', 'ventral', 'lateral', 'caudal', 'medial', 'dorsal'.
    hemi : 'lh' | 'rh' | 'both' | 'split'
        Which hemispheres to plot (default based on data).
    colorbar : bool
        Whether to add a colorbar to the figure.
    time_label : str
        Label to show time point. Use ``'ms'`` or ``'s'`` to display time in
        milliseconds or in seconds, or supply a custom format string to format
        time values (in seconds; default is ``'ms'``).
    w, h, axw, axh : scalar
        Layout parameters (figure width/height, subplot width/height).
    foreground : mayavi color
        Figure foreground color (i.e., the text color).
    background : mayavi color
        Figure background color.
    parallel : bool
        Set views to parallel projection (default ``True``).
    smoothing_steps : None | int
        Number of smoothing steps if data is spatially undersampled (pysurfer
        ``Brain.add_data()`` argument).
    mask : bool
        Shade areas that are not in ``src``.
    subjects_dir : None | str
        Override the subjects_dir associated with the source space dimension.

    Returns
    -------
    brain : surfer.Brain
        PySurfer Brain instance containing the plot.
    """
    src = asndvar(src)
    unit = src.info.get('unit', None)
    if src.has_case:
        src = src.summary()

    if hemi is None:
        if src.source.lh_n and src.source.rh_n:
            hemi = 'split'
        elif src.source.lh_n:
            hemi = 'lh'
        elif not src.source.rh_n:
            raise ValueError('No data')
        else:
            hemi = 'rh'
    elif (hemi == 'lh' and src.source.rh_n) or (hemi == 'rh' and src.source.lh_n):
        src = src.sub(source=hemi)

    if subjects_dir is None:
        subjects_dir = src.source.subjects_dir

    brain = _surfer_brain(unit, src.source.subject, surf, hemi, views, w, h, axw, axh,
                          foreground, background, subjects_dir)

    # general PySurfer data args
    alpha = 1
    if smoothing_steps is None and src.source.kind == 'ico':
        smoothing_steps = src.source.grade + 1

    if src.has_dim('time'):
        times = src.time.times
        data_dims = ('source', 'time')
        if time_label == 'ms':
            import surfer
            if LooseVersion(surfer.__version__) > LooseVersion('0.5'):
                time_label = lambda x: '%s ms' % int(round(x * 1000))
            else:
                times = times * 1000
                time_label = '%i ms'
        elif time_label == 's':
            time_label = '%.3f s'
    else:
        times = None
        data_dims = ('source',)

    # add data
    if src.source.lh_n:
        if hemi == 'lh':
            colorbar_ = colorbar
            colorbar = False
            time_label_ = time_label
            time_label = None
        else:
            colorbar_ = False
            time_label_ = None

        src_hemi = src.sub(source='lh')
        data = src_hemi.get_data(data_dims)
        vertices = src.source.lh_vertno
        brain.add_data(data, vmin, vmax, None, colormap, alpha, vertices,
                       smoothing_steps, times, time_label_, colorbar_, 'lh')

    if src.source.rh_n:
        src_hemi = src.sub(source='rh')
        data = src_hemi.get_data(data_dims)
        vertices = src.source.rh_vertno
        brain.add_data(data, vmin, vmax, None, colormap, alpha, vertices,
                       smoothing_steps, times, time_label, colorbar, 'rh')

    # mask
    if mask:
        lh, rh = src.source._mask_label()
        if src.source.lh_n and lh:
            brain.add_label(lh, alpha=0.5)
        if src.source.rh_n and rh:
            brain.add_label(rh, alpha=0.5)

    # set parallel view
    if parallel:
        _set_parallel(brain, surf)

    # without this sometimes the brain position is off
    brain.screenshot()

    return brain


def _set_parallel(brain, surf):
    if surf == 'inflated':
        camera_scale = 95
    else:
        camera_scale = 65

    for figs in brain._figures:
        for fig in figs:
            fig.scene.camera.parallel_scale = camera_scale
            fig.scene.camera.parallel_projection = True


def _dspm_lut(fmin, fmid, fmax):
    """Create a color look up table (lut) for a dSPM plot

    Parameters
    ----------
    fmin, fmid, fmax : scalar
        Start-, mid- and endpoint for the color gradient.
    """
    if not (fmin < fmid < fmax):
        raise ValueError("Invalid colormap, we need fmin < fmid < fmax")
    elif fmin < 0:
        msg = ("The dSPM color gradient is symmetric around 0, fmin needs to "
               "be > 0.")
        raise ValueError(msg)

    n = 256
    lut = np.zeros((n, 4), dtype=np.uint8)
    i0 = _idx(n / 2)
    imin = _idx((fmin / fmax) * i0)
    min_n = i0 - imin
    min_p = i0 + imin
    imid = _idx((fmid / fmax) * i0)
    mid_n = i0 - imid
    mid_p = i0 + imid

    # red end
    lut[i0:, 0] = 255
    lut[mid_p:, 1] = np.linspace(0, 255, n - mid_p)

    # blue end
    lut[:i0, 2] = 255
    lut[:mid_n, 0] = np.linspace(127, 0, mid_n)
    lut[:mid_n, 1] = np.linspace(127, 0, mid_n)

    # alpha
    lut[:mid_n, 3] = 255
    lut[mid_n:min_n, 3] = np.linspace(255, 0, min_n - mid_n)
    lut[min_n:min_p, 3] = 0
    lut[min_p:mid_p, 3] = np.linspace(0, 255, mid_p - min_p)
    lut[mid_p:, 3] = 255

    return lut


def _p_lut(pmap, tmap, p0=0.05, p1=0.01, n=256, solid=False):
    """Creat a color look up table (lut) for p-values

    Parameters
    ----------
    pmap : NDVar
        Map of p-values.
    tmap : NDVar
        Map of signed statistic (only used to code the sign of each p-value).
    p0 : scalar
        Highest p-vale that should be visible.
    p1 : scalar
        P-value where the colormap changes from ramping alpha to ramping color.
    n : int
        Number of color categories in the lut.
    solid : bool
        Instead of ramping alpha/color hue, create steps at p0 and p1.
    """
    if p1 >= p0:
        raise ValueError("p1 needs to be smaller than p0.")

    pstep = 2 * p0 / _idx(n / 2 - 1)

    # max p-value that needs to be represented
    vmax = p0 + pstep

    # bring interesting p-values to the range [pstep vmax]
    pmap = vmax - pmap

    # set uninteresting values to zero
    pmap.x.clip(0, vmax, pmap.x)

    # add sign to p-values
    if tmap is not None:
        pmap.x *= np.sign(tmap.x)

    # http://docs.enthought.com/mayavi/mayavi/auto/example_custom_colormap.html
    lut = np.zeros((n, 4), dtype=np.uint8)

    middle = n / 2
    p0n = _idx(p0 / pstep)
    p0p = n - p0n
    p1n = _idx(p1 / pstep)
    p1p = n - p1n

    # negative colors
    lut[:middle, 2] = 255
    lut[:p1n, 0] = np.linspace(255, 0, p1n)

    # positive colors
    lut[middle:, 0] = 255
    lut[p1p:, 1] = np.linspace(0, 255, n - p1p)

    # alpha
    if solid:
        lut[:p0n, 3] = 255
        lut[p0p:, 3] = 255
    else:
        lut[:p1n, 3] = 255
        lut[p1n:p0n, 3] = np.linspace(255, 0, p0n - p1n)
        lut[p0p:p1p, 3] = np.linspace(0, 255, p1p - p0p)
        lut[p1p:, 3] = 255

    return pmap, lut, vmax



def _activation_lut(threshold=3, vmax=8):
    """
    Creates a lookup table containing a color map for plotting stc activation.

    Parameters
    ----------
    threshold : scalar
        threshold is point at which values gain 50% visibility.
        from threshold and beyond, alpha = 100% visibility.
    vmax : scalar
        the upper range of activation values. values are clipped above this range. When None,
        vmax = two standard deviations above and below the mean.

    Notes
    -----
    Colors:
    - negative is blue.
    - super negative is purple.
    - positive is red.
    - super positive is yellow.
    """

    values = np.linspace(-vmax, vmax, 256)
    trans_uidx = np.argmin(np.abs(values - threshold))
    trans_lidx = np.argmin(np.abs(values + threshold))

    # Transparent ramping
    lut = np.zeros((256, 4), dtype=np.uint8)
    lut[127:trans_uidx, 3] = np.linspace(0, 128, trans_uidx - 127)
    lut[trans_lidx:127, 3] = np.linspace(128, 0, 127 - trans_lidx)
    lut[trans_uidx:, 3] = 255
    lut[:trans_lidx, 3] = 255

    # negative -> Blue
    lut[:127, 2] = 255
    # super negative -> Purple
    lut[:trans_lidx, 0] = np.linspace(0, 255, trans_lidx)
    # positive -> Red
    lut[127:, 0] = 255
    # super positive -> Yellow
    lut[trans_uidx:, 1] = np.linspace(0, 255, 256 - trans_uidx)

    return lut


def _voxel_brain(data, lut, vmin, vmax):
    """Plot spheres for volume source space

    Parameters
    ----------
    data : NDVar
        Data to plot.
    lut : array
        Color LUT.
    vmin, vmax : scalar
        Data range.
    """
    if data.dimnames != ('source',):
        raise ValueError("Can only plot 1 dimensional source space NDVars")

    from mayavi import mlab

    x, y, z = data.source.coordinates.T

    figure = mlab.figure()
    mlab.points3d(x, y, z, scale_factor=0.002, opacity=0.5)
    pts = mlab.points3d(x, y, z, data.x, vmin=vmin, vmax=vmax)
    pts.module_manager.scalar_lut_manager.lut.table = lut
    return figure


def bin_table(ndvar, tstart=None, tstop=None, tstep=0.1, surf='smoothwm',
              views=('lat', 'med'), hemi=None, summary=np.sum, vmax=None,
              axw=300, axh=250, *args, **kwargs):
    """Create a table with images for time bins

    Parameters
    ----------
    ndvar : NDVar (time x source)
        Data to be plotted.
    tstart : None | scalar
        Time point of the start of the first bin (inclusive; None to use the
        first time point in ndvar).
    tstop : None | scalar
        End of the last bin (exclusive; None to end with the last time point
        in ndvar).
    tstep : scalar
        Size of each bin (in seconds).
    surf : 'inflated' | 'pial' | 'smoothwm' | 'sphere' | 'white'
        Freesurfer surface to use as brain geometry.
    views : list of str
        Views to display (for each hemisphere, lh first). Options are:
        'rostral', 'parietal', 'frontal', 'ventral', 'lateral', 'caudal',
        'medial', 'dorsal'.
    hemi : 'lh' | 'rh' | 'both'
        Which hemispheres to plot (default based on data).
    summary : callable
        How to summarize data in each time bin. The value should be a function
        that takes an axis parameter (e.g., numpy summary functions like
        numpy.sum, numpy.mean, numpy.max, ..., default is numpy.sum).
    vmax : scalar != 0
        Maximum value in the colormap. Default is the maximum value in the
        cluster.
    axw, axh : scalar
        Subplot width/height (default axw=300, axh=250).
    foreground : mayavi color
        Figure foreground color (i.e., the text color).
    background : mayavi color
        Figure background color.
    parallel : bool
        Set views to parallel projection (default ``True``).
    smoothing_steps : None | int
        Number of smoothing steps if data is spatially undersampled (pysurfer
        ``Brain.add_data()`` argument).
    subjects_dir : None | str
        Override the subjects_dir associated with the source space dimension.


    Returns
    -------
    images : Image
        FMTXT Image object that can be saved as SVG or integrated into an
        FMTXT document.
    """
    data = ndvar.bin(tstep, tstart, tstop, summary)
    ims = []
    if vmax is None:
        vmax = max(-data.min(), data.max())
        if vmax == 0:
            raise NotImplementedError("The data of ndvar is all zeros")
    elif vmax == 0:
        raise ValueError("vmax can't be 0")
    elif vmax < 0:
        vmax = -vmax

    if hemi is None:
        hemis = []
        if data.source.lh_n:
            hemis.append('lh')
        if data.source.rh_n:
            hemis.append('rh')
    elif hemi == 'both':
        hemis = ['lh', 'rh']
    elif hemi == 'lh' or hemi == 'rh':
        hemis = [hemi]
    else:
        raise ValueError("hemi=%s" % repr(hemi))

    for hemi in hemis:
        brain = cluster(data, vmax, surf, views[0], hemi, False, None, axw, axh,
                        None, None, *args, **kwargs)

        hemi_lines = [[] for _ in views]
        for i in xrange(len(data.time)):
            brain.set_data_time_index(i)
            for line, view in izip(hemi_lines, views):
                brain.show_view(view)
                im = brain.screenshot_single('rgba', True)
                line.append(im)
        ims += hemi_lines
#         brain.close() # causes segfault in wx

    header = ['%i - %i ms' % (ms(t0), ms(t1)) for t0, t1 in data.info['bins']]
    return im_table(ims, header)


def connectivity(source):
    """Plot source space connectivity

    Parameters
    ----------
    source : SourceSpace | NDVar
        Source space or NDVar containing the source space.

    Returns
    -------
    figure : mayavi Figure
        The figure.
    """
    from mayavi import mlab

    if isinstance(source, NDVar):
        source = source.get_dim('source')

    connections = source.connectivity()
    coords = source.coordinates
    x, y, z = coords.T

    figure = mlab.figure()
    src = mlab.pipeline.scalar_scatter(x, y, z, figure=figure)
    src.mlab_source.dataset.lines = connections
    lines = mlab.pipeline.stripper(src)
    mlab.pipeline.surface(lines, colormap='Accent', line_width=1, opacity=1.,
                          figure=figure)
    return figure


def image(brain, filename, alt=None, close=False):
    """Deprecated, use Brain.image()"""
    warn("plot.brain.image() has been deprecated, Brain.image() instead",
         DeprecationWarning)

    name, ext = os.path.splitext(filename)
    if ext:
        format = ext[1:]
    else:
        format = 'png'

    im = brain.screenshot('rgba', True)
    if close:
        brain.close()
    return Image.from_array(im, name, format, alt)


def copy(brain):
    "Copy the figure to the clip board"
    import wx

    tempdir = mkdtemp()
    tempfile = os.path.join(tempdir, "brain.png")
    brain.save_image(tempfile)

    bitmap = wx.Bitmap(tempfile, wx.BITMAP_TYPE_PNG)
    bitmap_obj = wx.BitmapDataObject(bitmap)

    if not wx.TheClipboard.IsOpened():
        open_success = wx.TheClipboard.Open()
        if open_success:
            wx.TheClipboard.SetData(bitmap_obj)
            wx.TheClipboard.Close()
            wx.TheClipboard.Flush()
