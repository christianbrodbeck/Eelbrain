'''
Plot :class:`NDVar` objects containing source estimates with mayavi/pysurfer.

.. autosummary::
   :toctree: generated

   activation
   cluster
   dspm
   stat
   surfer_brain

Functions that can be applied to the :class:`surfer.Brain` instances that are
returned by the plotting functions:

.. autosummary::
   :toctree: generated

    bin_table
    copy
    image

'''
# author: Christian Brodbeck
from __future__ import division
from itertools import izip
import os
from tempfile import mkdtemp

import numpy as np

from .._data_obj import asndvar, NDVar, UTS
from ..fmtxt import Image, im_table


__all__ = ['activation', 'dspm', 'surfer_brain', 'stat']


def _idx(i):
    return int(round(i))


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
    fmin, fmax : scalar
        Start- and endpoint for the color gradient. Values between -fmin
        and fmin are transparent.
    fmid : None | scalar
        Midpoint for the color gradient. If fmid is None (default) it is set
        half way between fmin and fmax.
    surf : 'inflated' | 'pial' | 'smoothwm' | 'sphere' | 'white'
        Freesurfer surface to use as brain geometry.
    views : str | iterator of str
        View or views to show in the figure.
    colorbar : bool
        Whether to add a colorbar to the figure.
    w, h, axw, axh : scalar
        Layout parameters (figure width/height, subplot width/height).
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


def stat(p_map, param_map=None, p0=0.05, p1=0.01, solid=False, *args,
         **kwargs):
    """
    Plot a statistic in source space.

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
        View or views to show in the figure.
    colorbar : bool
        Whether to add a colorbar to the figure.
    w, h, axw, axh : scalar
        Layout parameters (figure width/height, subplot width/height).
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
    """
    Plot activation in source space.

    Parameters
    ----------
    src : NDVar, dims = ([case,] source, [time])
        NDVar with SourceSpace dimension. If stc contains a case dimension,
        the average across cases is taken.
    threshold : scalar | None
        the point at which alpha transparency is 50%. When None,
        threshold = one standard deviation above and below the mean.
    vmax : scalar | None
        the upper range of activation values. values are clipped above this
        range. When None, vmax = two standard deviations above and below the
        mean.
    surf : 'inflated' | 'pial' | 'smoothwm' | 'sphere' | 'white'
        Freesurfer surface to use as brain geometry.
    views : str | iterator of str
        View or views to show in the figure.
    colorbar : bool
        Whether to add a colorbar to the figure.
    w, h, axw, axh : scalar
        Layout parameters (figure width/height, subplot width/height).
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
    vmax : scalar
        Maximum value in the colormap. Default is the maximum value in the
        cluster.
    surf : 'inflated' | 'pial' | 'smoothwm' | 'sphere' | 'white'
        Freesurfer surface to use as brain geometry.
    views : str | iterator of str
        View or views to show in the figure.
    colorbar : bool
        Whether to add a colorbar to the figure.
    w, h, axw, axh : scalar
        Layout parameters (figure width/height, subplot width/height).
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

    lut = _dspm_lut(0, vmax / 10, vmax)
    return _plot(cluster, lut, -vmax, vmax, *args, **kwargs)


def surfer_brain(src, colormap='hot', vmin=0, vmax=9, surf='smoothwm',
                 views=['lat', 'med'], colorbar=True, time_label='%.3g s',
                 w=None, h=None, axw=None, axh=None, background=None,
                 parallel=True, smoothing_steps=None, subjects_dir=None):
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
        View or views to show in the figure.
    colorbar : bool
        Whether to add a colorbar to the figure.
    w, h, axw, axh : scalar
        Layout parameters (figure width/height, subplot width/height).
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
    from surfer import Brain

    if isinstance(views, basestring):
        views = [views]
    elif not isinstance(views, list):
        views = list(views)

    src = asndvar(src)  # , sub=None, ds=ds)
    if src.has_case:
        src = src.summary()

    if src.source.lh_n and src.source.rh_n:
        hemi = 'split'
        n_hemi = 2
    elif src.source.lh_n:
        hemi = 'lh'
        n_hemi = 1
    elif not src.source.rh_n:
        raise ValueError('No data')
    else:
        hemi = 'rh'
        n_hemi = 1

    title = None
    config_opts = {}
    if w is not None:
        config_opts['width'] = w
    elif axw is not None:
        config_opts['width'] = axw * n_hemi
    else:
        config_opts['width'] = 500 * n_hemi

    if h is not None:
        config_opts['height'] = h
    elif axh is not None:
        config_opts['height'] = axh * len(views)
    else:
        config_opts['height'] = 400 * len(views)

    if background is not None:
        config_opts['background'] = background

    if subjects_dir is None:
        subjects_dir = src.source.subjects_dir

    brain = Brain(src.source.subject, hemi, surf, True, title,
                  config_opts=config_opts, views=views,
                  subjects_dir=subjects_dir)

    # general PySurfer data args
    alpha = 1
    if smoothing_steps is None and src.source.kind == 'ico':
        smoothing_steps = src.source.grade + 1

    if src.has_dim('time'):
        times = src.time.times
        data_dims = ('source', 'time')
    else:
        times = None
        data_dims = ('source',)

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

    if parallel:  # set parallel view
        if surf == 'inflated':
            camera_scale = 95
        else:
            camera_scale = 65

        for figs in brain._figures:
            for fig in figs:
                fig.scene.camera.parallel_scale = camera_scale
                fig.scene.camera.parallel_projection = True

    return brain


def _dspm_lut(fmin, fmid, fmax):
    """Create a color look up table (lut) for a dSPM plot

    Parameters
    ----------
    fmin, fmid, fmax : scalar
        Start-, mid- and endpoint for the color gradient.
    """
    if not (fmin < fmid < fmax):
        raise ValueError("Invalid colormap, we need fmin < fmid < fmax")

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
              views=['lat', 'med'], summary=np.sum):
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
        Views to display (for each hemisphere, lh first).
    summary : callable
        How to summarize data in each time bin. The value should be a function
        that takes an axis parameter (e.g., numpy summary functions like
        numpy.sum, numpy.mean, numpy.max, ..., default is numpy.sum).

    Returns
    -------
    images : Image
        FMTXT Image object that can be saved as SVG or integrated into an
        FMTXT document.
    """
    data = _bin_data(ndvar, tstart, tstop, tstep, summary)
    ims = []
    vmax = max(abs(data.min()), data.max())

    hemis = []
    if data.source.lh_n:
        hemis.append('lh')
    if data.source.rh_n:
        hemis.append('rh')

    for hemi in hemis:
        hemi_data = data.sub(source=hemi)
        brain = cluster(hemi_data, vmax, surf, views[0], colorbar=False, w=300,
                        h=250, time_label=None)
        im = brain.screenshot_single('rgba', True)

        hemi_lines = [[] for _ in views]
        for i in xrange(len(data.time)):
            brain.set_data_time_index(i)
            for line, view in izip(hemi_lines, views):
                brain.show_view(view)
                im = brain.screenshot_single('rgba', True)
                line.append(im)
        ims += hemi_lines
#         brain.close() # causes segfault in wx

    bins = data.info['bins']
    header = ['%i - %i ms' % (t0 * 1000, t1 * 1000) for t0, t1 in bins]
    im = im_table(ims, header)

    return im


def _bin_data(ndvar, tstart, tstop, tstep, summary):
    data = ndvar.get_data(('source', 'time'))

    # times
    if tstart is None:
        tstart = ndvar.time.tmin
    if tstop is None:
        tstop = ndvar.time.tmax + ndvar.time.tstep
    times = np.arange(tstart, tstop, tstep)
    if times[-1] < tstop:
        times = np.append(times, tstop)

    n_bins = len(times) - 1
    x = np.empty((len(data), n_bins))
    bins = []
    for i in xrange(n_bins):
        t0 = times[i]
        t1 = times[i + 1]
        bins.append((t0, t1))
        idx = ndvar.time.dimindex((t0, t1))
        x[:, i] = summary(data[:, idx], axis=1)

    time = UTS(tstart + tstep / 2, tstep, n_bins)
    dims = (ndvar.source, time)
    info = ndvar.info.copy()
    info['bins'] = bins
    out = NDVar(x, dims, info, ndvar.name)
    return out


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
    """Create an FMText Image from a brain instance

    Parameters
    ----------
    brain : Brain
        Pysurfer Brain instance.
    filename : str
        Filename for the image (should end with the desired extension).
    alt : None | str
        Alternate text, placeholder in case the image can not be found
        (HTML `alt` tag).
    close : bool
        Close the brain window after creating the image.
    """
    im = brain.screenshot('rgba', True)
    if close:
        brain.close()
    return Image.from_array(im, filename, alt)


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
