# -*- coding: utf-8 -*-
# Author: Proloy Das <proloy@umd.edu>
"""2d projections of an ROI/mask image visualization via nilearn.plotting.glassbrain

Contains code from nilearn governed by the following license (3-Clause BSD):

Copyright (c) 2007 - 2015 The nilearn developers.
All rights reserved.


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of the nilearn developers nor the names of
     its contributors may be used to endorse or promote products
     derived from this software without specific prior written
     permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.

"""
import warnings

import numpy as np

from .._data_obj import NDVar, VolumeSourceSpace, asndvar
from .._utils.numpy_utils import newaxis
from ._base import ColorBarMixin, TimeSlicerEF, Layout, EelFigure, butterfly_data
from ._utsnd import Butterfly


# Copied from nilearn.plotting.img_plotting
def _crop_colorbar( cbar, cbar_vmin, cbar_vmax ):
    """crop a colorbar to show from cbar_vmin to cbar_vmax.(symmetric_cbar=False)"""
    if (cbar_vmin is None) and (cbar_vmax is None):
        return
    cbar_tick_locs = cbar.locator.locs
    if cbar_vmax is None:
        cbar_vmax = cbar_tick_locs.max()
    if cbar_vmin is None:
        cbar_vmin = cbar_tick_locs.min()
    new_tick_locs = np.linspace(cbar_vmin, cbar_vmax,
                                len(cbar_tick_locs))
    cbar.ax.set_ylim(cbar.norm(cbar_vmin), cbar.norm(cbar_vmax))
    outline = cbar.outline.get_xy()
    outline[:2, 1] += cbar.norm(cbar_vmin)
    outline[2:6, 1] -= (1. - cbar.norm(cbar_vmax))
    outline[6:, 1] += cbar.norm(cbar_vmin)
    cbar.outline.set_xy(outline)
    cbar.set_ticks(new_tick_locs, update_ticks=True)


class GlassBrain(TimeSlicerEF, ColorBarMixin, EelFigure):
    """Plot 2d projections of a brain volume

    Based on :func:`nilearn.plotting.plot_glass_brain`.

    Parameters
    ----------
    ndvar : NDVar  ([case,] time, source[, space])
        Data to plot; if ``ndvar`` has a case dimension, the mean is plotted.
        if ``ndvar`` has a space dimension, the norm is plotted.
    cmap : str
        Colormap (name of a matplotlib colormap).
    vmin : scalar
        Plot data range minimum.
    vmax : scalar
        Plot data range maximum.
    dest : 'mri' | 'surf'
        If 'mri' the volume is defined in the coordinate system of
        the original T1 image. If 'surf' the coordinate system
        of the FreeSurfer surface is used (Surface RAS).
    mri_resolution: bool
        If True the image is created in MRI resolution through upsampling.
        WARNING: it can result in significantly high memory usage.
    mni305 : bool
        Project data from MNI-305 space to MNI-152 space (by default this
        is enabled iff the source space subject is ``fsaverage``).
    black_bg : boolean. Default is 'False'
        If True, the background of the image is set to be black.
    display_mode : str
        Direction of the cuts:

        - ``'x'``: sagittal
        - ``'y'``: coronal
        - ``'z'``: axial
        - ``'l'``: sagittal, left hemisphere only
        - ``'r'``: sagittal, right hemisphere only
        - ``'ortho'``: three cuts in orthogonal directions, equivalent to
          ``'yxz'``

        Possible values are: 'ortho', 'x', 'y', 'z', 'xz', 'yx', 'yz',
        'l', 'r', 'lr', 'lzr', 'lyr', 'lzry', 'lyrz'. Default depends on
        hemispheres in data.
    threshold : scalar | 'auto'
        If a number is given, values below the threshold (in absolute value) are
        plotted as transparent. If ``'auto'`` is given, the threshold is
        determined magically by analysis of the image.
    colorbar : boolean
        If True, display a colorbar on the right of the plots.
    draw_cross: boolean
        If draw_cross is True, a cross is drawn on the plot to
        indicate the cut plosition.
    annotate: boolean
        If annotate is True, positions and left/right annotation
        are added to the plot.
    alpha : float between 0 and 1
        Alpha transparency for the brain schematics
    plot_abs : bool
        Plot the maximum intensity projection of the absolute value (rendering
        positive and negative values in the same manner). By default,
        (``False``), the sign of the maximum intensity will be represented with
        different colors. See `examples <http://nilearn.github.io/auto_examples/
        01_plotting/plot_demo_glass_brain_extensive.html>`_.
    draw_arrows: boolean
        If set to True arrows pointing the direction of activation is
        drawn over the glassbrain plots. Naturally, for this to work
        ndvar needs to contain space dimension i.e (3D vector data).
        By default it is set to False.
    symmetric_cbar : boolean | 'auto'
        Specifies whether the colorbar should range from -vmax to vmax
        or from vmin to vmax. Setting to 'auto' will select the latter if
        the range of the whole image is either positive or negative.
        Note: The colormap will always be set to range from -vmax to vmax.
    interpolation : str
        Interpolation to use when resampling the image to the destination
        space. Can be "continuous" (default) to use 3rd-order spline
        interpolation, or "nearest" to use nearest-neighbor mapping.
        "nearest" is faster but can be noisier in some cases.
    title : str | bool
        Figure title. Set to ``True`` to display current time point as figure
        title.
    ...
        Also accepts :ref:`general-layout-parameters`.

    Notes
    -----
    The brain overlay assumes coordinates in MNI152 space
    (see `The MNI brain and the Talairach atlas
    <http://imaging.mrc-cbu.cam.ac.uk/imaging/MniTalairach>`_)
    """
    _name = 'GlassBrain'
    _make_axes = False
    _display_time_in_frame_title = True

    def __init__(self, ndvar, cmap=None, vmin=None, vmax=None, dest='mri',
                 draw_arrows=False, mri_resolution=False, mni305=None,
                 black_bg=False, display_mode=None, threshold='auto',
                 colorbar=False, draw_cross=True, annotate=True,
                 alpha=0.7, plot_abs=False, symmetric_cbar="auto",
                 interpolation='nearest', **kwargs):
        # Give wxPython a chance to initialize the menu before pyplot
        from .._wxgui import get_app
        get_app(jumpstart=True)

        # Lazy import of matplotlib.pyplot
        from nilearn.image import index_img
        from nilearn.plotting import cm
        from nilearn.plotting.displays import get_projector
        from nilearn.plotting.img_plotting import _get_colorbar_and_data_ranges

        if cmap is None:
            cmap = cm.cold_hot if black_bg else cm.cold_white_hot
        self.cmap = cmap

        if isinstance(ndvar, VolumeSourceSpace):
            source = ndvar
            ndvar = None
        else:
            ndvar = asndvar(ndvar)
            source = ndvar.get_dim('source')
            if not isinstance(source, VolumeSourceSpace):
                raise ValueError(f"ndvar={ndvar!r}:  need volume source space data")
            if isinstance(ndvar.x, np.ma.MaskedArray) and np.all(ndvar.x.mask):
                ndvar = None

        if mni305 is None:
            mni305 = source.subject == 'fsaverage'

        if ndvar:
            if ndvar.has_case:
                ndvar = ndvar.mean('case')

            if mni305 is None:
                mni305 = ndvar.source.subject == 'fsaverage'

            src = source.get_source_space()
            img = _stc_to_volume(ndvar, src, dest, mri_resolution, mni305)
            if ndvar.has_dim('time'):
                time = ndvar.get_dim('time')
                t0 = time[0]
                imgs = [index_img(img, i) for i in range(len(time))]
                img0 = imgs[0]
            else:
                img0 = img
                imgs = time = t0 = None
            if draw_arrows:
                if not ndvar.has_dim('space'):
                    warnings.warn('Cannot draw arrows:'
                                  'ndvar does not have space dimension.'
                                  'Continuing without arrows...')
                    draw_arrows = False
                    dir_imgs = None
                else:
                    dir_imgs = []
                    dir_img0 = []
                    for direction in ndvar.space._directions:
                        dir_img = _stc_to_volume(ndvar.sub(space=direction),
                                                 dest, mri_resolution, mni305)
                        if ndvar.has_dim('time'):
                            dir_imgs.append([index_img(dir_img, i)
                                             for i in range(len(ndvar.time))])
                            dir_img0.append(index_img(dir_img, 0))
                        else:
                            dir_imgs.append([dir_img])
                    if plot_abs:
                        warnings.warn('Cannot use maximum intensity projection'
                                      'of the absolute value in draw_arrows'
                                      'mode.')
                        plot_abs = False
            else:
                dir_imgs = None

            # determine parameters for colorbar
            if ndvar.has_dim('space'):
                data = ndvar.norm('space').x
            else:
                data = ndvar.x
            if plot_abs:
                cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
                    data, vmax, symmetric_cbar, kwargs, 0)
            else:
                cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
                    data, vmax, symmetric_cbar, kwargs)
        else:
            cbar_vmin = cbar_vmax = imgs = img0 = dir_imgs = time = t0 = None


        self.time = time
        self._src = src
        self._ndvar = ndvar
        self._imgs = imgs
        self._dir_imgs = dir_imgs

        show_nan_msg = False
        if vmax is not None and np.isnan(vmax):
            vmax = None
            show_nan_msg = True
        if vmin is not None and np.isnan(vmin):
            vmin = None
            show_nan_msg = True
        if show_nan_msg:
            warnings.warn('NaN is not permitted for the vmax and vmin arguments. '
                          'Tip: Use np.nanmax() instead of np.max().')

        # Deal with automatic settings of plot parameters
        if threshold == 'auto':
            # Threshold below a percentile value, to be sure that some
            # voxels pass the threshold
            threshold = _fast_abs_percentile(self._ndvar)

        # layout
        if display_mode is None:
            display_mode = ''
            if 'lh' in source.hemi:
                display_mode += 'l'
            display_mode += 'y'
            if 'rh' in source.hemi:
                display_mode += 'r'
            display_mode += 'z'
        n_plots = 3 if display_mode == 'ortho' else len(display_mode)
        layout = Layout(n_plots, 0.85, 2.6, tight=False, ncol=n_plots, **kwargs)
        # frame title
        if layout.name:
            frame_title = layout.name
        elif isinstance(layout.title, str):
            frame_title = layout.title
        elif ndvar and ndvar.name:
            frame_title = ndvar.name
        else:
            frame_title = source.subject
        EelFigure.__init__(self, frame_title, layout)

        project = get_projector(display_mode)
        display = project(img0, alpha=alpha, plot_abs=plot_abs, threshold=threshold, figure=self.figure, axes=None, black_bg=black_bg, colorbar=colorbar)
        if img0:
            display.add_overlay(img0, threshold=threshold, interpolation=interpolation, colorbar=colorbar, vmin=vmin, vmax=vmax, cmap=cmap)

        ColorBarMixin.__init__(self, self._colorbar_params, ndvar)

        self.display = display
        self.threshold = threshold
        self.interpolation = interpolation
        self.colorbar = colorbar
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax
        self._arrows = draw_arrows

        if draw_arrows:
            self.arrow_scale = 7*np.max(np.abs([vmax, vmin]))
            self._add_arrows(0)
        if annotate:
            display.annotate()
        if draw_cross:
            display.draw_cross()
        if layout.title is True:
            if t0 is None:
                raise TypeError(f"title=True; only allowed when displaying data with multiple time points")
            display.title("???")
            self._update_title(t0)
        elif layout.title:
            display.title(layout.title)
        if hasattr(display, '_cbar'):
            cbar = display._cbar
            _crop_colorbar(cbar, cbar_vmin, cbar_vmax)

        ndvars = [[ndvar]] if ndvar else None
        TimeSlicerEF.__init__(self, 'time', ndvars)

        self._show()

    def _fill_toolbar(self, tb):
        ColorBarMixin._fill_toolbar(self, tb)

    # used by update_time
    def _add_arrows(self, t, **kwargs):
        """Adds arrows using matplotlib.quiver"""
        # Format 3D data
        data_list = []
        extent_list = []

        for display_ax in self.display.axes.values():
            data = []
            for k, direction in enumerate(self._ndvar.space._directions):
                vol = self._dir_imgs[k][t]
                try:
                    vol_data = np.squeeze(_safe_get_data(vol))
                    data_2d = display_ax.transform_to_2d(vol_data,
                                                         vol.affine)
                    data_2d = np.squeeze(data_2d)
                except IndexError:
                    # We are cutting outside the indices of the data
                    data_2d = None
                data.append(data_2d)
            data_list.append(np.array(data))
            data_bounds = get_bounds(vol.shape, vol.affine)
            if display_ax.direction == 'y':
                (xmin, xmax), (_, _), (zmin, zmax) = data_bounds
            elif display_ax.direction in 'xlr':
                (_, _), (xmin, xmax), (zmin, zmax) = data_bounds
            elif display_ax.direction == 'z':
                (xmin, xmax), (zmin, zmax), (_, _) = data_bounds
            extent = (xmin, xmax, zmin, zmax)
            extent_list.append(extent)

        to_iterate_over = zip(self.display.axes.values(), data_list,
                              extent_list)

        # Plotting using quiver
        if self.display._black_bg:
            color = 'w'
        else:
            color = 'k'
        ims = []
        for display_ax, data_2d, extent in to_iterate_over:
            if data_2d is not None:
                # get data mask
                thr = self.threshold ** 2
                data = (data_2d.copy() ** 2).sum(axis=0)
                not_mask = data > thr

                # If data_2d is completely masked, then there is nothing to
                # plot. Hence, continued to loop over. This problem came up
                # with matplotlib 2.1.0. See issue #9280 in matplotlib.
                if not_mask.any():
                    affine_2d = get_transform(extent, data.shape)
                    indices = np.where(not_mask)
                    x, y = coord_transform_2d(indices, affine_2d)
                    if display_ax.direction == 'y':
                        dir_data = (data_2d[0][indices], data_2d[2][indices])
                    elif display_ax.direction == 'l':
                        dir_data = (-data_2d[1][indices], data_2d[2][indices])
                    elif display_ax.direction in 'xr':
                        dir_data = (data_2d[1][indices], data_2d[2][indices])
                    elif display_ax.direction == 'z':
                        dir_data = (data_2d[0][indices], data_2d[1][indices])

                    im = display_ax.ax.quiver(x, y, dir_data[0], dir_data[1],
                                              color=color,
                                              scale=self.arrow_scale)
                else:
                    continue
            ims.append(im)

        self._quivers = ims

        return

    # used by _update_time
    def _remove_overlay(self):
        for axis in self.display._cut_displayed:
            if len(self.display.axes[axis].ax.images) > 0:
                self.display.axes[axis].ax.images[-1].remove()

    # used by _update_time
    def _update_title(self, t):
        if self._layout.title is True:
            first_axis = self.display._cut_displayed[0]
            ax = self.display.axes[first_axis].ax
            ax.texts[-1].set_text('time = %s ms' % round(t * 1e3))

    def _update_time(self, t, fixate):
        index = self.time._array_index(t)
        self._remove_overlay()
        self.display.add_overlay(
            self._imgs[index], threshold=self.threshold, interpolation=self.interpolation,
            colorbar=False, vmin=self.vmin, vmax=self.vmax, cmap=self.cmap)
        # A little hack to make sure that the display
        # still has correct colorbar flag.
        self.display._colorbar = self.colorbar

        # take care of arrows
        if self._arrows: self._remove_arrows()
        if self._arrows: self._add_arrows(index)

        self._update_title(t)

        self.draw()

    def animate(self):
        for t in self.time:
            self.set_time(t)

    def _colorbar_params(self):
        return self.cmap, self.vmin, self.vmax

    @classmethod
    def butterfly(cls, y, cmap=None, vmin=None, vmax=None, draw_arrows=False,
                  dest='mri', mri_resolution=False, mni305=None,
                  black_bg=False, display_mode=None, threshold='auto',
                  colorbar=False, alpha=0.7, plot_abs=False,
                  symmetric_cbar="auto", interpolation='nearest',
                  w=5, h=2.5, xlim=None, name=None, **kwargs):
        """Shortcut for a butterfly-plot with a time-linked glassbrain plot

        Parameters
        ----------
        y : NDVar  ([case,] time, source[, space])
            Data to plot; if ``ndvar`` has a case dimension, the mean is plotted.
            if ``ndvar`` has a space dimension, the norm is plotted.
        cmap : str
            Colormap (name of a matplotlib colormap).
        vmin : scalar
            Plot data range minimum.
        vmax : scalar
            Plot data range maximum.
        dest : 'mri' | 'surf'
            If 'mri' the volume is defined in the coordinate system of
            the original T1 image. If 'surf' the coordinate system
            of the FreeSurfer surface is used (Surface RAS).
        mri_resolution: bool, Default is False
            If True the image will be created in MRI resolution.
            WARNING: it can result in significantly high memory usage.
        mni305 : bool
            Project data from MNI-305 space to MNI-152 space (by default this
            is enabled iff the source space subject is ``fsaverage``).
        black_bg : boolean
            If True, the background of the image is set to be black.
        display_mode : str
            Direction of the cuts:

            - ``'x'``: sagittal
            - ``'y'``: coronal
            - ``'z'``: axial
            - ``'l'``: sagittal, left hemisphere only
            - ``'r'``: sagittal, right hemisphere only
            - ``'ortho'``: three cuts in orthogonal directions, equivalent to
              ``'yxz'``

            Possible values are: 'ortho', 'x', 'y', 'z', 'xz', 'yx', 'yz',
            'l', 'r', 'lr', 'lzr', 'lyr', 'lzry', 'lyrz'. Default depends on
            hemispheres in data.
        threshold : scalar | 'auto'
            If a number is given, values below the threshold (in absolute value) are
            plotted as transparent. If ``'auto'`` is given, the threshold is
            determined magically by analysis of the image.
        colorbar : boolean, Default is False
            If True, display a colorbar on the right of the plots.
        alpha : float between 0 and 1
            Alpha transparency for the brain schematics
        plot_abs : bool
            Plot the maximum intensity projection of the absolute value (rendering
            positive and negative values in the same manner). By default,
            (``False``), the sign of the maximum intensity will be represented with
            different colors. See `examples <http://nilearn.github.io/auto_examples/
            01_plotting/plot_demo_glass_brain_extensive.html>`_. Only affects
            GlassBrain plot.
        draw_arrows: boolean
            If set to True arrows pointing the direction of activation is
            drawn over the glassbrain plots. Naturally, for this to work
            ndvar needs to contain space dimension. By default it is set to
            False.
        symmetric_cbar : boolean or 'auto'
            Specifies whether the colorbar should range from -vmax to vmax
            or from vmin to vmax. Setting to 'auto' will select the latter if
            the range of the whole image is either positive or negative.
            Note: The colormap will always be set to range from -vmax to vmax.
        interpolation : str
            Interpolation to use when resampling the image to the destination
            space. Can be "continuous" (default) to use 3rd-order spline
            interpolation, or "nearest" to use nearest-neighbor mapping.
            "nearest" is faster but can be noisier in some cases.
        w : scalar
            Butterfly plot width (inches).
        h : scalar
            Plot height (inches; applies to butterfly and brain plot).
        xlim : scalar | (scalar, scalar)
            Initial x-axis view limits as ``(left, right)`` tuple or as ``length``
            scalar (default is the full x-axis in the data).
        name : str
            The window title (default is ndvar.name).

        Returns
        -------
        butterfly_plot : plot.Butterfly
            Butterfly plot.
        glassbrain : GlassBrain
            GlassBrain plot.
        """
        import wx
        from .._wxgui import get_app, needs_jumpstart
        from .._wxgui.mpl_canvas import CanvasFrame
        jumpstart = needs_jumpstart()

        hemis, bfly_data, brain_data = butterfly_data(y, None)

        if name is None:
            name = brain_data.name

        p = Butterfly(bfly_data, vmin=vmin, vmax=vmax, xlim=xlim, h=h, w=w, ncol=1, name=name, color='black', ylabel=hemis, axtitle=False)

        # Give wxPython a chance to initialize the menu before pyplot
        if jumpstart:
            get_app().jumpstart()

        # GlassBrain plot
        p_glassbrain = GlassBrain(y, cmap, vmin, vmax, dest, mri_resolution, mni305, black_bg, display_mode, threshold, colorbar, True, True, alpha, plot_abs, symmetric_cbar, interpolation, h=h, name=name, **kwargs)

        # position the brain window next to the butterfly-plot
        if isinstance(p._frame, CanvasFrame):
            px, py = p._frame.GetPosition()
            pw, _ = p._frame.GetSize()
            display_w, _ = wx.DisplaySize()
            brain_w, _ = p_glassbrain._frame.GetSize()
            brain_x = min(px + pw, display_w - brain_w)
            p_glassbrain._frame.SetPosition((brain_x, py))

        p.link_time_axis(p_glassbrain)

        return p, p_glassbrain


def _to_MNI152(trans):
    """Transfrom from MNI-305 space (fsaverage) to MNI-152

    parameters
    ----------
    trans: ndarray
        The affine transform.

    Notes
    -----
    uses approximate transformation mentioned `Link here <https://surfer.nmr.mgh.harvard.edu/fswiki/CoordinateSystems>`_
    """
    t = np.array([[ 0.9975, -0.0073,  0.0176, -0.0429],
                  [ 0.0146,  1.0009, -0.0024,  1.5496],
                  [-0.0130, -0.0093,  0.9971,  1.1840],
                  [ 0,       0,       0,       1     ]])
    return np.dot(t, trans)


def _stc_to_volume(ndvar, src, dest='mri', mri_resolution=False, mni305=False):
    """Save a volume source estimate in a NIfTI file.

    Parameters
    ----------
    ndvar : NDVar
        The source estimate
    src : list | string
        The list of source spaces (should actually be of length 1). If
        string, it is the filepath.
    dest : 'mri' | 'surf'
        If 'mri' the volume is defined in the coordinate system of
        the original T1 image. If 'surf' the coordinate system
        of the FreeSurfer surface is used (Surface RAS).
    mri_resolution : bool
        It True the image is saved in MRI resolution.
        WARNING: if you have many time points the file produced can be
        huge.
    mni305 : bool
        Set to True to convert RAS coordinates of a voxel in MNI305 space (fsaverage space)
        to MNI152 space via updating the affine transformation matrix.

    Returns
    -------
    img : instance Nifti1Image
        The image object.
    """
    src_type = src[0]['type']
    if src_type != 'vol':
        raise ValueError(f"You need a volume source space. Got type: {src_type}")

    if ndvar.has_dim('space'):
        ndvar = ndvar.norm('space')

    if ndvar.has_dim('time'):
        data = ndvar.get_data(('source', 'time'), 0)
    else:
        data = ndvar.get_data(('source', newaxis), 0)

    if not np.all(np.isfinite(data)):
        raise ValueError("Not all values are finite")

    n_times = data.shape[1]
    shape = src[0]['shape']
    shape3d = (shape[2], shape[1], shape[0])
    shape = (n_times, shape[2], shape[1], shape[0])
    vol = np.zeros(shape)

    if mri_resolution:
        mri_shape3d = (src[0]['mri_height'], src[0]['mri_depth'],
                       src[0]['mri_width'])
        mri_shape = (n_times, src[0]['mri_height'], src[0]['mri_depth'],
                     src[0]['mri_width'])
        mri_vol = np.zeros(mri_shape)
        interpolator = src[0]['interpolator']

    n_vertices_seen = 0
    for this_src in src:  # loop over source instants, which is basically one element only!
        assert tuple(this_src['shape']) == tuple(src[0]['shape'])
        mask3d = this_src['inuse'].reshape(shape3d).astype(np.bool)
        n_vertices = np.sum(mask3d)

        for k, v in enumerate(vol):  # loop over time instants
            stc_slice = slice(n_vertices_seen, n_vertices_seen + n_vertices)
            v[mask3d] = data[stc_slice, k]

        n_vertices_seen += n_vertices

    if mri_resolution:
        for k, v in enumerate(vol):
            mri_vol[k] = (interpolator * v.ravel()).reshape(mri_shape3d)
        vol = mri_vol

    vol = vol.T

    if mri_resolution:
        affine = src[0]['vox_mri_t']['trans'].copy()
    else:
        affine = src[0]['src_mri_t']['trans'].copy()
    if dest == 'mri':
        affine = np.dot(src[0]['mri_ras_t']['trans'], affine)

    affine[:3] *= 1e3
    if mni305:
        affine = _to_MNI152(affine)

    # write the image in nifty format
    import nibabel as nib
    header = nib.nifti1.Nifti1Header()
    header.set_xyzt_units('mm', 'msec')
    if ndvar.has_dim('time'):
        header['pixdim'][4] = 1e3 * ndvar.time.tstep
    else:
        header['pixdim'][4] = None
    with warnings.catch_warnings(record=True):  # nibabel<->numpy warning
        img = nib.Nifti1Image(vol, affine, header=header)
    return img


def _fast_abs_percentile(ndvar, percentile=80):
    """A fast version of the percentile of the absolute value.

    Parameters
    ----------
    data: ndvar
        The input data
    percentile: number between 0 and 100
        The percentile that we are asking for

    Returns
    -------
    value: number
        The score at percentile

    Notes
    -----
    This is a faster, and less accurate version of
    scipy.stats.scoreatpercentile(np.abs(data), percentile)
    # inspired from nilearn._utils.extmath.fast_abs_percentile
    """
    if ndvar.has_dim('space'):
        data = ndvar.norm('space').x
    else:
        data = abs(ndvar.x)

    data = data.ravel()
    index = int(data.size * .01 * percentile)
    try:
        # Partial sort: faster than sort
        data = np.partition(data, index)
    except ImportError:
        data.sort()

    return data[index]


def get_transform(extent, shape):
    xmin, xmax, zmin, zmax = extent
    T = np.eye(3)
    T[0, 0] = (zmin - zmax) / shape[0]
    T[0, 2] = zmax
    T[1, 1] = (xmax - xmin) / shape[1]
    T[1, 2] = xmin

    # T[1, 0] = (zmin - zmax) / shape[0]
    # T[1, 2] = zmax
    # T[0, 1] = (xmax - xmin) / shape[1]
    # T[0, 2] = xmin
    return T


def coord_transform_2d(indices, affine):
    rows, cols = indices
    old_coords = np.array([np.array(rows), np.array(cols), np.ones(len(rows))])
    y, x, _ = np.dot(affine, old_coords)
    return (x, y)


def _safe_get_data(img):
    """Get the data in the Nifti1Image object avoiding non-finite values
    Parameters
    ----------
    img: Nifti image/object
        Image to get data.
    Returns
    -------
    data: numpy array
        get_data() return from Nifti image.
        # inspired from nilearn._utils.niimg._safe_get_data
    """
    data = img.get_data()
    non_finite_mask = np.logical_not(np.isfinite(data))
    if non_finite_mask.sum() > 0: # any non_finite_mask values?
        data[non_finite_mask] = 0

    return data
