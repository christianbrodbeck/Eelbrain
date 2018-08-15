# Standard library imports
import warnings
from os.path import join

# Standard scientific libraries imports (more specific imports are
# delayed, so that the part module can be used without them).
import numpy as np

from nilearn._utils.numpy_conversions import as_ndarray
from nilearn._utils.niimg import _safe_get_data
from nilearn._utils.extmath import fast_abs_percentile
from nilearn.image import new_img_like
from nilearn.plotting import cm
from nilearn.plotting.displays import get_projector
from nilearn.plotting.img_plotting import _get_colorbar_and_data_ranges

from ..plot._base import TimeSlicer, Layout, EelFigure
from ._nifti_utils import _save_stc_as_volume

# default GlassBrain height and width
DEFAULT_H = 2.6
DEFAULT_W = 2.2


def _get_mne_source_space(ndvar):
    """
    load a mne source space from a ndvar source space
    """
    # mne src
    from mne import read_source_spaces
    return read_source_spaces (join(ndvar.source.subjects_dir, ndvar.source.subject, 'bem',
                                         '%s-%s-src.fif' % (ndvar.source.subject, ndvar.source.src)))


# Copied from nilear.plotting.img_plotting
# Weired that it could not be imported!
def _crop_colorbar(cbar, cbar_vmin, cbar_vmax):
    """
    crop a colorbar to show from cbar_vmin to cbar_vmax
    Used when symmetric_cbar=False is used.
    """
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


class GlassBrain(TimeSlicer, EelFigure):
    """`GlassBrain` Class to hold 2d projections of an ROI/mask image
    (by default 3 projections: Frontal, Axial, and Lateral).

    `GlassBrain` subclass comes with Eelbrain GUI integration and
    methods to visualize data in :class:`NDVar` format


    Parameters
    ----------
    ndvar : NDVar  ([case,] time, source[, space])
        Data to plot; if ``ndvar`` has a case dimension, the mean is plotted.
        if ``ndvar`` has a space dimension, the norm is plotted.
    dest : 'mri' | 'surf'
        If 'mri' the volume is defined in the coordinate system of
        the original T1 image. If 'surf' the coordinate system
        of the FreeSurfer surface is used (Surface RAS).
    mri_resolution: bool
        It True the image is saved in MRI resolution.
        WARNING: if you have many time points the file produced can be
        huge.
    black_bg : boolean, optional
        If True, the background of the image is set to be black.
    display_mode : string, optional. Default is 'ortho'
        Choose the direction of the cuts: 'x' - sagittal, 'y' - coronal,
        'z' - axial, 'l' - sagittal left hemisphere only,
        'r' - sagittal right hemisphere only, 'ortho' - three cuts are
        performed in orthogonal directions. Possible values are: 'ortho',
        'x', 'y', 'z', 'xz', 'yx', 'yz', 'l', 'r', 'lr', 'lzr', 'lyr',
        'lzry', 'lyrz'.
    threshold : a number, None, or 'auto'
        If None is given, the image is not thresholded.
        If a number is given, it is used to threshold the image:
        values below the threshold (in absolute value) are plotted
        as transparent. If auto is given, the threshold is determined
        magically by analysis of the image.
    cmap : matplotlib colormap, optional
        The colormap for specified image
    colorbar : boolean, optional
        If True, display a colorbar on the right of the plots.
    draw_cross: boolean, optional
        If draw_cross is True, a cross is drawn on the plot to
        indicate the cut plosition.
    annotate: boolean, optional
        If annotate is True, positions and left/right annotation
        are added to the plot.
    alpha : float between 0 and 1
        Alpha transparency for the brain schematics
    vmin : float
        Lower bound for plotting, passed to matplotlib.pyplot.imshow
    vmax : float
        Upper bound for plotting, passed to matplotlib.pyplot.imshow
    plot_abs : boolean, optional
        If set to True (default) maximum intensity projection of the
        absolute value will be used (rendering positive and negative
        values in the same manner). If set to false the sign of the
        maximum intensity will be represented with different colors.
        See http://nilearn.github.io/auto_examples/01_plotting/plot_demo_glass_brain_extensive.html
        for examples.
    symmetric_cbar : boolean or 'auto', optional, default 'auto'
        Specifies whether the colorbar should range from -vmax to vmax
        or from vmin to vmax. Setting to 'auto' will select the latter if
        the range of the whole image is either positive or negative.
        Note: The colormap will always be set to range from -vmax to vmax.
    interpolation : str
        Interpolation to use when resampling the image to the destination
        space. Can be "continuous" (default) to use 3rd-order spline
        interpolation, or "nearest" to use nearest-neighbor mapping.
        "nearest" is faster but can be noisier in some cases.
    h : scalar
        Plot height (inches).
    w : scalar
        Plot width (inches).

    Notes
    -----
    The plotted image should be in MNI space for this function to work
    properly.
    This function internally converts the ndvar (or its time-slices)
    to NIfTI image.
    """
    def __init__(self, ndvar, dest='mri', mri_resolution=False, black_bg=False, display_mode='ortho',
                 threshold='auto', cmap=None, colorbar=False, draw_cross=True, annotate=True, alpha=0.7,
                 vmin=None, vmax=None, plot_abs=True, symmetric_cbar="auto", interpolation='nearest', h=None, w=None,
                 **kwargs):

        self.kwargs0 = dict(dest=dest,
                            mri_resolution=mri_resolution)

        if cmap is None:
            cmap = cm.cold_hot if black_bg else cm.cold_white_hot
        self.cmap = cmap

        if ndvar:
            if ndvar.has_case:
                ndvar = ndvar.mean('case')
            # instead check ndvar
            if ndvar.has_dim('space'):
                ndvar = ndvar.norm('space')

            self._ndvar = ndvar
            self.src = _get_mne_source_space(ndvar)
            src_type = self.src[0]['type']
            if src_type != 'vol':
                raise ValueError('You need a volume source space. Got type: %s.'
                                  % src_type)

            if ndvar.has_dim ('time'):
                t_in = 0
                self.time = ndvar.get_dim ('time')
                ndvar0 = ndvar.sub(time=self.time[t_in])
                title = 'time = %s ms' % round (t_in * 1e3)
            else:
                self.time = None
                title = None
                ndvar0 = ndvar

            if plot_abs:
                cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges (
                    ndvar.x,
                    vmax,
                    symmetric_cbar,
                    kwargs,
                    0)
            else:
                cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges (
                    ndvar.x,
                    vmax,
                    symmetric_cbar,
                    kwargs)
        else:
            cbar_vmin, cbar_vmax = None, None

        show_nan_msg = False
        if vmax is not None and np.isnan (vmax):
            vmax = None
            show_nan_msg = True
        if vmin is not None and np.isnan (vmin):
            vmin = None
            show_nan_msg = True
        if show_nan_msg:
            nan_msg = ('NaN is not permitted for the vmax and vmin arguments.\n'
                       'Tip: Use np.nanmax() instead of np.max().')
            warnings.warn(nan_msg)

        img = _save_stc_as_volume(None, ndvar0, self.src, **self.kwargs0)
        data = _safe_get_data(img, ensure_finite=True)
        affine = img.affine

        if np.isnan(np.sum (data)):
            data = np.nan_to_num(data)

        # Deal with automatic settings of plot parameters
        if threshold == 'auto':
            # Threshold epsilon below a percentile value, to be sure that some
            # voxels pass the threshold
            threshold = fast_abs_percentile (data) - 1e-5

        img = new_img_like(img, as_ndarray (data), affine)

        # layout
        if w is None:
            w = DEFAULT_W
            w *= 3 if display_mode == 'ortho' else len(display_mode)
            if colorbar:
                w += .7
        if h is None:
            h = DEFAULT_H
        layout = Layout(None, ax_aspect=0, axh_default=0, h=h, w=w, tight=False)
        EelFigure.__init__(self, 'GlassBrain-%s'%ndvar.source.subject, layout)

        display = get_projector(display_mode)(img, alpha=alpha, plot_abs=plot_abs,
                                              threshold=threshold, figure=self.figure, axes=None,
                                              black_bg=black_bg, colorbar=colorbar)

        display.add_overlay(new_img_like(img, data, affine),
                            threshold=threshold, interpolation=interpolation,
                            colorbar=colorbar, vmin=vmin, vmax=vmax, cmap=cmap,
                            **kwargs)

        self.display = display
        self.threshold = threshold
        self.interpolation = interpolation
        self.colorbar = colorbar
        self.vmin = vmin
        self.vmax = vmax

        if annotate:
            display.annotate()
        if draw_cross:
            display.draw_cross()
        if title is not None and not title == '':
            display.title(title)
        if hasattr(display, '_cbar'):
            cbar = display._cbar
            _crop_colorbar(cbar, cbar_vmin, cbar_vmax)

        TimeSlicer.__init__(self, (ndvar,))

        self._show()

    def save_as(self, output_file):
        if output_file is not None:
            self.display.savefig(output_file)

    def close(self):
        self.display.close()

    # used by _update_time
    def _add_overlay(self, ndvar0, threshold, interpolation, vmin, vmax, cmap, **kwargs):
        img = _save_stc_as_volume(None, ndvar0, self.src, **self.kwargs0)
        data = _safe_get_data(img, ensure_finite=True)
        affine = img.affine

        if np.isnan(np.sum(data)):
            data = np.nan_to_num(data)

        img = new_img_like(img, as_ndarray (data), affine)
        self.display.add_overlay(new_img_like(img, data, affine),
                                 threshold=threshold, interpolation=interpolation,
                                 colorbar=False, vmin=vmin, vmax=vmax, cmap=cmap,
                                 **kwargs)
        # A little hack to make sure that the display
        # still has correct colorbar flag.
        self.display._colorbar = self.colorbar

        for axis in self.display._cut_displayed:
            self.display.axes[axis].ax.redraw_in_frame()

    # used by _update_time
    def _remove_overlay(self):
        for axis in self.display._cut_displayed:
            if len(self.display.axes[axis].ax.images) > 0:
                self.display.axes[axis].ax.images[-1].remove()

    # used by _update_time
    def _update_title(self, t):
        first_axis = self.display._cut_displayed[0]
        ax = self.display.axes[first_axis].ax
        ax.texts[-1].set_text('time = %s ms' % round(t * 1e3))
        ax.redraw_in_frame()

    def _update_time(self, t, fixate):
        ndvart = self._ndvar.sub(time=t)

        # remove the last overlay
        self._remove_overlay()

        self._add_overlay(ndvart, threshold=self.threshold, interpolation=self.interpolation,
                          vmin=self.vmin, vmax=self.vmax, cmap=self.cmap)

        self._update_title(t)

        self._show()

    def animate(self):
        for t in self.time:
            self.set_time(t)

    # this is only needed for Eelbrain < 0.28
    def set_time(self, time):
        """Set the time point to display

        Parameters
        ----------
        time : scalar
            Time to display.
        """
        self._set_time(time, True)


def butterfly(ndvar, dest='mri', mri_resolution=False, black_bg=False, display_mode='lyrz',
               threshold='auto', cmap=None, colorbar=False, alpha=0.7, vmin=None, vmax=None, plot_abs=True,
               symmetric_cbar="auto", interpolation='nearest', name=None, h=2.5, w=5, **kwargs):
    """Shortcut for a Butterfly-plot with a time-linked glassbrain plot

    Parameters
    ----------
    ndvar : NDVar  ([case,] time, source[, space])
        Data to plot; if ``ndvar`` has a case dimension, the mean is plotted.
        if ``ndvar`` has a space dimension, the norm is plotted.
    dest : 'mri' | 'surf'
        If 'mri' the volume is defined in the coordinate system of
        the original T1 image. If 'surf' the coordinate system
        of the FreeSurfer surface is used (Surface RAS).
    mri_resolution: bool
        It True the image is saved in MRI resolution.
        WARNING: if you have many time points the file produced can be
        huge.
    black_bg : boolean, optional
        If True, the background of the image is set to be black.
    display_mode : string, optional. Default is 'lyrz'
        Choose the direction of the cuts: 'x' - sagittal, 'y' - coronal,
        'z' - axial, 'l' - sagittal left hemisphere only,
        'r' - sagittal right hemisphere only, 'ortho' - three cuts are
        performed in orthogonal directions. Possible values are: 'ortho',
        'x', 'y', 'z', 'xz', 'yx', 'yz', 'l', 'r', 'lr', 'lzr', 'lyr',
        'lzry', 'lyrz'.
    threshold : a number, None, or 'auto'
        If None is given, the image is not thresholded.
        If a number is given, it is used to threshold the image:
        values below the threshold (in absolute value) are plotted
        as transparent. If auto is given, the threshold is determined
        magically by analysis of the image.
    cmap : matplotlib colormap, optional
        The colormap for specified image
    colorbar : boolean, optional
        If True, display a colorbar on the right of the plots.
    alpha : float between 0 and 1
        Alpha transparency for the brain schematics
    vmin : float
        Lower bound for plotting, passed to matplotlib.pyplot.imshow
    vmax : float
        Upper bound for plotting, passed to matplotlib.pyplot.imshow
    plot_abs : boolean, optional
        If set to True (default) maximum intensity projection of the
        absolute value will be used (rendering positive and negative
        values in the same manner). If set to false the sign of the
        maximum intensity will be represented with different colors.
        See http://nilearn.github.io/auto_examples/01_plotting/plot_demo_glass_brain_extensive.html
        for examples.
    symmetric_cbar : boolean or 'auto', optional, default 'auto'
        Specifies whether the colorbar should range from -vmax to vmax
        or from vmin to vmax. Setting to 'auto' will select the latter if
        the range of the whole image is either positive or negative.
        Note: The colormap will always be set to range from -vmax to vmax.
    interpolation : str
        Interpolation to use when resampling the image to the destination
        space. Can be "continuous" (default) to use 3rd-order spline
        interpolation, or "nearest" to use nearest-neighbor mapping.
        "nearest" is faster but can be noisier in some cases.
    name : str
        The window title (default is ndvar.name).
    h : scalar
        Plot height (inches).
    w : scalar
        Butterfly plot width (inches).

    Returns
    -------
    butterfly_plot : plot.Butterfly
        Butterfly plot.
    glassbrain : GlassBrain
        GlassBrain plot.
    """

    from ..plot._utsnd import Butterfly

    if name is None:
        name = ndvar.name

    if ndvar.has_case:
        ndvar = ndvar.mean('case')

    # butterfly-plot
    if ndvar.has_dim('space'):
        data = ndvar.norm('space')
    else:
        data = ndvar
    p = Butterfly(data, vmin=vmin, vmax=vmax, h=h, w=w, name=name, color='black', ylabel=False)

    # position the brain window next to the butterfly-plot
    # needs to be figured out

    # GlassBrain plot
    p_glassbrain = GlassBrain(ndvar, display_mode=display_mode, colorbar=colorbar, threshold=threshold,
                              dest=dest, mri_resolution=mri_resolution, draw_cross=True, annotate=True,
                              black_bg=black_bg, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax,
                              plot_abs=plot_abs, symmetric_cbar=symmetric_cbar, interpolation=interpolation,
                              **kwargs)

    p.link_time_axis(p_glassbrain)

    return p, p_glassbrain


