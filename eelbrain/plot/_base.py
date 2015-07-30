# -*- coding: utf-8 -*-
"""
Implementation
==============

Plotting is implemented hierarchically in 3 different types of
functions/classes:

top-level (public names)
    Top-level functions or classes have public names create an entire figure.
    Some classes also retain the figure and provide methods for manipulating
    it.

_ax_
    Functions beginning with _ax_ organize an axes object. They do not
    create their own axes object (this is provided by the top-level function),
    but change axes formatting such as labels and extent.

_plt_
    Functions beginning with _plt_ only plot data to a given axes object
    without explicitly changing aspects of the axes themselves.


Top-level plotters can be called with nested lists of data-objects (NDVar
instances). They create a separate axes for each list element. Axes
themselves can have multiple layers (e.g., a difference map visualized through
a colormap, and significance levels indicated by contours).


Example: t-test
---------------

For example, the default plot for testnd.ttest() results is the
following list (assuming the test compares A and B):

``[A, B, [diff(A,B), p(A, B)]]``

where ``diff(...)`` is a difference map and ``p(...)`` is a map of p-values.
The main plot function creates a separate axes object for each list element:

- ``A``
- ``B``
- ``[diff(A,B), p(A, B)]``

Each of these element is then plotted with the corresponding _ax_ function.
The _ax_ function calls _plt_ for each of its input elements. Thus, the
functions executed are:

#. plot([A, B, [diff(A,B), p(A, B)]])
#. ---> _ax_(A)
#. ------> _plt_(A)
#. ---> _ax_(B)
#. ------> _plt_(B)
#. ---> _ax_([diff(A,B), p(A, B)])
#. ------> _plt_(diff(A,B))
#. ------> _plt_(p(A, B))

"""
from __future__ import division
import __main__

from itertools import chain
import math
import os
import shutil
import subprocess
import tempfile

import matplotlib as mpl
from matplotlib.figure import SubplotParams
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, ScalarFormatter
import numpy as np
import PIL

from .._utils.subp import command_exists
from ..fmtxt import Image, texify
from .._colorspaces import symmetric_cmaps, zerobased_cmaps
from .._data_obj import ascategorial, asndvar, isnumeric, cellname, \
    DimensionMismatchError


# defaults
defaults = {'maxw': 16, 'maxh': 10}
backend = {'eelbrain': True, 'autorun': None, 'show': True}

# store figures (they need to be preserved)
figures = []

# constants
default_cmap = None
default_meas = '?'


def do_autorun(run=None):
    # http://stackoverflow.com/a/2356420/166700
    if run is not None:
        return run
    elif backend['autorun'] is None:
        return not hasattr(__main__, '__file__')
    else:
        backend['autorun']


def configure(frame=True, autorun=None, show=True):
    """Set basic configuration parameters for the current session

    Parameters
    ----------
    frame : bool
        Open figures in the Eelbrain application. This provides additional
        functionality such as copying a figure to the clipboard. If False, open
        figures as normal matplotlib figures.
    autorun : bool
        When a figure is created, automatically enter the GUI mainloop. By
        default, this is True when the figure is created in interactive mode
        but False when the figure is created in a script (in order to run the
        GUI at a specific point in a script, call :func:`eelbrain.gui.run`).
    show : bool
        Show plots on the screen when they're created (disable this to create
        plots and save them without showing them on the screen).
    """
    if autorun is not None:
        autorun = bool(autorun)
    backend['eelbrain'] = bool(frame)
    backend['autorun'] = autorun
    backend['show'] = bool(show)


meas_display_unit = {'time': u'ms',
                     'V': u'µV',
                     'B': u'fT',
                     'sensor': int}
unit_format = {u'ms': 1e3,
               u'mV': 1e3,
               u'µV': 1e6,
               u'pT': 1e12,
               u'fT': 1e15,
               u'dSPM': 1,
               int: int}
scale_formatters = {1: ScalarFormatter(),
                    1e3: FuncFormatter(lambda x, pos: '%g' % (1e3 * x)),
                    1e6: FuncFormatter(lambda x, pos: '%g' % (1e6 * x)),
                    1e9: FuncFormatter(lambda x, pos: '%g' % (1e9 * x)),
                    1e12: FuncFormatter(lambda x, pos: '%g' % (1e12 * x)),
                    1e15: FuncFormatter(lambda x, pos: '%g' % (1e15 * x)),
                    int: FormatStrFormatter('%i')}


def find_axis_params_data(v, label):
    """

    Parameters
    ----------
    v : NDVar | Var | str | scalar
        Unit or scale of the axis.

    Returns
    -------
    tick_formatter : Formatter
        Matplotlib axis tick formatter.
    label : str | None
        Axis label.
    """
    if isinstance(v, basestring):
        if v in unit_format:
            scale = unit_format[v]
            unit = v
        else:
            raise ValueError("Unknown unit: %s" % repr(v))
    elif isinstance(v, float):
        scale = v
        unit = None
    elif isnumeric(v):
        meas = v.info.get('meas', None)
        data_unit = v.info.get('unit', None)
        if meas in meas_display_unit:
            unit = meas_display_unit[meas]
            scale = unit_format[unit]
            if data_unit in unit_format:
                scale /= unit_format[data_unit]
        else:
            scale = 1
            unit = data_unit
    else:
        raise TypeError("unit=%s" % repr(v))

    if label is True:
        if meas and unit and meas != unit:
            label = '%s [%s]' % (meas, unit)
        elif meas:
            label = meas
        elif unit:
            label = unit
        else:
            label = getattr(v, 'name', None)

    return scale_formatters[scale], label


def find_axis_params_dim(meas, label):
    """Find an axis label

    Parameters
    ----------
    dimname : str
        Name of the dimension.
    label : None | True | str
        Label argument.

    Returns
    -------
    label : str | None
        Returns the default axis label if label==True, otherwise the label
        argument.
    """
    if meas in meas_display_unit:
        unit = meas_display_unit[meas]
        scale = unit_format[unit]
        if label is True:
            if isinstance(unit, basestring):
                label = "%s [%s]" % (meas.capitalize(), unit)
            else:
                label = meas.capitalize()
    else:
        scale = 1
        if label is True:
            label = meas.capitalize()

    return scale_formatters[scale], label


def find_ct_args(ndvar, overlay, contours={}):
    """Construct a dict with kwargs for a contour plot

    Parameters
    ----------
    ndvar : NDVar
        Data to be plotted.
    overlay : bool
        Whether the NDVar is plotted as a first layer or as an overlay.
    contours : dict
        Externally specified contours as {meas: {level: color}} mapping.

    Returns
    -------
    ct_args : dict
        {level: color} mapping for contour plots.

    Notes
    -----
    The NDVar's info dict contains default arguments that determine how the
    NDVar is plotted as base and as overlay. In case of insufficient
    information, defaults apply. On the other hand, defaults can be overridden
    by providing specific arguments to plotting functions.
    """
    if overlay:
        kind = ndvar.info.get('overlay', ('contours',))
    else:
        kind = ndvar.info.get('base', ())

    ct_args = {}
    if 'contours' in kind:
        info_ct = ndvar.info.get('contours', None)
        if overlay:
            info_ct = ndvar.info.get('overlay_contours', info_ct)
        else:
            info_ct = ndvar.info.get('base_contours', info_ct)

        if info_ct:
            ct_args.update(info_ct)

    meas = ndvar.info.get('meas', default_meas)
    if meas in contours:
        ct_args.update(contours[meas])

    return ct_args


def find_im_args(ndvar, overlay, vlims={}, cmaps={}):
    """Construct a dict with kwargs for an im plot

    Parameters
    ----------
    ndvar : NDVar
        Data to be plotted.
    overlay : bool
        Whether the NDVar is plotted as a first layer or as an overlay.
    vlims : dict
        {(meas, cmap): (vmax, vmin)} mapping to replace v-limits based on the
        ndvar.info dict.
    cmaps : dict
        {meas: cmap} mapping to replace the cmap in the ndvar.info dict.

    Returns
    -------
    im_args : dict
        Arguments for the im plot (cmap, vmin, vmax).

    Notes
    -----
    The NDVar's info dict contains default arguments that determine how the
    NDVar is plotted as base and as overlay. In case of insufficient
    information, defaults apply. On the other hand, defaults can be overridden
    by providing specific arguments to plotting functions.
    """
    if overlay:
        kind = ndvar.info.get('overlay', ('contours',))
    else:
        kind = ndvar.info.get('base', ('im',))

    if 'im' in kind:
        if 'meas' in ndvar.info:
            meas = ndvar.info['meas']
        else:
            meas = default_meas

        if meas in cmaps:
            cmap = cmaps[meas]
        elif 'cmap' in ndvar.info:
            cmap = ndvar.info['cmap']
        else:
            cmap = default_cmap

        key = (meas, cmap)
        if key in vlims:
            vmin, vmax = vlims[key]
        else:
            vmin, vmax = find_vlim_args(ndvar)
            vmin, vmax = fix_vlim_for_cmap(vmin, vmax, cmap)
        im_args = {'cmap': cmap, 'vmin': vmin, 'vmax': vmax}
    else:
        im_args = None

    return im_args


def find_uts_args(ndvar, overlay, color=None):
    """Construct a dict with kwargs for a uts plot

    Parameters
    ----------
    ndvar : NDVar
        Data to be plotted.
    overlay : bool
        Whether the NDVar is plotted as a first layer or as an overlay.
    vlims : dict
        Vmax and vmin values by (meas, cmap).

    Returns
    -------
    uts_args : dict
        Arguments for a uts plot (color).

    Notes
    -----
    The NDVar's info dict contains default arguments that determine how the
    NDVar is plotted as base and as overlay. In case of insufficient
    information, defaults apply. On the other hand, defaults can be overridden
    by providing specific arguments to plotting functions.
    """
    if overlay:
        kind = ndvar.info.get('overlay', ())
    else:
        kind = ndvar.info.get('base', ('trace',))

    if 'trace' in kind:
        args = {}
        color = color or ndvar.info.get('color', None)
        if color is not None:
            args['color'] = color
    else:
        args = None

    return args


def find_uts_hlines(ndvar):
    """Find horizontal lines for uts plots (based on contours)

    Parameters
    ----------
    ndvar : NDVar
        Data to be plotted.

    Returns
    -------
    h_lines : iterator
        Iterator over (y, kwa) tuples.
    """
    contours = ndvar.info.get('contours', None)
    if contours:
        for level in sorted(contours):
            args = contours[level]
            if isinstance(args, dict):
                yield level, args.copy()
            else:
                yield level, {'color': args}


def find_uts_ax_vlim(layers, vlims={}):
    """Find y axis limits for uts axes

    Parameters
    ----------
    layers : list of NDVar
        Data to be plotted.
    vlims : dict
        Vmax and vmin values by (meas, cmap).

    Returns
    -------
    bottom : None | scalar
        Lowest value on y axis.
    top : None | scalar
        Highest value on y axis.
    """
    bottom = None
    top = None
    overlay = False
    for ndvar in layers:
        if overlay:
            kind = ndvar.info.get('overlay', ())
        else:
            kind = ndvar.info.get('base', ('trace',))
        overlay = True

        if 'trace' not in kind:
            continue

        meas = ndvar.info.get('meas', default_meas)
        cmap = ndvar.info.get('cmap', default_cmap)
        key = (meas, cmap)
        if key in vlims:
            bottom_, top_ = vlims[key]
            if bottom is None:
                bottom = bottom_
            elif bottom_ != bottom:
                raise RuntimeError("Double vlim specification")
            if top is None:
                top = top_
            elif top_ != top:
                raise RuntimeError("Double vlim specification")

    return bottom, top


def find_fig_vlims(plots, range_by_measure=False, vmax=None, vmin=None):
    """Find vmin and vmax parameters for every (meas, cmap) combination

    Parameters
    ----------
    plots : nested list of NDVar
        Unpacked plot data.
    range_by_measure : bool
        Constrain the vmax - vmin range such that the range is constant within
        measure (for uts plots).
    vmax : None | dict | scalar
        Dict: predetermined vlims (take precedence). Scalar: user-specified
        vmax parameter (used for for the first meas kind).
    vmin : None | scalar
        User-specified vmin parameter. If vmax is user-specified but vmin is
        None, -vmax is used.

    Returns
    -------
    vlims : dict
        Dictionary of im limits: {(meas, cmap): (vmin, vmax)}.
    """
    if isinstance(vmax, dict):
        vlims = vmax
        user_vlim = None
    else:
        vlims = {}
        if vmax is None:
            user_vlim = None
        elif vmin is None:
            user_vlim = (vmax, 0)
        else:
            user_vlim = (vmax, vmin)

    out = {}  # (meas, cmap): (vmin, vmax)
    first_meas = None  # what to use user-specified vmax for
    for ndvar in chain(*plots):
        meas = ndvar.info.get('meas', '?')
        if user_vlim is not None and first_meas is None:
            first_meas = meas
            vmin, vmax = user_vlim
        else:
            vmin, vmax = find_vlim_args(ndvar)
        cmap = ndvar.info.get('cmap', None)
        key = (meas, cmap)
        if key in vlims:
            continue
        elif user_vlim is not None and meas == first_meas:
            vmax, vmin = user_vlim
        elif key in out:
            vmin_, vmax_ = out[key]
            vmin = min(vmin, vmin_)
            vmax = max(vmax, vmax_)
        vmin, vmax = fix_vlim_for_cmap(vmin, vmax, cmap)
        out[key] = (vmin, vmax)

    out.update(vlims)

    if range_by_measure:
        range_ = {}
        for (meas, cmap), (vmin, vmax) in out.iteritems():
            r = vmax - vmin
            range_[meas] = max(range_.get(meas, 0), r)
        for key in out.keys():
            meas, cmap = key
            vmin, vmax = out[key]
            diff = range_[meas] - (vmax - vmin)
            if diff:
                if cmap in zerobased_cmaps:
                    vmax += diff
                else:
                    diff /= 2
                    vmax += diff
                    vmin -= diff
                out[key] = vmin, vmax

    return out


def find_vlim_args(ndvar, vmin=None, vmax=None):
    if vmax is None:
        vmax = ndvar.info.get('vmax', None)
        if vmin is None:
            vmin = ndvar.info.get('vmin', None)

    if vmax is None:
        xmax = np.nanmax(ndvar.x)
        xmin = np.nanmin(ndvar.x)
        abs_max = max(abs(xmax), abs(xmin)) or 1e-14
        scale = math.floor(np.log10(abs_max))
        vmax = math.ceil(xmax * 10 ** -scale) * 10 ** scale
        vmin = math.floor(xmin * 10 ** -scale) * 10 ** scale

    return vmin, vmax


def fix_vlim_for_cmap(vmin, vmax, cmap):
    "Fix the vmin value to yield an appropriate range for the cmap"
    if cmap in symmetric_cmaps:
        if vmax is None and vmin is None:
            pass
        elif vmin is None:
            vmax = abs(vmax)
            vmin = -vmax
        elif vmax is None:
            vmax = abs(vmin)
            vmin = -vmax
        else:
            vmax = max(abs(vmax), abs(vmin))
            vmin = -vmax
    elif cmap in zerobased_cmaps:
        vmin = 0
    return vmin, vmax


def find_data_dims(ndvar, dims):
    """Find dimensions in data
    """
    if isinstance(dims, int):
        if ndvar.ndim == dims:
            return ndvar.dimnames
        elif ndvar.ndim - 1 == dims:
            return ndvar.dimnames[1:]
        else:
            raise ValueError("NDVar does not have the right number of dimensions")
    else:
        if len(dims) == ndvar.ndim:
            all_dims = list(ndvar.dimnames)
        elif len(dims) == ndvar.ndim - 1 and ndvar.has_case:
            all_dims = list(ndvar.dimnames[1:])
        else:
            raise ValueError("NDVar does not have the right number of dimensions")

        out_dims = []
        for dim in dims:
            if dim is None:
                for dim in all_dims:
                    if dim not in dims:
                        break
                else:
                    raise ValueError("NDVar does not have requested dimensions %s" % repr(dims))
            elif dim not in all_dims:
                raise ValueError("NDVar does not have requested dimension %s" % dim)
            out_dims.append(dim)
            all_dims.remove(dim)
    return out_dims


def unpack_epochs_arg(Y, dims, Xax=None, ds=None):
    """Unpack the first argument to top-level NDVar plotting functions

    Parameters
    ----------
    Y : NDVar | list
        the first argument.
    dims : int | tuple
        The number of dimensions needed for the plotting function, or tuple
        with dimension entries (str | None).
    Xax : None | categorial
        A model to divide Y into different axes. Xax is currently applied on
        the first level, i.e., it assumes that Y's first dimension is cases.
    ds : None | Dataset
        Dataset containing data objects which are provided as str.

    Returns
    -------
    axes_data : list of list of NDVar
        The processed data to plot.
    dims : tuple of str
        Names of the dimensions.

    Notes
    -----
    Ndvar plotting functions above 1-d UTS level should support the following
    API:

     - simple NDVar: summary ``plot(meg)``
     - list of ndvars: summary for each ``plot(meg.as_list())``
     - NDVar and Xax argument: summary for each  ``plot(meg, Xax=subject)
     - nested list of layers (e.g., ttest results: [c1, c0, [c1-c0, p]])
    """
    # get proper Y
    if hasattr(Y, '_default_plot_obj'):
        Y = Y._default_plot_obj

    if isinstance(Y, (tuple, list)):
        data_dims = None
        if isinstance(dims, int):
            ndims = dims
        else:
            ndims = len(dims)
    else:
        Y = asndvar(Y, ds=ds)
        data_dims = find_data_dims(Y, dims)
        ndims = len(data_dims)

    if Xax is not None and isinstance(Y, (tuple, list)):
        err = ("Xax can only be used to divide Y into different axes if Y is "
               "a single NDVar (got a %s)." % Y.__class__.__name__)
        raise TypeError(err)

    # create list of plots
    if isinstance(Xax, str) and Xax.startswith('.'):
        dimname = Xax[1:]
        if dimname == 'case':
            if not Y.has_case:
                err = ("Xax='.case' supplied, but Y does not have case "
                       "dimension")
                raise ValueError(err)
            values = range(len(Y))
            unit = ''
        else:
            dim = Y.get_dim(dimname)
            values = dim.values
            unit = getattr(dim, 'unit', '')

        name = dimname.capitalize() + ' = %s'
        if unit:
            name += ' ' + unit
        axes = [Y.sub(name=name % v, **{dimname: v}) for v in values]
    elif Xax is not None:
        Xax = ascategorial(Xax, ds=ds)
        axes = []
        for cell in Xax.cells:
            v = Y[Xax == cell]
            v.name = cell
            axes.append(v)
    elif isinstance(Y, (tuple, list)):
        axes = Y
    else:
        axes = [Y]

    axes = [unpack_ax(ax, ndims, ds) for ax in axes]
    if data_dims is None:
        for layers in axes:
            for l in layers:
                if data_dims is None:
                    data_dims = find_data_dims(l, dims)
                else:
                    find_data_dims(l, data_dims)

    return axes, data_dims


def unpack_ax(ax, ndim, ds):
    # returns list of NDVar
    if isinstance(ax, (tuple, list)):
        return [_unpack_layer(layer, ndim, ds) for layer in ax]
    else:
        return [_unpack_layer(ax, ndim, ds)]

def _unpack_layer(y, ndim, ds):
    # returns NDVar
    ndvar = asndvar(y, ds=ds)

    if ndvar.ndim == ndim + 1:
        if ndvar.has_case:
            ndvar = ndvar.mean('case')

    if ndvar.ndim != ndim:
        err = ("Plot requires ndim=%i, got %r with ndim=%i" %
               (ndim, ndvar, ndvar.ndim))
        raise DimensionMismatchError(err)

    return ndvar


def str2tex(txt):
    """If matplotlib usetex is enabled, replace tex sensitive characters in the
    string.
    """
    if txt and mpl.rcParams['text.usetex']:
        return texify(txt)
    else:
        return txt


class mpl_figure:
    "cf. _wxgui.mpl_canvas"
    def __init__(self, **fig_kwargs):
        "creates self.figure and self.canvas attributes and returns the figure"
        from matplotlib import pyplot

        self._plt = pyplot
        self.figure = pyplot.figure(**fig_kwargs)
        self.canvas = self.figure.canvas

    def Close(self):
        self._plt.close(self.figure)

    def SetStatusText(self, text):
        pass

    def Show(self):
        if mpl.get_backend() == 'WXAgg' and do_autorun():
            self._plt.show()

    def redraw(self, axes=[], artists=[]):
        "Adapted duplicate of mpl_canvas.FigureCanvasPanel"
        self.canvas.restore_region(self._background)
        for ax in axes:
            ax.draw_artist(ax)
            extent = ax.get_window_extent()
            self.canvas.blit(extent)
        for artist in artists:
            ax = artist.get_axes()
            ax.draw_artist(ax)
            extent = ax.get_window_extent()
            self.canvas.blit(extent)

    def store_canvas(self):
        self._background = self.canvas.copy_from_bbox(self.figure.bbox)


# MARK: figure composition

def _loc(name, size=(0, 0), title_space=0, frame=.01):
    """
    takes a loc argument and returns x,y of bottom left edge

    """
    if isinstance(name, basestring):
        y, x = name.split()
    # interpret x
    elif len(name) == 2:
        x, y = name
    else:
        raise NotImplementedError("loc needs to be string or len=2 tuple/list")
    if isinstance(x, basestring):
        if x == 'left':
            x = frame
        elif x in ['middle', 'center', 'centre']:
            x = .5 - size[0] / 2
        elif x == 'right':
            x = 1 - frame - size[0]
        else:
            raise ValueError(x)
    # interpret y
    if isinstance(y, basestring):
        if y in ['top', 'upper']:
            y = 1 - frame - title_space - size[1]
        elif y in ['middle', 'center', 'centre']:
            y = .5 - title_space / 2. - size[1] / 2.
        elif y in ['lower', 'bottom']:
            y = frame
        else:
            raise ValueError(y)
    return x, y


def frame_title(plot, y, x=None, xax=None):
    """Generate frame title from common data structure

    Parameters
    ----------
    plot : str
        Name of the plot.
    y : data-obj
        Dependent variable.
    x : data-obj
        Predictor.
    xax : data-obj
        Grouping variable for axes.
    """
    if xax is None:
        if x is None:
            return "%s: %s" % (plot, y.name)
        else:
            return "%s: %s ~ %s" % (plot, y.name, x.name)
    elif x is None:
        return "%s: %s | %s" % (plot, y.name, xax.name)
    else:
        return "%s: %s ~ %s | %s" % (plot, y.name, x.name, xax.name)


class _EelFigure(object):
    """Parent class for Eelbrain figures.

    In order to subclass:

     - find desired figure properties and then use them to initialize
       the _EelFigure superclass; then use the
       :py:attr:`_EelFigure.figure` and :py:attr:`_EelFigure.canvas` attributes.
     - end the initialization by calling `_EelFigure._show()`
     - add the :py:meth:`_fill_toolbar` method
    """
    _default_format = 'png'  # default format when saving for fmtext
    _default_xlabel_ax = -1
    _default_ylabel_ax = 0
    _make_axes = True

    def __init__(self, frame_title, nax, axh_default, ax_aspect, tight=True,
                 title=None, frame=True, yaxis=True, *args, **kwargs):
        """Parent class for Eelbrain figures.

        Parameters
        ----------
        frame_title : str
            Frame title.
        nax : None | int
            Number of axes to produce layout for. If None, no layout is
            produced.
        axh_default : scalar
            Default height per axes.
        ax_aspect : scalar
            Width to height ration (axw / axh).
        tight : bool
            Rescale axes so that the space in the figure is used optimally
            (default True).
        title : str
            Figure title (default is no title).
        frame : bool | 't'
            How to frame the plots.
            ``True`` (default): normal matplotlib frame;
            ``False``: omit top and right lines;
            ``'t'``: draw spines at x=0 and y=0, common for ERPs.
        yaxis : bool
            Draw the y-axis (default True).
        h : scalar
            Height of the figure.
        w : scalar
            Width of the figure.
        axh : scalar
            Height of the axes.
        axw : scalar
            Width of the axes.
        nrow : int
            Set a limit to the number of rows (default is no limit).
        ncol : int
            Set a limit to the number of columns (defaut is no limit). If
            neither nrow or ncol is specified, a square layout is preferred.
        dpi : int
            DPI for the figure (default is to use matplotlib rc parameters).
        show : bool
            Show the figure in the GUI (default True). Use False for creating
            figures and saving them without displaying them on the screen.
        run : bool
            Run the Eelbrain GUI app (default is True for interactive plotting and
            False in scripts).
        """
        if title:
            frame_title = '%s: %s' % (frame_title, title)

        # layout
        layout = Layout(nax, ax_aspect, axh_default, tight, *args, **kwargs)

        # find the right frame
        if backend['eelbrain']:
            from .._wxgui import get_app
            from .._wxgui.mpl_canvas import CanvasFrame
            get_app()
            frame_ = CanvasFrame(None, frame_title, eelfigure=self, **layout.fig_kwa)
        else:
            frame_ = mpl_figure(**layout.fig_kwa)

        figure = frame_.figure
        if title:
            self._figtitle = figure.suptitle(title)
        else:
            self._figtitle = None

        # make axes
        axes = []
        if self._make_axes and nax is not None:
            for i in xrange(1, nax + 1):
                ax = figure.add_subplot(layout.nrow, layout.ncol, i)
                axes.append(ax)

                # axes modifications
                if frame == 't':
                    ax.tick_params(direction='inout', bottom=False, top=True,
                                   left=False, right=True, labelbottom=True,
                                   labeltop=False, labelleft=True,
                                   labelright=False)
                    ax.spines['right'].set_position('zero')
                    ax.spines['left'].set_visible(False)
                    ax.spines['top'].set_position('zero')
                    ax.spines['bottom'].set_visible(False)
                elif not frame:
                    ax.yaxis.set_ticks_position('left')
                    ax.spines['right'].set_visible(False)
                    ax.xaxis.set_ticks_position('bottom')
                    ax.spines['top'].set_visible(False)

                if not yaxis:
                    ax.yaxis.set_ticks(())
                    ax.spines['left'].set_visible(False)

        # store attributes
        self._frame = frame_
        self.figure = figure
        self._axes = axes
        self.canvas = frame_.canvas
        self._layout = layout
        self._tight_arg = tight

        # add callbacks
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.canvas.mpl_connect('axes_leave_event', self._on_leave_axes)

    def _show(self):
        if self._tight_arg:
            self._tight()

        self.draw()
        if backend['show'] and self._layout.show:
            self._frame.Show()
            if backend['eelbrain'] and do_autorun(self._layout.run):
                from .._wxgui import run
                run()

    def _tight(self):
        "Default implementation based on matplotlib"
        self.figure.tight_layout()
        if self._figtitle:
            trans = self.figure.transFigure.inverted()
            extent = self._figtitle.get_window_extent(self.figure.canvas.renderer)
            bbox = trans.transform(extent)
            t_bottom = bbox[0, 1]
            self.figure.subplots_adjust(top=1 - 2 * (1 - t_bottom))

    def _on_leave_axes(self, event):
        "update the status bar when the cursor leaves axes"
        self._frame.SetStatusText(':-)')

    def _on_motion(self, event):
        "update the status bar for mouse movement"
        ax = event.inaxes
        if ax:
            x = ax.xaxis.get_major_formatter().format_data(event.xdata)
            y = ax.yaxis.get_major_formatter().format_data(event.ydata)
            self._frame.SetStatusText('x = %s, y = %s' % (x, y))

    def _fill_toolbar(self, tb):
        """
        Subclasses should add their toolbar items in this function which
        is called by CanvasFrame.FillToolBar()

        """
        pass

    def close(self):
        "Close the figure."
        self._frame.Close()

    def _configure_xaxis_dim(self, meas, label, ticklabels, axes=None):
        """Configure the x-axis based on a dimension

        Parameters
        ----------
        meas : str
            The measure assigned to this axis.
        label : None | str
            Axis label.
        ticklabels : bool
            Whether to print tick-labels.
        axes : list of Axes
            Axes which to format (default is EelFigure._axes)
        """
        if axes is None:
            axes = self._axes
        formatter, label = find_axis_params_dim(meas, label)

        if ticklabels:
            for ax in axes:
                ax.xaxis.set_major_formatter(formatter)
        else:
            for ax in axes:
                ax.xaxis.set_ticklabels(())

        if label:
            self.set_xlabel(label)

    def _configure_xaxis(self, v, label, axes=None):
        if axes is None:
            axes = self._axes
        formatter, label = find_axis_params_data(v, label)
        for ax in axes:
            ax.xaxis.set_major_formatter(formatter)

        if label:
            self.set_xlabel(label)

    def _configure_yaxis_dim(self, meas, label, axes=None):
        "Configure the y-axis based on a dimension"
        if axes is None:
            axes = self._axes
        formatter, label = find_axis_params_dim(meas, label)
        for ax in axes:
            ax.yaxis.set_major_formatter(formatter)

        if label:
            self.set_ylabel(label)

    def _configure_yaxis(self, v, label, axes=None):
        if axes is None:
            axes = self._axes
        formatter, label = find_axis_params_data(v, label)
        for ax in axes:
            ax.yaxis.set_major_formatter(formatter)

        if label:
            self.set_ylabel(label)

    def draw(self):
        "(Re-)draw the figure (after making manual changes)."
        self._frame.canvas.draw()

    def _asfmtext(self):
        return self.image()

    def image(self, name=None, format=None):
        """Create FMTXT Image from the figure

        Parameters
        ----------
        name : str
            Name for the file (without extension; default is 'image').
        format : str
            File format (default 'png').

        Returns
        -------
        image : fmtxt.Image
            Image FMTXT object.
        """
        if format is None:
            format = self._default_format

        image = Image(name, format)
        self.figure.savefig(image, format=format)
        return image

    def save(self, *args, **kwargs):
        "Short-cut for Matplotlib's :meth:`~matplotlib.figure.Figure.savefig()`"
        self.figure.savefig(*args, **kwargs)

    def set_xtick_rotation(self, rotation):
        """Rotate every x-axis tick-label by an angle (counterclockwise, in degrees)

        Parameters
        ----------
        rotation : scalar
            Counterclockwise rotation angle, in degrees.
        """
        for ax in self._axes:
            for t in ax.get_xticklabels():
                t.set_rotation(rotation)
        self.draw()

    def set_xlabel(self, label, ax=None):
        """Set the label for the x-axis

        Parameters
        ----------
        label : str
            X-axis label.
        ax : int
            Axis on which to set the label (default is usually the last axis).
        """
        if ax is None:
            ax = self._default_xlabel_ax
        self._axes[ax].set_xlabel(label)

    def set_ylabel(self, label, ax=None):
        """Set the label for the y-axis

        Parameters
        ----------
        label : str
            Y-axis label.
        ax : int
            Axis on which to set the label (default is usually the first axis).
        """
        if ax is None:
            ax = self._default_ylabel_ax
        self._axes[ax].set_ylabel(label)


class Layout():
    """Create layouts for figures with several axes of the same size
    """
    def __init__(self, nax, ax_aspect, axh_default, tight, h=None, w=None,
                 axh=None, axw=None, nrow=None, ncol=None, dpi=None, show=True,
                 run=None):
        """Create a grid of axes based on variable parameters.

        Parameters
        ----------
        nax : int
            Number of axes required.
        ax_aspect : scalar
            Width / height aspect of the axes.
        axh_default : scalar
            The default axes height if it can not be determined from the other
            parameters.
        tight : bool
            Rescale axes so that the space in the figure is used optimally
            (default True).
        h : scalar
            Height of the figure.
        w : scalar
            Width of the figure.
        axh : scalar
            Height of the axes.
        axw : scalar
            Width of the axes.
        nrow : int
            Set a limit to the number of rows (default is no limit).
        ncol : int
            Set a limit to the number of columns (defaut is no limit). If
            neither nrow or ncol is specified, a square layout is preferred.
        dpi : int
            DPI for the figure (default is to use matplotlib rc parameters).
        show : bool
            Show the figure in the GUI (default True). Use False for creating
            figures and saving them without displaying them on the screen.
        run : bool
            Run the Eelbrain GUI app (default is True for interactive plotting and
            False in scripts).
        """
        if h and axh:
            if h < axh:
                raise ValueError("h < axh")
        if w and axw:
            if w < axw:
                raise ValueError("w < axw")

        self.w_fixed = w or axw

        if nax is None:
            if w is None:
                if h is None:
                    h = axh_default
                w = ax_aspect * h
            elif h is None:
                h = w / ax_aspect
        elif nrow is None and ncol is None:
            if w and axw:
                ncol = math.floor(w / axw)
                nrow = math.ceil(nax / ncol)
                if h:
                    axh = axh or h / nrow
                elif axh:
                    h = axh * nrow
                else:
                    axh = axw / ax_aspect
                    h = axh * nrow
            elif h and axh:
                nrow = math.floor(h / axh)
                ncol = math.ceil(nax / nrow)
                if w:
                    axw = axw or w / ncol
                elif axw:
                    w = axw * ncol
                else:
                    axw = axh * ax_aspect
                    w = axw * ncol
            elif w:
                if axh:
                    ncol = round(w / (axh * ax_aspect))
                else:
                    ncol = round(w / (axh_default * ax_aspect))
                ncol = max(1, min(nax, ncol))
                axw = w / ncol
                nrow = math.ceil(nax / ncol)
                if h:
                    axh = h / nrow
                else:
                    if not axh:
                        axh = axw / ax_aspect
                    h = nrow * axh
            elif h:
                if axw:
                    nrow = round(h / (axw / ax_aspect))
                else:
                    nrow = round(h / axh_default)

                if nax < nrow:
                    nrow = nax
                elif nrow < 1:
                    nrow = 1

                axh = h / nrow
                ncol = math.ceil(nax / nrow)
                if w:
                    axw = w / ncol
                else:
                    if not axw:
                        axw = axh * ax_aspect
                    w = ncol * axw
            elif axh or axw:
                axh = axh or axw / ax_aspect
                axw = axw or axh * ax_aspect
                ncol = min(nax, math.floor(defaults['maxw'] / axw))
                nrow = math.ceil(nax / ncol)
                h = nrow * axh
                w = ncol * axw
            else:
                maxh = defaults['maxh']
                maxw = defaults['maxw']

                # try default
                axh = axh_default
                axw = axh_default * ax_aspect
                ncol = min(nax, math.floor(maxw / axw))
                nrow = math.ceil(nax / ncol)
                h = axh * nrow
                if h > maxh:
                    col_to_row_ratio = maxw / (ax_aspect * maxh)
                    # nax = ncol * nrow
                    # nax = (col_to_row * nrow) * nrow
                    nrow = math.ceil(math.sqrt(nax / col_to_row_ratio))
                    ncol = math.ceil(nax / nrow)
                    h = maxh
                    axh = h / nrow
                    w = maxw
                    axw = w / ncol
                else:
                    w = axw * ncol
        else:
            if nrow is None:
                ncol = min(nax, ncol)
                nrow = int(math.ceil(nax / ncol))
            elif ncol is None:
                nrow = min(nax, nrow)
                ncol = int(math.ceil(nax / nrow))

            if h:
                axh = axh or h / nrow
            if w:
                axw = axw or w / ncol

            if not axw and not axh:
                axh = axh_default

            if axh and not axw:
                axw = axh * ax_aspect
            elif axw and not axh:
                axh = axw / ax_aspect

        if nax is not None:
            nrow = int(nrow)
            ncol = int(ncol)
            if w is None:
                w = axw * ncol
            if h is None:
                h = axh * nrow

        fig_kwa = dict(figsize=(w, h), dpi=dpi)

        # make subplot parameters absolute
        if nax and not tight:
            size = 2
            bottom = mpl.rcParams['figure.subplot.bottom'] * size / h
            left = mpl.rcParams['figure.subplot.left'] * size / w
            right = 1 - (1 - mpl.rcParams['figure.subplot.right']) * size / w
            top = 1 - (1 - mpl.rcParams['figure.subplot.top']) * size / h
            hspace = mpl.rcParams['figure.subplot.hspace'] * size / h
            wspace = mpl.rcParams['figure.subplot.wspace'] * size / w
            fig_kwa['subplotpars'] = SubplotParams(left, bottom, right, top,
                                                   wspace, hspace)

        self.nax = nax
        self.h = h
        self.w = w
        self.axh = axh
        self.axw = axw
        self.nrow = nrow
        self.ncol = ncol
        self.fig_kwa = fig_kwa
        self.show = show
        self.run = run


class LegendMixin(object):
    __choices = ('invisible', 'separate window', 'draggable', 'upper right',
                 'upper left', 'lower left', 'lower right', 'right',
                 'center left', 'center right', 'lower center', 'upper center',
                 'center')
    __args = (False, 'fig', 'draggable', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

    def __init__(self, legend, legend_handles):
        """Legend toolbar menu mixin

        Parameters
        ----------
        legend : str | int | 'fig' | None
            Matplotlib figure legend location argument or 'fig' to plot the
            legend in a separate figure.
        legend_hamdles : dict
            {cell: handle} dictionary.
        """
        self.__handles = legend_handles
        self.legend = None
        if self.__handles:
            self.plot_legend(legend)

    def _fill_toolbar(self, tb):
        import wx

        choices = [name.title() for name in self.__choices]
        self.__ctrl = wx.Choice(tb, choices=choices, name='Legend')
        tb.AddControl(self.__ctrl, "Legend")
        self.__ctrl.Bind(wx.EVT_CHOICE, self.__OnChoice, source=self.__ctrl)

    def __OnChoice(self, event):
        self.__plot(self.__args[event.GetSelection()])

    def plot_legend(self, loc='fig', *args, **kwargs):
        """Plots (or removes) the legend from the figure.

        Parameters
        ----------
        loc : False | 'fig' | 'draggable' | str | int
            Where to plot the legend (see Notes; default 'fig').

        Returns
        -------
        legend_figure : None | legend
            If loc=='fig' the Figure, otherwise None.

        Notes
        -----
        legend content can be modified through the figure's
        ``legend_handles`` and ``legend_labels`` attributes.

        Possible values for the ``loc`` argument:

        ``False``:
            Make the current legend invisible
        'fig':
            Plot the legend in a new figure
        'draggable':
            The legend can be dragged to the desired position with the mouse
            pointer.
        str | int:
            Matplotlib position argument: plot the legend on the figure


        Matplotlib Position Arguments:

         - 'upper right'  : 1,
         - 'upper left'   : 2,
         - 'lower left'   : 3,
         - 'lower right'  : 4,
         - 'right'        : 5,
         - 'center left'  : 6,
         - 'center right' : 7,
         - 'lower center' : 8,
         - 'upper center' : 9,
         - 'center'       : 10,
        """
        out = self.__plot(loc, *args, **kwargs)
        if loc:
            if isinstance(loc, basestring):
                if loc == 'fig':
                    loc = 'separate window'
                loc = self.__choices.index(loc)
            self.__ctrl.SetSelection(loc)
        return out

    def save_legend(self, *args, **kwargs):
        """Save the legend as image file

        Parameters for Matplotlib's figure.savefig()
        """
        p = self.plot_legend(show=False)
        p.save(*args, **kwargs)
        p.close()

    def __plot(self, loc, *args, **kwargs):
        if loc and self.__handles:
            cells = sorted(self.__handles)
            labels = [cellname(cell) for cell in cells]
            handles = [self.__handles[cell] for cell in cells]
            if loc == 'fig':
                return Legend(handles, labels, *args, **kwargs)
            else:
                # take care of old legend; remove() not implemented as of mpl 1.3
                if self.legend is not None and loc == 'draggable':
                    self.legend.draggable(True)
                elif self.legend is not None:
                    self.legend.set_visible(False)
                    self.legend.draggable(False)
                elif loc == 'draggable':
                    self.legend = self.figure.legend(handles, labels, loc=1)
                    self.legend.draggable(True)

                if loc != 'draggable':
                    self.legend = self.figure.legend(handles, labels, loc=loc)
                self.draw()
        elif self.legend is not None:
            self.legend.set_visible(False)
            self.legend = None
            self.draw()
        elif not self.__handles:
            raise RuntimeError("No handles to produce legend.")


class Legend(_EelFigure):
    def __init__(self, handles, labels, *args, **kwargs):
        _EelFigure.__init__(self, "Legend", None, 2, 1, False, *args, **kwargs)

        self.legend = self.figure.legend(handles, labels, loc=2)

        # resize figure to match legend
        if not self._layout.w_fixed:
            self.draw()
            bb = self.legend.get_window_extent()
            w0, h0 = self._frame.GetSize()
            h = int(h0 + bb.x0 - bb.y0)
            w = int(bb.x0 + bb.x1)
            self._frame.SetSize((w, h))

        self._show()


class Figure(_EelFigure):
    def __init__(self, nax=None, title='Figure', *args, **kwargs):
        _EelFigure.__init__(self, title, nax, 2, 1, *args, **kwargs)

    def show(self):
        self._show()


class ImageTiler(object):
    """
    Create tiled images and animations from individual image files.

    Parameters
    ----------
    ext : str
        Extension to append to generated file names.
    nrow : int
        Number of rows of tiles in a frame.
    ncol : int
        Number of columns of tiles in a frame.
    nt : int
        Number of time points in the animation.
    dest : str(directory)
        Directory in which to place files. If None, a temporary directory
        is created and removed upon deletion of the ImageTiler instance.
    """
    def __init__(self, ext='.png', nrow=1, ncol=1, nt=1, dest=None):
        if dest is None:
            self.dir = tempfile.mkdtemp()
        else:
            if not os.path.exists(dest):
                os.makedirs(dest)
            self.dir = dest

        # find number of digits necessary to name images
        row_fmt = '%%0%id' % (np.floor(np.log10(nrow)) + 1)
        col_fmt = '%%0%id' % (np.floor(np.log10(ncol)) + 1)
        t_fmt = '%%0%id' % (np.floor(np.log10(nt)) + 1)
        self._tile_fmt = 'tile_%s_%s_%s%s' % (row_fmt, col_fmt, t_fmt, ext)
        self._frame_fmt = 'frame_%s%s' % (t_fmt, ext)

        self.dest = dest
        self.ncol = ncol
        self.nrow = nrow
        self.nt = nt

    def __del__(self):
        if self.dest is None:
            shutil.rmtree(self.dir)

    def get_tile_fname(self, col=0, row=0, t=0):
        if col >= self.ncol:
            raise ValueError("col: %i >= ncol" % col)
        if row >= self.nrow:
            raise ValueError("row: %i >= nrow" % row)
        if t >= self.nt:
            raise ValueError("t: %i >= nt" % t)

        if self.ncol == 1 and self.nrow == 1:
            return self.get_frame_fname(t)

        fname = self._tile_fmt % (col, row, t)
        return os.path.join(self.dir, fname)

    def get_frame_fname(self, t=0, dirname=None):
        if t >= self.nt:
            raise ValueError("t: %i >= nt" % t)

        if dirname is None:
            dirname = self.dir

        fname = self._frame_fmt % (t,)
        return os.path.join(dirname, fname)

    def make_frame(self, t=0, redo=False):
        """Produce a single frame."""
        dest = self.get_frame_fname(t)

        if os.path.exists(dest):
            if redo:
                os.remove(dest)
            else:
                return

        # collect tiles
        images = []
        colw = [0] * self.ncol
        rowh = [0] * self.nrow
        for r in xrange(self.nrow):
            row = []
            for c in xrange(self.ncol):
                fname = self.get_tile_fname(c, r, t)
                if os.path.exists(fname):
                    im = PIL.Image.open(fname)
                    colw[c] = max(colw[c], im.size[0])
                    rowh[r] = max(rowh[r], im.size[1])
                else:
                    im = None
                row.append(im)
            images.append(row)

        cpos = np.cumsum([0] + colw)
        rpos = np.cumsum([0] + rowh)
        out = PIL.Image.new('RGB', (cpos[-1], rpos[-1]))
        for r, row in enumerate(images):
            for c, im in enumerate(row):
                if im is None:
                    pass
                else:
                    out.paste(im, (cpos[c], rpos[r]))
        out.save(dest)

    def make_frames(self):
        for t in xrange(self.nt):
            self.make_frame(t=t)

    def make_movie(self, dest, framerate=10, codec='mpeg4'):
        """Make all frames and export a movie"""
        dest = os.path.expanduser(dest)
        dest = os.path.abspath(dest)
        root, ext = os.path.splitext(dest)
        dirname = os.path.dirname(dest)
        if ext not in ['.mov', '.avi']:
            if len(ext) == 4:
                dest = root + '.mov'
            else:
                dest = dest + '.mov'

        if not command_exists('ffmpeg'):
            err = ("Need ffmpeg for saving movies. Download from "
                   "http://ffmpeg.org/download.html")
            raise RuntimeError(err)
        elif os.path.exists(dest):
            os.remove(dest)
        elif not os.path.exists(dirname):
            os.mkdir(dirname)

        self.make_frames()

        # make the movie
        frame_name = self._frame_fmt
        cmd = ['ffmpeg',  # ?!? order of options matters
               '-f', 'image2',  # force format
               '-r', str(framerate),  # framerate
               '-i', frame_name,
               '-c', codec,
               '-sameq', dest,
               '-pass', '2'  #
               ]
        sp = subprocess.Popen(cmd, cwd=self.dir, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
        stdout, stderr = sp.communicate()
        if not os.path.exists(dest):
            raise RuntimeError("ffmpeg failed:\n" + stderr)

    def save_frame(self, dest, t=0, overwrite=False):
        if not overwrite and os.path.exists(dest):
            raise IOError("File already exists: %r" % dest)
        self.make_frame(t=t)
        fname = self.get_frame_fname(t)
        im = PIL.Image.open(fname)
        im.save(dest)
