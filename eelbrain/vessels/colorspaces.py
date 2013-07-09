"""
Colorspace objects provide settings for plotting functions.

Besides the :py:class:`Colorspace` class, this module also provides functions
to create some default colorspaces (the ``get_...`` functions).

In addition to matplotlib colormaps, the following names can be used:

"polar"
    white at 0, red for positive and blue for negative values.
"xpolar"
    like cm_polar, but extends the range by fading red and blue into black at
    extremes

"""

import numpy as np
import matplotlib as mpl
from mpl_toolkits.axisartist.axis_artist import Ticks


def colorbars_toFig_row_(cspaces, fig, row, nRows=None):
    """plots several colorbars into one row of a figure"""
    # interpret / prepare arguments
    if nRows == None:
        nRows = row
    if not np.iterable(cspaces):
        cspaces = [cspaces]
#    nCols = len(cspaces)
    # plot
    ysize = 1. / nRows
    ymin = ysize * (nRows - row + .5)
    height = ysize / 5
    xmin = np.r_[0.:len(cspaces)] / len(cspaces) + .1 / len(cspaces)
    width = .8 / len(cspaces)
    for i, c in enumerate(cspaces):
        # ax = fig.add_subplot(nRows, nCols, nCols*(row-1)+i+1)
        ax = fig.add_axes([xmin[i], ymin, width, height])
        c.toax(ax)
        # c.toAxes_(ax)


def _make_cmaps():
    "Create some local colormaps"
    _cdict = {'red':  [(.0, .0, .0),
                       (.5, 1., 1.),
                       (1., 1., 1.)],
              'green':[(.0, .0, .0),
                       (.5, 1., 1.),
                       (1., .0, .0)],
              'blue': [(.0, 1., 1.),
                       (.5, 1., 1.),
                       (1., .0, .0)]}
    cm_polar = mpl.colors.LinearSegmentedColormap("polar", _cdict)
    cm_polar.set_bad('w', alpha=0.)

    x = .3
    _cdict = {'red':  [(0, 0., 0.),
                       (0 + x, 0., 0.),
                       (.5, 1., 1.),
                       (1 - x, 1., 1.),
                       (1, 0., 0.)],
              'green':[(0, 0., 0.),
                       (0 + x, 0., 0.),
                       (.5, 1., 1.),
                       (1 - x, 0., 0.),
                       (1., 0., 0.)],
              'blue': [(0, 0., 0.),
                       (0 + x, 1., 1.),
                       (.5, 1., 1.),
                       (1 - x, 0., 0.),
                       (1, .0, .0)]}
    cm_xpolar = mpl.colors.LinearSegmentedColormap("extended polar", _cdict)
    cm_xpolar.set_bad('w', alpha=0.)

    cdict = {'red':   [(0.0, 0., 0.),
                       (0.5, 1., 1.),
                       (1.0, 0., 0.)],
             'green': [(0.0, 0., 0.),
                       (0.5, 0., 0.),
                       (1.0, 0., 0.)],
             'blue':  [(0.0, 1., 1.),
                       (0.5, 0., 0.),
                       (1.0, 1., 1.)]}
    cm_phase = mpl.colors.LinearSegmentedColormap("phaseCmap", cdict)
    cm_phase.set_bad('w', alpha=0.)

    cmaps = {'polar': cm_polar, 'xpolar': cm_xpolar, 'phase': cm_phase}
    return cmaps

cmaps = _make_cmaps()


class Colorspace(object):
    """
    - Stores information for mapping segment data to colors
    - can plot a colorbar for the legend with toax(ax) method
    """
    def __init__(self, cmap=None, vmax=None, vmin=None,
                 # contours: {v -> color}
                 contours={}, ps={},
                 # decoration
                 sensor_color='k', sensor_marker='x', cbar_data='vrange',
                 unit=None, ticks=None, ticklabels=None):
        """
        cmap: matplotlib colormap
            colormap for image plot, default is ``mpl.cm.jet``. See `mpl
            colormap documentation <http://matplotlib.sourceforge.net/api/cm_api.html>`_
        contours : {value: color} dict
            contours are drawn as lines on top of the im
        ps : {value: p-value} dict
            Mappings from parameter value to p-value. Used for labeling contours.
        unit: str
            the unit of measurement (only used for labels)
        vmax, vmin:
            max and min values that the colormap should be mapped to. If
            vmin is not specified, it defaults to -vmax.

        """
        if (cmap is not None) and not isinstance(cmap, str):
            raise TypeError("cmap must be None or str, got %r" % cmap)
        self._cmap_arg = cmap

        # save state attributes that differ from instance
        self._cbar_data = cbar_data

        # adjust instance attributes
        if (vmin is None) and (vmax is not None):
            vmin = -vmax

        if cbar_data == 'vrange':
            cbar_data = [(vmin, vmax)]

        # save instance attributes
        self.vmax = vmax
        self.vmin = vmin
        self.unit = unit
        self.ticks = ticks
        self.ticklabels = ticklabels
        self.sensor_color = sensor_color
        self.sensor_marker = sensor_marker
        self.cbar_data = cbar_data
        self.contours = contours
        self.ps = ps

        self.contour_kwargs = {'linestyles': 'solid'}

    def __setstate__(self, state):
        self.__init__(**state)

    def __getstate__(self):
        state = dict(cmap=self._cmap_arg, vmax=self.vmax, vmin=self.vmin,
                     unit=self.unit, ticks=self.ticks,
                     ticklabels=self.ticklabels,
                     sensor_color=self.sensor_color,
                     sensor_marker=self.sensor_marker,
                     cbar_data=self._cbar_data, contours=self.contours,
                     ps=self.ps)
        return state

    def __repr__(self):
        temp = "%s(%s)"
        name = self.__class__.__name__
        args = self._repr_args()
        return temp % (name, ', '.join(args))

    def _repr_args(self):
        args = []
        if self._cmap_arg:
            args.insert(0, repr(self._cmap_arg))
        if self.vmax is not None:
            args.append("vmax=%s" % self.vmax)
            if self.vmin not in [-self.vmax, None]:
                args.append("vmin=%s" % self.vmin)
        if self.contours:
            args.append("contours=%s" % self.contours)
        if self.ps:
            args.append("ps=%s" % self.ps)
        return args

    @property
    def cmap(self):
        if not hasattr(self, '_cmap'):
            self._cmap = self._make_cmap()
        return self._cmap

    def _make_cmap(self):
        if self._cmap_arg is None:
            return None
        else:
            return cmaps.get(self._cmap_arg, self._cmap_arg)

    def get_imkwargs(self, vmax=None):
        """
        Parameters
        ----------
        vmax : None | scalar
            Override the vmax and vmin values.
        """
        if vmax is None:
            kwargs = {'vmin': self.vmin,
                      'vmax': self.vmax,
                      'cmap': self.cmap}
        else:
            kwargs = {'vmin':-vmax,
                      'vmax': vmax,
                      'cmap': self.cmap}
        return kwargs

    def get_contour_kwargs(self):
        """
        Example::

            >>> map_kwargs = {'origin': "lower",
                              'extent': (emin, emax, emin, emax)}
            >>> map_kwargs.update(colorspace.get_contour_kwargs())
            >>> ax.contour(im, **map_kwargs)

        """
        levels = sorted(self.contours)
        d = {'levels': levels,
             'colors': [self.contours[l] for l in levels],
             }
        d.update(self.contour_kwargs)
        return d

    def toax(self, ax, data='auto', num=1001):  # NEW !!
        "kwarg data: data to use for plot (array)"
        if data == 'auto':
            num_part = num / len(self.cbar_data)
            data = np.hstack([np.linspace(x1, x2, num=num_part) for x1, x2 in self.cbar_data])
        if data.ndim == 1:
            data = data[None, :]
        # print data[:,::20]
        # print self.vmin, self.vmax
        ax.imshow(data, cmap=self.cmap, aspect='auto', extent=(0, num, 0, 1), vmin=self.vmin, vmax=self.vmax)
        # labelling
        ax.set_xlabel(self.unit)
        if self.ticks == None:
            ticks = ax.xaxis.get_ticklocs()
            ticks = np.unique(np.clip(ticks.astype(int), 0, num - 1)).astype(int)
            ticklabels = data[0, ticks]
        else:
            ticks = np.hstack([np.max(np.where(data[0] <= t)[0]) for t in self.ticks])
        if self.ticklabels == None:
            ticklabels = data[0, ticks]
        else:
            ticklabels = self.ticklabels
        ax.xaxis.set_ticks(ticks)
        ax.xaxis.set_ticklabels(ticklabels)
        ax.yaxis.set_visible(False)


class SigColorspace(Colorspace):
    def __init__(self, p=0.5, contours={.01: '.5', .001: '0'}):
        self._p = p

        pstr = str(p)[1:]
        cbar_data = [(0, 1.5 * p)]
        ticks = [0, p]
        ticklabels = ['0', pstr]
        Colorspace.__init__(self, None, 1, 0, contours, sensor_color='.5',
                            cbar_data=cbar_data, unit='p', ticks=ticks,
                            ticklabels=ticklabels)

    def __getstate__(self):
        state = dict(p=self._p, contours=self.contours)
        return state

    def _repr_args(self):
        return ['p=%s' % self._p, 'contours=%s' % self.contours]

    def _make_cmap(self):
        p = self._p
        cdict = {'red':[(0.0, 1., 1.),
                        (p, 1., 0.),
                        (1.0, 0., 0.)],
             'green':  [(0.0, 1., 1.),
                        (p, .0, 0.),
                        (1.0, 0., 0.)],
             'blue':   [(0.0, 0., 0.),
                        (p, 0., 0.),
                        (1.0, 0., 0.)]}
        cmap = mpl.colors.LinearSegmentedColormap("sigCmap", cdict, N=1000)
        cmap.set_bad('w', alpha=0.)
        return cmap


class SigWhiteColorspace(SigColorspace):
    def _make_cmap(self):
        p = self._p
        cdict = {'red':[(0.0, 1., 1.),
                        (p, 1., 1.),
                        (1.0, 1., 1.)],
             'green':  [(0.0, 1., 1.),
                        (p, .0, 1.),
                        (1.0, 1., 1.)],
             'blue':   [(0.0, 0., 0.),
                        (p, 0., 1.),
                        (1.0, 1., 1.)]}
        cmap = mpl.colors.LinearSegmentedColormap("sig White", cdict, N=1000)
        cmap.set_bad('w', alpha=0.)
        return cmap


class SymSigColorspace(SigColorspace):
    def __init__(self, p=.05, contours={.99: '.5', -.99: '.5', .999: '1',
                                        - .999: '1'}):
        self._p = p

        pstr = str(p)[1:]
        ticks = [-1, -1 + p, 1 - p, 1]
        ticklabels = ['0', pstr, pstr, '0']
        cbar_data = [(-1, -1 + 2 * p), (1 - 2 * p, 1)]
        Colorspace.__init__(self, None, 1, 0, contours, sensor_color='.5',
                            cbar_data=cbar_data, unit='p', ticks=ticks,
                            ticklabels=ticklabels)

    def _make_cmap(self):
        p = self._p
        cdict = {'red':   [(0., 1., 1.),
                           (p / 2, .25, 0.),
                           (1. - p / 2, 0., 1.),
                           (1., 1., 1.)],
                 'green': [(0.0, 0., 0.),
                           (1. - p / 2, 0., 0.),
                           (1., 1., 1.)],
                 'blue':  [(.0, 1., 1.),
                           (p / 2, 1., 0.),
                           (1.0, 0., 0.)]}
        cmap = mpl.colors.LinearSegmentedColormap("sigCmapSym", cdict, N=2000)
        cmap.set_bad('w', alpha=0.)
        return cmap


def get_default():
    return Colorspace('jet')

def get_EEG(vmax=1.5, unit=r'$\mu V$', **kwargs):
    return Colorspace('xpolar', vmax=vmax, unit=unit, **kwargs)

def get_MEG(vmax=2e-12, unit='Tesla', **kwargs):
    return Colorspace('xpolar', vmax=vmax, unit=unit, **kwargs)
