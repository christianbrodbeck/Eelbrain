'''Plot univariate data (:class:`Var` objects).

.. autosummary::
   :toctree: generated

   Barplot
   boxcox_explore
   Boxplot
   Correlation
   Histogram
   MultiTimeplot
   Regression
   Timeplot

'''
# author: Christian Brodbeck

from __future__ import division

from itertools import izip
import logging
from math import factorial

import numpy as np
import scipy.stats  # without this sp.stats is not available
import matplotlib.pyplot as plt
import matplotlib as mpl

from .._stats import test
from .._data_obj import (asfactor, isvar, asvar, ascategorial, assub, cellname,
                         Celltable)
from ._base import _EelFigure, str2tex


defaults = dict(title_kwargs={'size': 14,
                                'family': 'serif'},
              mono=False,  # use monochrome instead of colors
              # defaults for color
              hatch=['', '', '//', '', '*', 'O', '.', '', '/', '//'],
              linestyle=['-', '-', '--', ':'],
              c={'pw': ['#00FF00', '#FFCC00', '#FF6600', '#FF3300'],
#                   'colors': ['.3', '.7', '1.', '1.'],
                   'colors': [(0.99609375, 0.12890625, 0.0),
                              (0.99609375, 0.5859375, 0.0),
                              (0.98046875, 0.99609375, 0.0),
                              (0.19921875, 0.99609375, 0.0)],
                   'markers': ['o', 'o', '^', 'o'],
                   },
              # defaults for monochrome
              cm={'pw': ['.6', '.4', '.2', '.0'],
                    'colors': ['.3', '.7', '1.', '1.'],
                    'markers': ['o', 'o', '^', 'o'],
                    },
                )  # set by __init__








def _mark_plot_pairwise(ax, ct, par, y_min, y_unit, x0=0,
                        corr='Hochberg', levels=True, trend=".", pwcolors=None,
                        font_size=plt.rcParams['font.size'] * 1.5
                        ):
    "returns y_max"
    if levels is not True:  # to avoid test.star() conflict
        trend = False
    # tests
    if not pwcolors:
        if defaults['mono']:
            pwcolors = defaults['cm']['pw'][1 - bool(trend):]
        else:
            pwcolors = defaults['c']['pw'][1 - bool(trend):]
    k = len(ct.cells)
    tests = test._pairwise(ct.get_data(), within=ct.all_within, parametric=par, trend=trend,
                            levels=levels, corr=corr)
    reservation = np.zeros((factorial(k - 1), k - 1))
    y_top = y_min  # track top of plot
    y_start = y_min + 2 * y_unit
    # loop through possible connections
    for distance in xrange(1, k):
        for i in xrange(0, k - distance):
            j = i + distance  # i, j are data indexes for the categories being compared
            index = tests['pw_indexes'][(i, j)]
            stars = tests['stars'][index]
            if stars:
                c = pwcolors[stars - 1]
                symbol = tests['symbols'][index]

                free_levels = np.where(reservation[:, i:j].sum(1) == 0)[0]
                level = min(free_levels)
                reservation[level, i:j] = 1

                y1 = y_start + 2 * y_unit * level
                y2 = y1 + y_unit
                y_top = max(y2, y_top)
                x1 = (x0 + i) + .025
                x2 = (x0 + j) - .025
                ax.plot([x1, x1, x2, x2], [y1, y2, y2, y1], color=c)
                ax.text((x1 + x2) / 2, y2, symbol, color=c, size=font_size,
                        horizontalalignment='center', clip_on=False,
                        verticalalignment='center', backgroundcolor='w')
    y_top = y_top + 2 * y_unit
    return y_top

def _mark_plot_1sample(ax, ct, par, y_min, y_unit, x0=0, corr='Hochberg',
                        levels=True, trend=".", pwcolors=None,
                        popmean=0,  # <- mod
                        font_size=plt.rcParams['font.size'] * 1.5
                        ):
    "returns y_max"
    if levels is not True:  # to avoid test.star() conflict
        trend = False
    # tests
    if not pwcolors:
        if defaults['mono']:
            pwcolors = defaults['cm']['pw'][1 - bool(trend):]
        else:
            pwcolors = defaults['c']['pw'][1 - bool(trend):]
    # mod
    ps = []
    if par:
        for d in ct.get_data():
            t, p = scipy.stats.ttest_1samp(d, popmean)
            ps.append(p)
    else:
        raise NotImplementedError("nonparametric 1-sample test")
    stars = test.star(ps, int, levels, trend, corr)
    stars_str = test.star(ps, levels=levels, trend=trend)
    if any(stars):
        y_stars = y_min + 1.75 * y_unit
        for i, n_stars in enumerate(stars):
            if n_stars > 0:
                c = pwcolors[n_stars - 1]
                ax.text(x0 + i, y_stars, stars_str[i], color=c, size=font_size,
                        horizontalalignment='center', clip_on=False,
                        verticalalignment='center')
        return y_min + 4 * y_unit
    else:
        return y_min


class _SimpleFigure(_EelFigure):
    def __init__(self, wintitle, title=None, xlabel=None, ylabel=None,
                 titlekwargs=defaults['title_kwargs'], yticks=None,
                 figsize=(5, 5), xtick_rotation=0, ytick_rotation=0):
        _EelFigure.__init__(self, wintitle, fig_kwa={'figsize': figsize})

        # axes
        ax_x0 = .025 + .07 * bool(ylabel)
        ax_y0 = .065 + .055 * bool(xlabel)
        ax_dx = .975 - ax_x0
        ax_dy = .95 - ax_y0 - .08 * bool(title)
        rect = [ax_x0, ax_y0, ax_dx, ax_dy]
        ax = self.figure.add_axes(rect)
        self._axes.append(ax)
        self._ax = ax

        # ticks / tick labels
        self._yticks = yticks
        self._x_tick_rotation = xtick_rotation
        self._y_tick_rotation = ytick_rotation
        xax = ax.get_xaxis()
        xax.set_ticks_position('none')

        # collector for handles for figure legend
        self._handles = []
        self._legend = None

        # title and labels
        if title:
            if 'verticalalignment' not in titlekwargs:
                titlekwargs['verticalalignment'] = 'bottom'
            ax.set_title(title, **titlekwargs)
        if ylabel:
            ax.set_ylabel(ylabel)  # , labelpad=-20.)
        if xlabel:
            ax.set_xlabel(xlabel)

    def add_legend_handles(self, *handles):
        for handle in handles:
            label = handle.get_label()
            if not self.legend_has_label(label):
                self._handles.append(handle)

    def legend_has_label(self, label):
            return any(label == h.get_label() for h in self._handles)

    def legend(self, loc=0, fig=False, zorder=-1, ncol=1):
        "add a legend to the plot"
        if fig:
            labels = (h.get_label() for h in self._handles)
            l = self.figure.legend(self._handles, labels, loc, ncol=ncol)
            self._legend = l
        else:
            l = self._ax.legend(loc=loc, ncol=ncol)
            if l:
                l.set_zorder(-1)
            else:
                raise ValueError("No labeled plot elements for legend")

    def _show(self):
        "resizes the axes to take into account tick spacing"

        if self._yticks:
            yticks = self._yticks
            if np.iterable(yticks[0]):
                locations, labels = yticks
            else:
                locations = yticks
                labels = None
            self._ax.set_yticks(locations)
            if labels:
                self._ax.set_yticklabels(labels)
        if self._x_tick_rotation:
            for t in self._ax.get_xticklabels():
                t.set_rotation(self._x_tick_rotation)
        if self._y_tick_rotation:
            for t in self._ax.get_yticklabels():
                t.set_rotation(self._y_tick_rotation)

        # adjust the position of the axes to show all labels
        self.draw()
        x_in, y_in = self.figure.get_size_inches()
        dpi = self.figure.get_dpi()
        border_x0 = 0.05  # in inches
        border_x1 = 0.05  # in inches
        border_y0 = 0.05  # in inches
        border_y1 = 0.05  # in inches
        if self._legend:
            w = self._legend.get_window_extent()
            border_y0 += w.ymax / dpi

        xmin = x_in * dpi
        ymin = y_in * dpi
        xmax = 0
        ymax = 0
        for c in self._ax.get_children():
            try:
                w = c.get_window_extent()
            except:
                pass
            else:
                xmin = min(xmin, w.xmin)
                ymin = min(ymin, w.ymin)
                xmax = max(xmax, w.xmax)
                ymax = max(ymax, w.ymax)

        for label in self._ax.get_ymajorticklabels() + \
                     [self._ax.get_yaxis().get_label()]:
            w = label.get_window_extent()
            xmin = min(xmin, w.xmin)

        for label in self._ax.get_xmajorticklabels() + \
                     [self._ax.get_xaxis().get_label()]:
            w = label.get_window_extent()
            ymin = min(ymin, w.ymin)

        # to figure proportion
        xmin = (xmin / dpi - border_x0) / x_in
        xmax = (xmax / dpi + border_x1 - x_in) / x_in
        ymin = (ymin / dpi - border_y0) / y_in
        ymax = (ymax / dpi + border_y1 - y_in) / y_in

        p = self._ax.get_position()
        p.x0 -= xmin
        p.x1 -= xmax
        p.y0 -= ymin
        p.y1 -= ymax
        self._ax.set_position(p)

        _EelFigure._show(self, False)



class Boxplot(_SimpleFigure):
    def __init__(self, Y, X=None, match=None, sub=None, datalabels=None,
                 bottom=None, top=None,
                 title=True, ylabel='{unit}', xlabel=True, xtick_delim='\n',
                 titlekwargs=defaults['title_kwargs'],
                 test=True, par=True, trend=".", corr='Hochberg',
                 pwcolors=None, hatch=False, colors=False,
                 ds=None, **kwargs):
        """
        Make a boxplot.

        Parameters
        ----------
        Y : Var
            Dependent variable.
        X : categorial
            Category definition (draw one box for every cell in X).
        datalabels : scalar
            Threshold for labeling outliers (in standard-deviation).
        test : bool | scalar
            True (default): perform pairwise tests;  False/None: no tests;
            scalar: 1-sample tests against this value.
        corr : None | 'hochberg' | 'bonferroni' | 'holm'
            Method for multiple comparison correction (default 'hochberg').

        Returns
        -------
        fig : matplotlib figure
            Reference to the matplotlib figure.
        """
        # get data
        ct = Celltable(Y, X, match=match, sub=sub, ds=ds, coercion=asvar)
        Y = ct.Y
        X = ct.X

        # kwargs
        if hatch == True:
            hatch = defaults['hatch']
        if colors == True:
            if defaults['mono']:
                colors = defaults['cm']['colors']
            else:
                colors = defaults['c']['colors']

        if title is True:
            title = str2tex(getattr(Y, 'name', None))

        # ylabel
        if hasattr(Y, 'info'):
            unit = Y.info.get('unit', '')
        else:
            unit = ''
        ylabel = ylabel.format(unit=unit)

        # xlabel
        if xlabel is True:
            xlabel = str2tex(X.name or False)

        # get axes


        _SimpleFigure.__init__(self, "Boxplot", title, xlabel, ylabel,
                               titlekwargs, **kwargs)
        ax = self._axes[0]

        # determine ax lim
        if bottom is None:
            if np.min(ct.Y.x) >= 0:
                bottom = 0
            else:
                d_min = np.min(ct.Y.x)
                d_max = np.max(ct.Y.x)
                d_range = d_max - d_min
                bottom = d_min - .05 * d_range

        # boxplot
        k = len(ct.cells)
        all_data = ct.get_data()
        bp = ax.boxplot(all_data)

        # Now fill the boxes with desired colors
        if hatch or colors:
            numBoxes = len(bp['boxes'])
            for i in range(numBoxes):
                box = bp['boxes'][i]
                boxX = box.get_xdata()[:5]  # []
                boxY = box.get_ydata()[:5]  # []
                boxCoords = zip(boxX, boxY)
                # Alternate between Dark Khaki and Royal Blue
                if len(colors) >= numBoxes:
                    c = colors[i]
                else:
                    c = '.5'
                if len(hatch) >= numBoxes:
                    h = hatch[i]
                else:
                    h = ''
                boxPolygon = mpl.patches.Polygon(boxCoords, facecolor=c, hatch=h, zorder=-999)
                ax.add_patch(boxPolygon)
        if defaults['mono']:
            for itemname in bp:
                plt.setp(bp[itemname], color='black')

        # labelling
        ax.set_xticks(np.arange(len(ct.cells)) + 1)
        ax.set_xticklabels(ct.cellnames(xtick_delim))
        y_min = np.max(np.hstack(all_data))
        y_unit = (y_min - bottom) / 15

        # tests
        if test is True:
            y_top = _mark_plot_pairwise(ax, ct, par, y_min, y_unit, corr=corr,
                                        x0=1, trend=trend)
        else:
            ax.axhline(test, color='black')
            y_top = _mark_plot_1sample(ax, ct, par, y_min, y_unit,
                                       x0=1, corr=corr, popmean=test, trend=trend)
        if top is None:
            top = y_top

        # data labels
        if datalabels:
            for i, cell in enumerate(ct.cells):
                d = ct.data[cell]
                indexes = np.where(np.abs(d) / d.std() >= datalabels)[0]
                for index in indexes:
                    label = ct.matchlabels[cell][index]
                    ax.annotate(label, (i + 1, d[index]))

        # set ax limits
        ax.set_ylim(bottom, top)
        ax.set_xlim(.5, k + .5)

        self._show()


class Barplot(_SimpleFigure):
    def __init__(self, Y, X=None, match=None, sub=None, test=True, par=True,
                 corr='Hochberg', title=True, trend=".",
                 # bar settings:
                 ylabel='{unit}{err}', err='2sem', ec='k', xlabel=True,
                 xtick_delim='\n', hatch=False, colors=False, bottom=0,
                 c='#0099FF', edgec=None, ds=None, **kwargs):
        """Barplot

        Parameters
        ----------
        Y : scalar Var
            Dependent variable.
        X : categorial
            Model (Factor or Interaction)
        bottom : scalar
            Lowest possible value on the y axis (default is 0).
        test : bool | scalar
            True (default): perform pairwise tests;  False: no tests;
            scalar: 1-sample tests against this value
        pwcolors : list of colors
            list of mpl colors corresponding to the cells in X (e.g. ['#FFCC00',
            '#FF6600','#FF3300'] with 3 levels
        err: func | "[x][type]"
            The magnitude of the error bars to display. Examples:
            'std' : 1 standard deviation;
            '.95ci' : 95% confidence interval (see :func:`..ci`);
            '2sem' : 2 standard error of the mean
        ylabel : str
            The following
            {err} = error bar description
        """
        if title is True:
            title = getattr(Y, 'name', None)

        if isinstance(err, basestring):
            if err.endswith('ci'):
                if len(err) > 2:
                    a = float(err[:-2])
                else:
                    a = .95
                err_desc = '$\pm %g ci$' % a
            elif err.endswith('sem'):
                if len(err) > 3:
                    a = float(err[:-3])
                else:
                    a = 1
                err_desc = '$\pm %g sem$' % a
            elif err.endswith('std'):
                if len(err) > 3:
                    a = float(err[:-3])
                else:
                    a = 1
                err_desc = '$\pm %g std$' % a
            else:
                raise ValueError('unrecognized statistic: %r' % err)
        else:
            err_desc = getattr(err, '__name__', '')

        # ylabel
        if ylabel:
            if hasattr(Y, 'info'):
                unit = Y.info.get('unit', '')
            else:
                unit = ''

            if ylabel is True:
                if unit:
                    ylabel = unit
                else:
                    ylabel = False
            elif isinstance(ylabel, str):
                ylabel = ylabel.format(unit=unit, err=err_desc)

        # xlabel
        if xlabel is True:
            if hasattr(X, 'name'):
                xlabel = X.name.replace('_', ' ')
            else:
                xlabel = False

        _SimpleFigure.__init__(self, "BarPlot", title, xlabel, ylabel, **kwargs)

        ct = Celltable(Y, X, match=match, sub=sub, ds=ds, coercion=asvar)

        x0, x1, y0, y1 = _barplot(self._ax, ct,
                                  test=test, par=par, trend=trend, corr=corr,
                                  # bar settings:
                                  err=err, ec=ec,
                                  hatch=hatch, colors=colors,
                                  bottom=bottom, c=c, edgec=edgec,
                                  return_lim=True)

        self._ax.set_xlim(x0, x1)
        self._ax.set_ylim(y0, y1)

        # figure decoration
        self._ax.set_xticks(np.arange(len(ct.cells)))
        self._ax.set_xticklabels(ct.cellnames(xtick_delim))

        self._show()


def _barplot(ax, ct,
             test=True, par=True, trend=".", corr='Hochberg',
             # bar settings:
             err='2sem', ec='k',
             hatch=False, colors=False,
             bottom=0, c='#0099FF', edgec=None,
             left=None, width=.5,
             return_lim=False,
             ):
    """
    draw a barplot to axes ax for Celltable ct.

    return_lim: return axes limits (x0, x1, y0, y1)
    """
    # kwargs
    if hatch == True:
        hatch = defaults['hatch']
    if colors == True:
        if defaults['mono']:
            colors = defaults['cm']['colors']
        else:
            colors = defaults['c']['colors']
    # data
    k = len(ct.cells)
    if left is None:
        left = np.arange(k) - width / 2
    height = np.array(ct.get_statistic(np.mean))
    y_error = np.array(ct.get_statistic(err))


    # fig spacing
    plot_max = np.max(height + y_error)
    plot_min = np.min(height - y_error)
    plot_span = plot_max - plot_min
    y_bottom = min(bottom, plot_min - plot_span * .05)

    # main BARPLOT
    bars = ax.bar(left, height - bottom, width, bottom=bottom,
                  color=c, edgecolor=edgec, linewidth=1,
                  ecolor=ec, yerr=y_error)

    # hatch
    if hatch:
        for bar, h in zip(bars, hatch):
            bar.set_hatch(h)
    if colors:
        for bar, c in zip(bars, colors):
            bar.set_facecolor(c)

    # pairwise tests
    # prepare pairwise plotting
    if y_error == None:
        y_min = np.max(height)
    else:
        y_min = np.max(height + y_error)
    y_unit = (y_min - y_bottom) / 15
    if test is True:
        y_top = _mark_plot_pairwise(ax, ct, par, y_min, y_unit,
                                    corr=corr, trend=trend)
    elif (test is False) or (test is None):
        y_top = y_min + y_unit
    else:
        ax.axhline(test, color='black')
        y_top = _mark_plot_1sample(ax, ct, par, y_min, y_unit,
                                   popmean=test, corr=corr, trend=trend)

    #      x0,                 x1,                  y0,       y1
    if return_lim:
        lim = (min(left) - .5 * width, max(left) + 1.5 * width, y_bottom, y_top)
        return lim


class Timeplot(_SimpleFigure):
    def __init__(self, Y, categories, time, match=None, sub=None, ds=None,
                 # data plotting
                 main=np.mean,
                 spread='box', x_jitter=False,
                 datalabels=None,
                 # labelling
                 ylabel=True, xlabel=True,
                 legend=True, loc=0,
                 colors=True, hatch=False, markers=True,
                 **kwargs
                 ):
        """Plot a variable over time.

        Parameters
        ----------
        spread : str
            How to indicator data spread.
            None - without
            'box' - boxplots
            for lineplots:
            'Xsem' X standard error of the means
            'Xstd' X standard deviations
        main : numpy function
            draw lines to connect values across time (default: np.mean)
            can be 'bar' for barplots or False
        datalabels : scalar
            Label outlier data. The argument is provided in std.
        diff : scalar
            Use this value as baseline for plotting; test other conditions
            agaist baseline (instead of pairwise)
        legend : bool | 'fig'
            Plot a legend; with `fig`, plot as figlegend.
        loc :
            The legend location.
        """
        sub = assub(sub, ds)
        Y = asvar(Y, sub, ds)
        categories = ascategorial(categories, sub, ds)
        time = asvar(time, sub, ds)
        if match is not None:
            match = ascategorial(match, sub, ds)

        # transform to 3 kwargs:
        # - local_plot ('bar' or 'box')
        # - line_plot (function for values to connect)
        # - spread
        if main == 'bar':
            assert spread != 'box'
            local_plot = 'bar'
            line_plot = None
        else:
            line_plot = main
            if spread == 'box':
                local_plot = 'box'
                spread = None
            else:
                local_plot = None
        del main

        # hatch/marker
        if hatch == True:
            hatch = defaults['hatch']
        if markers == True:
            if defaults['mono']:
                markers = defaults['cm']['markers']
            else:
                markers = defaults['c']['markers']

        # colors: {category index -> color, ...}
        colors_arg = colors
        if defaults['mono']:
            colors = dict(zip(categories.cells, defaults['cm']['colors']))
        else:
            colors = dict(zip(categories.cells, defaults['c']['colors']))
        if isinstance(colors_arg, dict):
            colors.update(colors_arg)
        elif isinstance(colors_arg, (tuple, list)):
            colors.update(dict(zip(categories.cells, colors_arg)))

        for c in categories.cells:
            if c not in colors:
                colors[c] = '.5'

        color_list = [colors[i] for i in sorted(colors.keys())]

        # ylabel
        if ylabel is True:
            ylabel = str2tex(getattr(Y, 'name', None))

        # xlabel
        if xlabel is True:
            xlabel = str2tex(time.name)

        # get axes
        _SimpleFigure.__init__(self, "Timeplot", xlabel=xlabel, ylabel=ylabel,
                               **kwargs)
        ax = self._ax

        # categories
        n_cat = len(categories.cells)

        # find time points
        if isvar(time):
            time_points = np.unique(time.x)
            n_time_points = len(time_points)
            time_step = min(np.diff(time_points))
            if local_plot in ['box', 'bar']:
                within_spacing = time_step / (2 * n_cat)
                padding = (2 + n_cat / 2) * within_spacing
            else:
                within_spacing = time_step / (8 * n_cat)
                padding = time_step / 4  # (2 + n_cat/2) * within_spacing

            rel_pos = np.arange(0, n_cat * within_spacing, within_spacing)
            rel_pos -= np.mean(rel_pos)

            t_min = min(time_points) - padding
            t_max = max(time_points) + padding
        else:
            raise NotImplementedError("time needs to be Var object")

        # prepare array for timelines
        if line_plot:
            line_values = np.empty((n_cat, n_time_points))
        if spread  and  local_plot != 'bar':
            yerr = np.empty((n_cat, n_time_points))

        # loop through time points
        for i_t, t in enumerate(time_points):
            ct = Celltable(Y, categories, match=match, sub=(time == t))
            if line_plot:
                line_values[:, i_t] = ct.get_statistic(line_plot)

            pos = rel_pos + t
            if local_plot == 'box':
                # boxplot
                bp = ax.boxplot(ct.get_data(out=list), positions=pos, widths=within_spacing)

                # Now fill the boxes with desired colors
#               if hatch or colors:
#                   numBoxes = len(bp['boxes'])
                for i, cell in enumerate(ct.cells):
                    box = bp['boxes'][i]
                    boxX = box.get_xdata()[:5]
                    boxY = box.get_ydata()[:5]
                    boxCoords = zip(boxX, boxY)

                    c = colors[cell]
                    try:
                        h = hatch[i]
                    except:
                        h = ''
                    boxPolygon = mpl.patches.Polygon(boxCoords, facecolor=c, hatch=h, zorder=-999)
                    ax.add_patch(boxPolygon)

                if True:  # defaults['mono']:
                    for itemname in bp:
                        plt.setp(bp[itemname], color='black')
            elif local_plot == 'bar':
                lim = _barplot(ax, ct, test=False, err=spread,  # ec=ec,
                               # bar settings:
                               hatch=hatch, colors=color_list,
                               bottom=0, left=pos, width=within_spacing)
            elif spread:
                yerr[:, i_t] = ct.get_statistic(spread)

        if line_plot:
            # plot means
            x = time_points
            for i, cell in enumerate(categories.cells):
                y = line_values[i]
                name = cellname(cell)
                color = colors[cell]

                if hatch:
                    ls = defaults['linestyle'][i]
                    if color == '1.':
                        color = '0.'
                else:
                    ls = '-'

                if ls == '-':
                    mfc = color
                else:
                    mfc = '1.'

                try:
                    marker = markers[i]
                except:
                    marker = None

                handles = ax.plot(x, y, color=color, linestyle=ls, label=name,
                                  zorder=6, marker=marker, mfc=mfc)
                self.add_legend_handles(*handles)

                if spread:
                    if x_jitter:
                        x_errbars = x + rel_pos[i]
                    else:
                        x_errbars = x
                    ax.errorbar(x_errbars, y, yerr=yerr[i], fmt=None, zorder=5,
                                ecolor=color, linestyle=ls, label=name)
        else:
            legend = False

        # finalize
        ax.set_xlim(t_min, t_max)
        ax.set_xticks(time_points)

        if any (legend is i for i in (False, None)):
            pass
        elif legend == 'fig':
            self.figure.legend(fig=True, loc='lower center')
        else:
            if legend is True:
                loc = 0
            else:
                loc = legend
            self.legend(loc=loc)

        self._show()


class MultiTimeplot(_SimpleFigure):
    def __init__(self, figsize=(7, 2.5),
                 tpad=.5, ylim=None,
                 main=np.mean,
                 spread='box', x_jitter=False,
                 datalabels=None,
                 # labelling
                 title=None, ylabel=True, xlabel=True,
                 titlekwargs=defaults['title_kwargs'],
                 ):
        """
        Template for a figure including multiple timeplots.

        Create an empty template figure for a plot subsuming several
        :func:`timeplot`s. Add timeplots using the ``plot`` method.


        Parameters
        ----------

        ylim : None | (bottom, top) tuple of scalars
            By default (None), ylim depend on the data plotted.

        """
        self._ylim = ylim
        # ylabel
        if ylabel is True:
            self._ylabel = True
            ylabel = False
        else:
            self._ylabel = False
        # xlabel
        if xlabel is True:
            self._xlabel = True
            xlabel = False
        else:
            self._xlabel = False

        # transform to 3 kwargs:
        # - local_plot ('bar' or 'box')
        # - line_plot (function for values to connect)
        # - spread
        if main == 'bar':
            assert spread != 'box'
            local_plot = 'bar'
            line_plot = None
        else:
            line_plot = main
            if spread == 'box':
                local_plot = 'box'
                spread = None
            else:
                local_plot = None
        del main
        self._local_plot = local_plot
        self._line_plot = line_plot
        self._spread = spread
        self._x_jitter = x_jitter


        self._tstart = None
        self._tend = None
        self._tpad = tpad

        self._xticks = []
        self._xticklabels = []

        self._headings = []  # collect text objects for headings to adjust y
        # [(forced_y, line, text), ...]
        self._heading_y = None  # used heading y
        self._y_unit = 0

        # get axes
        _SimpleFigure.__init__(self, title, xlabel, ylabel, titlekwargs,
                               figsize=figsize)

    def plot(self, Y, categories, time, match=None, sub=None,
             tskip=1,
             heading=None, headingy=None,
             colors=True, hatch=False, markers=True):
        """
        Parameters
        ----------

        Y :
            variable to plot
        categories :
            Factor indicating different categories to plot
        time :
            variable indicating time
        match :
            Factor which indicates dependent measures (e.g. subject)
        sub :
            subset
        heading :
            heading for this set (if None, no heading is added)
        headingy :
            y coordinate on which to plot the heading


        See also timeplot parameters.

        """
        categories = asfactor(categories)

        ax = self._ax
        local_plot = self._local_plot
        line_plot = self._line_plot
        spread = self._spread

        # labels
        if self._ylabel is True:
            ylabel = str2tex(getattr(Y, 'name', None))
            if ylabel:
                ax.set_ylabel(ylabel)
                self._ylabel = False
        if self._xlabel is True:
            ax.set_xlabel(str2tex(time.name))
            self._xlabel = False

        ### same as timeplot() #####  #####  #####  #####  #####  #####
        # hatch/marker
        if hatch == True:
            hatch = defaults['hatch']
        if markers == True:
            if defaults['mono']:
                markers = defaults['cm']['markers']
            else:
                markers = defaults['c']['markers']

        # colors: {category index -> color, ...}
        colors_arg = colors
        if defaults['mono']:
            colors = dict(zip(categories.indexes, defaults['cm']['colors']))
        else:
            colors = dict(zip(categories.indexes, defaults['c']['colors']))
        colors.update(categories.colors)
        if isinstance(colors_arg, dict):
            colors.update(colors_arg)
        elif isinstance(colors_arg, (tuple, list)):
            colors.update(dict(zip(categories.indexes, colors_arg)))

        for c in categories.cells:
            if c not in colors:
                colors[c] = '.5'

        color_list = [colors[i] for i in sorted(colors.keys())]


        # sub
        if sub is not None:
            Y = Y[sub]
            categories = categories[sub]
            time = time[sub]
            match = match[sub]

        # categories
        n_cat = len(categories.cells)

        # find time points
        if isvar(time):
            time_points = np.unique(time.x)
        ### NOT same as timeplot() #####  #####  #####  #####  #####  #####
            if self._tend is None:
                t_add = 0
                self._tstart = time_points[0]
                self._tend = time_points[-1]
            else:
                t_add = self._tend + tskip - time_points[0]
                self._tend = t_add + time_points[-1]
        ### same as timeplot()     #####  #####  #####  #####  #####  #####
            n_time_points = len(time_points)
            time_step = min(np.diff(time_points))
            if local_plot in ['box', 'bar']:
                within_spacing = time_step / (2 * n_cat)
            else:
                within_spacing = time_step / (8 * n_cat)

            rel_pos = np.arange(0, n_cat * within_spacing, within_spacing)
            rel_pos -= np.mean(rel_pos)

        ### NOT same as timeplot() #####  #####  #####  #####  #####  #####
            t_min = self._tstart - self._tpad
            t_max = self._tend + self._tpad
        ### same as timeplot()     #####  #####  #####  #####  #####  #####

        else:
            raise NotImplementedError("time needs to be Var object")

        # prepare array for timelines
        if line_plot:
            line_values = np.empty((n_cat, n_time_points))
        if spread  and  local_plot != 'bar':
            yerr = np.empty((n_cat, n_time_points))

        # loop through time points
        ymax = None; ymin = None
        for i_t, t in enumerate(time_points):
            ct = Celltable(Y, categories, match=match, sub=(time == t))
            if line_plot:
                line_values[:, i_t] = ct.get_statistic(line_plot)

            pos = rel_pos + t
            if local_plot == 'box':
                # boxplot
                data = ct.get_data(out=list)
                bp = ax.boxplot(data, positions=pos, widths=within_spacing)

                ymax_loc = np.max(data)
                ymin_loc = np.min(data)

                # Now fill the boxes with desired colors
                for i, cat in enumerate(ct.indexes):
                    box = bp['boxes'][i]
                    boxX = box.get_xdata()[:5]
                    boxY = box.get_ydata()[:5]
                    boxCoords = zip(boxX, boxY)

                    c = colors[cat]
                    try:
                        h = hatch[i]
                    except:
                        h = ''
                    boxPolygon = mpl.patches.Polygon(boxCoords, facecolor=c, hatch=h, zorder=-999)
                    ax.add_patch(boxPolygon)

                if True:  # defaults['mono']:
                    for itemname in bp:
                        plt.setp(bp[itemname], color='black')
            elif local_plot == 'bar':
                lim = _barplot(ax, ct, test=False, err=spread,  # ec=ec,
                               # bar settings:
                               hatch=hatch, colors=color_list,
                               bottom=0, left=pos, width=within_spacing)
                ymax_loc = lim[1]
                ymin_loc = lim[0]
            elif spread:
                yerr_loc = np.array(ct.get_statistic(spread))
                yerr[:, i_t] = yerr_loc
                y_loc = np.array(ct.get_statistic(np.mean))
                ymax_loc = max(y_loc + yerr_loc)
                ymin_loc = min(y_loc - yerr_loc)

            if ymax is None:
                ymax = ymax_loc
                ymin = ymin_loc
            else:
                ymax = max(ymax, ymax_loc)
                ymin = min(ymin, ymin_loc)

        if line_plot:
            # plot means
            x = time_points + t_add
            for i, cell in enumerate(categories.cells):
                y = line_values[i]

                color = colors[cell]

                if hatch:
                    ls = defaults['linestyle'][i]
                    if color == '1.':
                        color = '0.'
                else:
                    ls = '-'

                if ls == '-':
                    mfc = color
                else:
                    mfc = '1.'

                try:
                    marker = markers[i]
                except:
                    marker = None

        ### NOT same as timeplot() #####  #####  #####  #####  #####  #####
                if self.legend_has_label(cell):
                    label = None
                else:
                    label = cell
                handles = ax.plot(x, y, color=color, linestyle=ls, label=label,
                                  zorder=6, marker=marker, mfc=mfc)
        ### same as timeplot()     #####  #####  #####  #####  #####  #####
                self.add_legend_handles(*handles)

                if spread:
                    if self._x_jitter:
                        x_errbars = x + rel_pos[i]
                    else:
                        x_errbars = x
                    ax.errorbar(x_errbars, y, yerr=yerr[i], fmt=None, zorder=5,
                                ecolor=color, linestyle=ls, label=cell)

        ### NOT same as timeplot() #####  #####  #####  #####  #####  #####
        # heading
        y_unit = (ymax - ymin) / 10
        if y_unit > self._y_unit:
            for forced_y, ln, hd in self._headings:
                if forced_y is not None:
                    hd.set_y(forced_y + self._y_unit / 2)
            self._y_unit = y_unit

        if heading:
            x0 = time_points[ 0] + t_add - tskip / 4
            x1 = time_points[-1] + t_add + tskip / 4
            x = (x0 + x1) / 2
            if headingy is None:
                y = ymax + self._y_unit
                y_text = y + self._y_unit / 2
                if self._heading_y is None:
                    self._heading_y = y
                elif y > self._heading_y:
                    self._heading_y = y
                    for forced_y, ln, hd in self._headings:
                        if forced_y is None:
                            hd.set_y(y_text)
                            ln.set_ydata([y, y])
                        # adjust y
                elif y < self._heading_y:
                    y = self._heading_y
                    y_text = self._heading_y + self._y_unit / 2
            else:
                y = headingy
                y_text = y + self._y_unit / 2

            hd = ax.text(x, y_text, heading, va='bottom', ha='center')
            ln = ax.plot([x0, x1], [y, y], c='k')[0]

            self._headings.append((headingy, ln, hd))
            if not self._ylim:
                ax.set_ylim(top=(y + 3 * y_unit))

        # finalize
        ax.set_xlim(t_min, t_max)
        for t in time_points:
            self._xticks.append(t + t_add)
            self._xticklabels.append('%g' % t)

        if self._ylim:
            ax.set_ylim(self._ylim)

        ax.set_xticks(self._xticks)
        ax.set_xticklabels(self._xticklabels)

        self._show()

    def add_legend(self, fig=False, loc=0, zorder=-1, **kwargs):
        if fig:
            self.figure.figlegend(loc, **kwargs)
        else:
            l = self._ax.legend(loc=loc, **kwargs)
            l.set_zorder(zorder)

        self._show()



def _reg_line(Y, reg):
    coeff = np.hstack([np.ones((len(reg), 1)), reg[:, None]])
    (a, b), residues, rank, s = np.linalg.lstsq(coeff, Y)
    regline_x = np.array([min(reg), max(reg)])
    regline_y = a + b * regline_x
    return regline_x, regline_y


class Correlation(_EelFigure):

    def __init__(self, Y, X, cat=None, sub=None, ds=None, title=None,
                 c=['b', 'r', 'k', 'c', 'm', 'y', 'g'], delim=' ',
                 lloc='lower center', lncol=2, figlegend=True, xlabel=True,
                 ylabel=True, rinxlabel=True, **layout):
        """
        Plot the correlation between two variables.

        Parameters
        ----------
        cat :
            categories
        rinxlabel :
            print the correlation in the xlabel
        """
        _EelFigure.__init__(self, "Correlation", 1, layout, 1, 5,
                            figtitle=title)

        sub = assub(sub, ds)
        Y = asvar(Y, sub, ds)
        X = asvar(X, sub, ds)
        if cat is not None:
            cat = ascategorial(cat, sub, ds)

        # determine labels
        if xlabel is True:
            xlabel = str2tex(X.name)
        if ylabel is True:
            ylabel = str2tex(Y.name)
        if rinxlabel:
            temp = "\n(r={r:.3f}{s}, p={p:.4f}, n={n})"
            if cat is None:
                r, p, n = test._corr(Y, X)
                s = test.star(p)
                xlabel += temp.format(r=r, s=s, p=p, n=n)
            else:
                pass

        # decorate axes
        ax = self._axes[0]
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

        if cat is None:
            ax.scatter(X.x, Y.x, alpha=.5)
        else:
            labels = []; handles = []
            for color, cell in zip(c, cat.cells):
                idx = (cat == cell)
                Xi = X[idx]
                Yi = Y[idx]
                cell = str2tex(cellname(cell))

                h = ax.scatter(Xi.x, Yi.x, c=color, label=cell, alpha=.5)
                handles.append(h)
                labels.append(cell)

            if figlegend:
                self.figure.legend(handles, labels, lloc, ncol=lncol)
            else:
                ax.legend(lloc, ncol=lncol)

        self._show()


class Regression(_EelFigure):
    def __init__(self, Y, X, cat=None, match=None, sub=None, ds=None,
                 ylabel=True, title=None, alpha=.2, legend=True, delim=' ',
                 c=['#009CFF', '#FF7D26', '#54AF3A', '#FE58C6', '#20F2C3'],
                 **layout):
        """
        Plot the regression of Y on a regressor X.


        parameters
        ----------

        alpha : scalar
            alpha for individual data points (to control visualization of
            overlap)
        legend : bool | str
            applies if cat != None: can be mpl ax.legend() loc kwarg
            http://matplotlib.sourceforge.net/api/axes_api.html#matplotlib.axes.Axes.legend

        """
        sub = assub(sub, ds)
        Y = asvar(Y, sub, ds)
        X = asvar(X, sub, ds)
        if cat is not None:
            cat = ascategorial(cat, sub, ds)
        if match is not None:
            raise NotImplementedError("match kwarg")

        if ylabel is True:
            ylabel = Y.name

        _EelFigure.__init__(self, "Regression", 1, layout, 1, 5, figtitle=title)
        ax = self._axes[0]

        # labels
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.set_xlabel(X.name)
        if title:
            ax.set_title(title, **defaults['title_kwargs'])
        # regplot
        scatter_kwargs = {'s': 100,
                          'alpha': alpha,
                           'marker': 'o',
                           'label': '_nolegend_'}
        if cat is None:
            if type(c) in [list, tuple]:
                color = c[0]
            else:
                color = c
            y = Y.x
            reg = X.x
            ax.scatter(reg, y, edgecolor=color, facecolor=color, **scatter_kwargs)
            x, y = _reg_line(y, reg)
            ax.plot(x, y, c=color)
        else:
            for i, cell in enumerate(cat.cells):
                idx = (cat == cell)
                # scatter
                y = Y.x[idx]
                reg = X.x[idx]
                color = c[i % len(c)]
                ax.scatter(reg, y, edgecolor=color, facecolor=color, **scatter_kwargs)
                # regression line
                x, y = _reg_line(y, reg)
                ax.plot(x, y, c=color, label=cellname(cell))
            if legend == True:
                ax.legend()
            elif legend != False:
                ax.legend(loc=legend)

        self._show()


# MARK: Requirements for Statistical Tests

def _difference(data, names):
    "data condition x subject"
    data_differences = []; diffnames = []; diffnames_2lines = []
    for i, (name1, data1) in enumerate(zip(names, data)):
        for name2, data2 in zip(names[i + 1:], data[i + 1:]):
            data_differences.append(data1 - data2)
            diffnames.append('-'.join([name1, name2]))
            diffnames_2lines.append('-\n'.join([name1, name2]))
    return data_differences, diffnames, diffnames_2lines




def _normality_plot(ax, data, **kwargs):
    """helper fubction for creating normality test figure"""
    n, bins, patches = ax.hist(data, normed=True, **kwargs)
    data = np.ravel(data)

    # normal line
    mu = np.mean(data)
    sigma = np.std(data)
    y = mpl.mlab.normpdf(bins, mu, sigma)
    ax.plot(bins, y, 'r--', linewidth=1)

    # TESTS
    # test Anderson
    A2, thresh, sig = scipy.stats.morestats.anderson(data)
    index = sum(A2 >= thresh)
    if index > 0:
        ax.text(.95, .95, '$^{*}%s$' % str(sig[index - 1] / 100), color='r', size=11,
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes,)
    logging.debug(" Anderson: %s, %s, %s" % (A2, thresh, sig))
    # test Lilliefors
    n_test = test.lilliefors(data)
    ax.set_xlabel(r"$D=%.3f$, $p_{est}=%.2f$" % n_test)  # \chi ^{2}
    # make sure ticks display int values
    # ax.yaxis.set_major_formatter(ticker.MaxNLocator(nbins=8, integer=True))
    ticks = ax.get_yticks()
    ticks_int = [int(l) for l in ticks]
    ax.set_yticks(ticks_int)
    ax.set_yticklabels(ticks_int)


class Histogram(_EelFigure):
    def __init__(self, Y, X=None, match=None, sub=None, ds=None, pooled=True,
                title=True, ylabel=True, **layout):
        """Make histogram plots and test normality.

        Parameters
        ----------
        pooled : bool
            Add one plot with all values/differences pooled.
        title : None | str
            Figure title.
        titlekwargs : dict
            Forwarded to :py:func:`pyplot.suptitle`.
        """
        ct = Celltable(Y, X, match=match, sub=sub, ds=ds, coercion=asvar)

        # ylabel
        if ylabel is True:
            ylabel = str2tex(getattr(ct.Y, 'name', False))

        # layout
        if X is None:
            nax = 1
            if title is True:
                title = "Test for Normality"
        elif ct.all_within:
            nax = None
            if title is True:
                title = "Tests for Normality of the Differences"
        else:
            nax = len(ct.cells)
            if title is True:
                title = "Tests for Normality"

        _EelFigure.__init__(self, "Histogram", nax, layout, 1, 4,
                            figtitle=title)

        if X is None:
            ax = self._axes[0]
            _normality_plot(ax, ct.Y.x)
        elif ct.all_within:
            # temporary:
            data = ct.get_data()
            names = ct.cellnames()

#             plt.subplots_adjust(hspace=.5)

            n_cmp = len(ct.cells) - 1
            pooled = []
            # i: row
            # j: -column
            for i in range(0, n_cmp + 1):
                for j in range(i + 1, n_cmp + 1):
                    difference = data[i] - data[j]
                    pooled.append(scipy.stats.zscore(difference))  # z transform?? (scipy.stats.zs())
                    ax_i = n_cmp * i + (n_cmp + 1 - j)
                    ax = self.figure.add_subplot(n_cmp, n_cmp, ax_i)
                    _normality_plot(ax, difference)
                    if i == 0:
                        ax.set_title(names[j], size=12)
                    if j == n_cmp:
                        ax.set_ylabel(names[i], size=12)
            # pooled diffs
            if len(names) > 2:
                ax = self.figure.add_subplot(n_cmp, n_cmp, n_cmp ** 2)
                _normality_plot(ax, pooled, facecolor='g')
                ax.set_title("Pooled Differences (n=%s)" % len(pooled),
                             weight='bold')
                self.figure.text(.99, .01, "$^{*}$ Anderson and Darling test "
                                 "thresholded at $[ .15,   .10,    .05,    "
                                 ".025,   .01 ]$.", color='r',
                                 verticalalignment='bottom',
                                 horizontalalignment='right')
        else:  # independent measures
#             plt.subplots_adjust(hspace=.5, left=.1, right=.9, bottom=.1, top=.8)
            for cell, ax in izip(ct.cells, self._axes):
                ax.set_title(cellname(cell))
                _normality_plot(ax, ct.data[cell])


def boxcox_explore(Y, params=[-1, -.5, 0, .5, 1], crange=False, ax=None, box=True):
    """
    Plot the distribution resulting from a Box-Cox transformation with
    different parameters.


    Parameters
    ----------

    Y : array-like
        array containing data whose distribution is to be examined
    crange :
        correct range of transformed data
    ax :
        ax to plot to (``None`` creates a new figure)
    box :
        use Box-Cox family

    """
    if hasattr(Y, 'x'):
        Y = Y.x
    else:
        Y = np.ravel(Y)

    if np.any(Y == 0):
        raise ValueError("data contains 0")

    y = []
    for p in params:
        if p == 0:
            if box:
                xi = np.log(Y)
            else:
                xi = np.log10(Y)
                # xi = np.log1p(x)
        else:
            if box:
                xi = (Y ** p - 1) / p
            else:
                xi = Y ** p
        if crange:
            xi -= min(xi)
            xi /= max(xi)
        y.append(xi)

    if not ax:
        plt.figure()
        ax = plt.subplot(111)

    ax.boxplot(y)
    ax.set_xticks(range(1, 1 + len(params)))
    ax.set_xticklabels(params)
    ax.set_xlabel("p")
    if crange:
        ax.set_ylabel("Value (Range Corrected)")


# backwards comp
barplot = Barplot
boxplot = Boxplot
corrplot = Correlation
histogram = Histogram
multitimeplot = MultiTimeplot
regplot = Regression
timeplot = Timeplot
