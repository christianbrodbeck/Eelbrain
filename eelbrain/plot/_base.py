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


Top-level plotters can be called with nested lists of data-objects (ndvar
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
#. -> _ax_(A)
#. ----> _plt_(A)
#. -> _ax_(B)
#. ----> _plt_(B)
#. -> _ax_([diff(A,B), p(A, B)])
#. ----> _plt_(diff(A,B))
#. ----> _plt_(p(A, B))


Base Class
----------

:py:class:`eelfigure` is the baseclass for eelbrain plots. Based on
availablility of wxPython, it selects between a general matplotlib backend and
a wx backend allowing additional frame propertie4s such as custom toolbar
items.

If the mpl figure is used, pyplot.show() is called after the plot is done.
The module attribute ``show_block_arg`` is submitted to
``plt.show(block=show_block_arg)``.

"""

import os
import shutil
import subprocess
import tempfile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import PIL

from ..utils.subp import cmd_exists
from eelbrain import vessels as _vsl
from eelbrain.vessels import data as _dt

try:
    from ..wxutils.mpl_canvas import CanvasFrame
    backend = 'wx'
except:
    backend = 'mpl'


defaults = {
          'figsize':(6, -1),  # was 7
        }
title_kwargs = {'size': 18,
                'family': 'serif'}
figs = []  # store callback figures (they need to be preserved)
show_block_arg = True  # if the mpl figure is used, this is submitted to plt.show(block=show_block_arg)


def unpack_epochs_arg(ndvars, ndim, dataset=None, levels=1):
    """
    Returns a nested list of epochs (through the summary method)
    """
    ndvars = getattr(ndvars, '_default_plot_obj', getattr(ndvars, 'all', ndvars))
    if not isinstance(ndvars, (tuple, list)):
        ndvars = [ndvars]

    if levels > 0:
        return [unpack_epochs_arg(v, ndim, dataset, levels - 1) for v in ndvars]
    else:
        out = []
        for ndvar in ndvars:
            if isinstance(ndvar, basestring):
                ndvar = dataset[ndvar]

            if ndvar.ndim == ndim + 1:
                if ndvar.has_case:
                    if len(ndvar) == 1:
                        ndvar = ndvar.summary(name='{name}')
                    else:
                        ndvar = ndvar.summary()

            if ndvar.ndim == ndim:
                pass
            else:
                err = ("Plot requires ndim=%i; %r ndim==%i" %
                       (ndim, ndvar, ndvar.ndim))
                raise _dt.DimensionMismatchError(err)
            out.append(ndvar)
        return out



def read_cs_arg(epoch, colorspace=None):
    if (colorspace is None):
        if 'colorspace' in epoch.properties:
            colorspace = epoch.properties['colorspace']
        else:
            colorspace = _vsl.colorspaces.get_default()

    return colorspace




class mpl_figure:
    "cf. wxutils.mpl_canvas"
    def __init__(self, **fig_kwargs):
        "creates self.figure and self.canvas attributes and returns the figure"
        self.figure = plt.figure(**fig_kwargs)
        self.canvas = self.figure.canvas
        figs.append(self)

    def Close(self):
        plt.close(self.figure)

    def SetStatusText(self, text):
        pass

    def Show(self):
        if mpl.get_backend() == 'WXAgg':
            plt.show(block=show_block_arg)

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


class eelfigure(object):
    """
    Parent class for eelbrain figures.

    In order to subclass:

     - find desired figure properties and then use them to initialize
       the eelfigure superclass; then use the
       :py:attr:`eelfigure.figure` and :py:attr:`eelfigure.canvas` attributes.
     - end the initialization by calling `eelfigure._show()`
     - add the :py:meth:`_fill_toolbar` method


    """
    def __init__(self, title="Eelbrain Figure", **fig_kwargs):
        # find the right frame
        frame = None
        self._is_wx = False
        if backend == 'wx':
            try:
                frame = CanvasFrame(title=title, eelfigure=self, **fig_kwargs)
                self._is_wx = True
            except:
                pass
        if frame is None:
            frame = mpl_figure(**fig_kwargs)

        # store attributes
        self._frame = frame
        self.figure = frame.figure
        self.canvas = frame.canvas

        self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.canvas.mpl_connect('axes_leave_event', self._on_leave_axes)

    def _get_statusbar_text(self, event):
        "subclass to add figure-specific content"
        return '%s'

    def _on_leave_axes(self, event):
        "update the status bar when the cursor leaves axes"
        self._frame.SetStatusText(':-)')

    def _on_motion(self, event):
        "update the status bar for mouse movement"
        ax = event.inaxes
        if ax:
            y_fmt = getattr(ax, 'y_fmt', 'y = %.3g')
            x_fmt = getattr(ax, 'x_fmt', 'x = %.3g')
            # update status bar
            y_txt = y_fmt % event.ydata
            x_txt = x_fmt % event.xdata
            pos_txt = ',  '.join((x_txt, y_txt))

            txt = self._get_statusbar_text(event)
            self._frame.SetStatusText(txt % pos_txt)

    def _show(self):
        self.draw()
        self._frame.Show()

    def _fill_toolbar(self, tb):
        """
        Subclasses should add their toolbar items in this function which
        is called by CanvasFrame.FillToolBar()

        """
        pass

    def close(self):
        "Close the figure."
        self._frame.Close()

    def draw(self):
        "(Re-)draw the figure (after making manual changes)."
        self._frame.canvas.draw()



class subplot_figure(eelfigure):
    def _show(self, figtitle=None):
        self.figure.tight_layout()
        if figtitle:
            t = self.figure.suptitle(figtitle)
            trans = self.figure.transFigure.inverted()
            bbox = trans.transform(t.get_window_extent(self.figure.canvas.renderer))
            print bbox
            t_bottom = bbox[0, 1]
            self.figure.subplots_adjust(top=1 - 2 * (1 - t_bottom))

        super(subplot_figure, self)._show()



class legend(eelfigure):
    def __init__(self, handles, labels, dpi=90, figsize=(2, 2)):
        super(legend, self).__init__(title="Legend", dpi=dpi, figsize=figsize)

        self.legend = self.figure.legend(handles, labels, loc=2)

        self._show()


def unpack(Y, X):
    "Returns a list of Y[cell] corresponding to the cells in X"
    epochs = []
    for cell in X.cells:
        y = Y[X == cell]
        y.name = cell
        epochs.append(y)
    return epochs


class ImageTiler(object):
    """
    Create tiled images and animations from individual image files.

    """
    def __init__(self, ext='.png', nrow=1, ncol=1, nt=1, dest=None):
        """
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

        if not cmd_exists('ffmpeg'):
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
