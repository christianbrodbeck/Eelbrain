"""GUI for selecting topographical components"""
# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>

# Document:  represents data
# ChangeAction:  modifies Document
# Model:  creates ChangeActions and applies them to the History
# Frame:
#  - visualizaes Document
#  - listens to Document changes
#  - issues commands to Model

from __future__ import division

from itertools import izip
from math import ceil

import mne
from matplotlib.patches import Rectangle
import numpy as np
import wx
from wx.lib.scrolledpanel import ScrolledPanel

from .. import load, plot
from .._data_obj import NDVar, Ordered
from ..plot._topo import _ax_topomap
from .._wxutils import Icon, ID
from .mpl_canvas import FigureCanvasPanel, CanvasFrame
from .history import Action, FileDocument, FileModel, FileFrame


COLOR = {True: (.5, 1, .5), False: (1, .3, .3)}
LINE_COLOR = {True: 'k', False: (1, 0, 0)}


class ChangeAction(Action):
    """Action objects are kept in the history and can do and undo themselves"""

    def __init__(self, desc, index=None, old_accept=None, new_accept=None,
                 old_path=None, new_path=None):
        """
        Parameters
        ----------
        desc : str
            Description of the action
            list of (i, name, old, new) tuples
        """
        self.desc = desc
        self.index = index
        self.old_path = old_path
        self.old_accept = old_accept
        self.new_path = new_path
        self.new_accept = new_accept

    def do(self, doc):
        if self.index is not None:
            doc.set_case(self.index, self.new_accept)
        if self.new_path is not None:
            doc.set_path(self.new_path)

    def undo(self, doc):
        if self.index is not None:
            doc.set_case(self.index, self.old_accept)
        if self.new_path is not None and self.old_path is not None:
            doc.set_path(self.old_path)


class Document(FileDocument):
    """Represents data for the current state of the Document

    (Data can be accessed, but should only be modified through the Model)
    """
    def __init__(self, path, epochs, sysname=None):
        FileDocument.__init__(self, path)

        self.ica = ica = mne.preprocessing.read_ica(path)
        self.accept = np.ones(self.ica.n_components_, bool)
        self.accept[ica.exclude] = False
        self.epoch = epochs
        self.epochs_ndvar = load.fiff.epochs_ndvar(epochs, sysname=sysname)

        data = np.dot(ica.mixing_matrix_.T, ica.pca_components_[:ica.n_components_])
        self.components = NDVar(data, ('case', self.epochs_ndvar.sensor),
                                info={'meas': 'component', 'cmap': 'xpolar'})

        # sources
        data = ica.get_sources(epochs).get_data().swapaxes(0, 1)
        epoch_dim = Ordered('epoch', np.arange(len(epochs)))
        self.sources = NDVar(data, ('case', epoch_dim, self.epochs_ndvar.time),
                             info={'meas': 'component', 'cmap': 'xpolar'})

        # publisher
        self._case_change_subscriptions = []

    def set_case(self, index, state):
        self.accept[index] = state
        for func in self._case_change_subscriptions:
            func(index)

    def save(self):
        self.ica.exclude = list(np.flatnonzero(np.invert(self.accept)))
        self.ica.save(self.path)

    def subscribe_to_case_change(self, callback):
        "callback(index)"
        self._case_change_subscriptions.append(callback)


class Model(FileModel):
    """Manages a document with its history"""
    def set_case(self, index, state, desc="Manual Change"):
        old_accept = self.doc.accept[index]
        action = ChangeAction(desc, index, old_accept, state)
        self.history.do(action)

    def toggle(self, case):
        old_accept = self.doc.accept[case]
        action = ChangeAction("Manual toggle", case, old_accept, not old_accept)
        self.history.do(action)

    def clear(self):
        action = ChangeAction("Clear", slice(None), self.doc.accept.copy(),
                              True)
        self.history.do(action)


class Frame(FileFrame):
    """
    Component Selection
    ===================

    Select components from ICA.

    * Click on components topographies to select/deselect them.


    *Keyboard shortcuts* in addition to the ones in the menu:

    =========== ============================================================
    Key         Effect
    =========== ============================================================
    t           topomap plot of the Component under the pointer
    a           array-plot of the source time course
    s           plot sources, starting with the component under the cursor
    =========== ============================================================
    """
    _doc_name = 'component selection'
    _name = 'SelectComponents'
    _title = 'Select Components'
    _wildcard = "ICA fiff file (*-ica.fif)|*.fif"

    def __init__(self, parent, pos, size, model, n_h=2):
        super(Frame, self).__init__(parent, pos, size, model)

        # setup layout
        self.ax_size = 200
        figsize = (10, 10)
        self.SetMinSize((400, 400))

        # setup scrolled panel
        panel = ScrolledPanel(self)
        self.panel = panel

        # setup figure canvas
        self.canvas = FigureCanvasPanel(panel, figsize=figsize)
        self.canvas.figure.subplots_adjust(0, 0, 1, 1, 0, 0)
        panel.SetupScrolling(False, scrollToTop=False, scrollIntoView=False)

        # sizer
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas, 0)
        panel.SetSizer(sizer)
        self.canvas_sizer = sizer

        # Toolbar
        tb = self.CreateToolBar(wx.TB_HORIZONTAL)
        tb.SetToolBitmapSize(size=(32, 32))
        tb.AddLabelTool(wx.ID_SAVE, "Save",
                        Icon("tango/actions/document-save"), shortHelp="Save")
        tb.AddLabelTool(wx.ID_SAVEAS, "Save As",
                        Icon("tango/actions/document-save-as"),
                        shortHelp="Save As")
        # tb.AddLabelTool(wx.ID_OPEN, "Load",
        #                 Icon("tango/actions/document-open"),
        #                 shortHelp="Open Rejections")
        tb.AddLabelTool(ID.UNDO, "Undo", Icon("tango/actions/edit-undo"),
                        shortHelp="Undo")
        tb.AddLabelTool(ID.REDO, "Redo", Icon("tango/actions/edit-redo"),
                        shortHelp="Redo")
        tb.AddSeparator()

        # right-most part
        tb.AddStretchableSpace()

        # --> Help
        tb.AddLabelTool(wx.ID_HELP, 'Help', Icon("tango/apps/help-browser"))
        self.Bind(wx.EVT_TOOL, self.OnHelp, id=wx.ID_HELP)

        tb.Realize()

        self.CreateStatusBar()

        # Bind Events ---
        self.doc.subscribe_to_case_change(self.CaseChanged)
        self.panel.Bind(wx.EVT_SIZE, self.OnPanelResize)
        self.canvas.mpl_connect('axes_enter_event', self.OnPointerEntersAxes)
        self.canvas.mpl_connect('axes_leave_event', self.OnPointerEntersAxes)
        self.canvas.mpl_connect('button_press_event', self.OnCanvasClick)
        self.canvas.mpl_connect('key_release_event', self.OnCanvasKey)

        # Finalize
        self.plot()
        self.UpdateTitle()

    def plot(self):
        n = self.doc.ica.n_components_
        fig = self.canvas.figure
        fig.clf()

        panel_w = self.panel.GetSize()[0]
        n_h = max(2, panel_w // self.ax_size)
        n_v = int(ceil(n / n_h))

        # adjust canvas size
        size = (self.ax_size * n_h, self.ax_size * n_v)
        self.canvas_sizer.SetItemMinSize(self.canvas, size)

        # plot
        axes = tuple(fig.add_subplot(n_v, n_h, i) for i in xrange(1, n + 1))
        # bgs = tuple(ax.patch)
        for i, ax, c, accept in izip(xrange(n), axes, self.doc.components, self.doc.accept):
            _ax_topomap(ax, [c], None)
            ax.text(0.5, 1, "# %i" % i, ha='center', va='top')
            p = Rectangle((0, 0), 1, 1, color=COLOR[accept], zorder=-1)
            ax.add_patch(p)
            ax.i = i
            ax.background = p

        self.axes = axes
        self.n_h = n_h
        self.canvas.store_canvas()
        self.Layout()

    def CaseChanged(self, index):
        "updates the states of the segments on the current page"
        if isinstance(index, int):
            index = [index]
        elif isinstance(index, slice):
            start = index.start or 0
            stop = index.stop or len(self.doc.components)
            index = xrange(start, stop)
        elif index.dtype.kind == 'b':
            index = np.nonzero(index)[0]

        # update epoch plots
        axes = []
        for idx in index:
            ax = self.axes[idx]
            ax.background.set_color(COLOR[self.doc.accept[idx]])
            axes.append(ax)

        self.canvas.redraw(axes=axes)

    def OnCanvasClick(self, event):
        "called by mouse clicks"
        if event.inaxes:
            self.model.toggle(event.inaxes.i)

    def OnCanvasKey(self, event):
        if not event.inaxes:
            return
        if event.key == 't':
            self.PlotCompTopomap(event.inaxes.i)
        elif event.key == 'a':
            self.PlotCompSourceArray(event.inaxes.i)
        elif event.key == 's':
            SourceFrame(self, event.inaxes.i)

    def OnPanelResize(self, event):
        w, h = event.GetSize()
        n_h = w // self.ax_size
        if n_h >= 2 and n_h != self.n_h:
            self.plot()

    def OnPointerEntersAxes(self, event):
        sb = self.GetStatusBar()
        if event.inaxes:
            sb.SetStatusText("#%i of %i ICA Components" %
                             (event.inaxes.i, len(self.doc.components)))
        else:
            sb.SetStatusText("%i ICA Components" % len(self.doc.components))

    def OnUpdateUIOpen(self, event):
        event.Enable(False)

    def PlotCompSourceArray(self, i_comp):
        plot.Array(self.doc.sources[i_comp], w=10, h=10, title='# %i' % i_comp,
                   axtitle=False, interpolation='none')

    def PlotCompTopomap(self, i_comp):
        plot.Topomap(self.doc.components[i_comp], w=10, sensorlabels='name',
                     title='# %i' % i_comp)


class SourceFrame(CanvasFrame):
    _make_axes = False

    def __init__(self, parent, i_first, *args, **kwargs):
        super(SourceFrame, self).__init__(parent, "ICA Source Time Course", None, *args, **kwargs)
        self.figure.subplots_adjust(0, 0, 1, 1, 0, 0)
        self.figure.set_facecolor('white')

        self.SetRect(wx.GetClientDisplayRect())

        self.parent = parent
        self.model = parent.model
        self.doc = parent.model.doc
        self.size = 1
        self.i_first = i_first
        self.n_epochs = 20
        self.i_first_epoch = 0
        self.n_epochs_in_data = len(self.doc.sources.epoch)
        self.y_scale = 5  # scale factor for y axis
        self._plot()

        # event bindings
        self.doc.subscribe_to_case_change(self.CaseChanged)
        self.Bind(wx.EVT_TOOL, self.OnBackward, id=wx.ID_BACKWARD)
        self.Bind(wx.EVT_TOOL, self.OnForward, id=wx.ID_FORWARD)
        self.canvas.mpl_connect('button_press_event', self.OnCanvasClick)
        self.canvas.mpl_connect('key_release_event', self.OnCanvasKey)
        self.Show()

    def _fill_toolbar(self, tb):
        self.back_button = tb.AddLabelTool(wx.ID_BACKWARD, "Back",
                                           Icon("tango/actions/go-previous"))
        self.next_button = tb.AddLabelTool(wx.ID_FORWARD, "Next",
                                           Icon("tango/actions/go-next"))

    def _get_source_data(self):
        "component by time"
        n_comp = self.n_comp
        n_comp_actual = self.n_comp_actual
        data = self.doc.sources.sub(case=slice(self.i_first, self.i_first + n_comp),
                                    epoch=(self.i_first_epoch, self.i_first_epoch + self.n_epochs))
        y = data.get_data(('case', 'epoch', 'time')).reshape((n_comp_actual, -1))
        if y.base is not None and data.x.base is not None:
            y = y.copy()
        start = n_comp - 1
        stop = -1 + (n_comp - n_comp_actual)
        y += np.arange(start * self.y_scale, stop * self.y_scale, -self.y_scale)[:, None]
        return y

    def _plot(self):
        self.figure.clf()
        figheight = self.figure.get_figheight()
        n_comp = int(self.figure.get_figheight() // self.size)
        n_comp_actual = min(len(self.doc.components) - self.i_first, n_comp)
        self.n_comp = n_comp
        self.n_comp_actual = n_comp_actual

        # topomaps
        axheight = self.size / figheight
        axwidth = self.size / self.figure.get_figwidth()
        left = axwidth / 2
        for i in xrange(n_comp_actual):
            i_comp = self.i_first + i
            ax = self.figure.add_axes((left, 1 - (i + 1) * axheight, axwidth, axheight))
            _ax_topomap(ax, [self.doc.components[i_comp]], None)
            ax.text(0, 0.5, "# %i" % i_comp, va='center', ha='right', color='k')
            ax.i = i
            ax.i_comp = i_comp

        # source time course
        left = 1.5 * axwidth
        bottom = 1 - n_comp * axheight
        ax = self.figure.add_axes((left, bottom, 1 - left, 1 - bottom),
                                  frameon=False, yticks=(), xticks=())
        ax.i = -1
        ax.i_comp = None

        # store canvas before plotting lines
        self.canvas.draw()
        self.canvas.store_canvas()

        # plot epochs
        y = self._get_source_data()
        self.lines = ax.plot(y.T, color=LINE_COLOR[True], clip_on=False)
        ax.set_ylim((-0.5 * self.y_scale, (n_comp - 0.5) * self.y_scale))
        ax.set_xlim((0, y.shape[1]))
        # line color
        reject_color = LINE_COLOR[False]
        for i in xrange(n_comp_actual):
            if not self.doc.accept[i + self.i_first]:
                self.lines[i].set_color(reject_color)
        # epoch markers
        elen = len(self.doc.sources.time)
        for x in xrange(elen, elen * self.n_epochs, elen):
            ax.axvline(x, ls='--', c='k')

        self.ax_tc = ax
        self.canvas.draw()

    def CanBackward(self):
        return self.i_first_epoch > 0

    def CanForward(self):
        return self.i_first_epoch + self.n_epochs < self.n_epochs_in_data

    def CaseChanged(self, index):
        "updates the states of the segments on the current page"
        if isinstance(index, int):
            index = [index]
        elif isinstance(index, slice):
            start = index.start or 0
            stop = index.stop or self.doc.n_epochs
            index = xrange(start, stop)
        elif index.dtype.kind == 'b':
            index = np.nonzero(index)[0]

        # filter to visible epochs
        i_last = self.i_first + self.n_comp_actual
        index = [i_comp for i_comp in index if self.i_first <= i_comp <= i_last]
        # update epoch plots
        if index:
            for i_comp in index:
                self.lines[i_comp - self.i_first].set_color(LINE_COLOR[self.doc.accept[i_comp]])
            self.canvas.redraw(axes=(self.ax_tc,))

    def OnBackward(self, event):
        "turns the page backward"
        self.SetFirstEpoch(self.i_first_epoch - self.n_epochs)

    def OnCanvasClick(self, event):
        "called by mouse clicks"
        if event.inaxes and event.inaxes.i_comp is not None:
            self.model.toggle(event.inaxes.i_comp)

    def OnCanvasKey(self, event):
        if event.key == 'right':
            if self.CanForward():
                self.OnForward(None)
        elif event.key == 'left':
            if self.CanBackward():
                self.OnBackward(None)
        elif not event.inaxes or event.inaxes.i_comp is None:
            return
        elif event.key == 't':
            self.parent.PlotCompTopomap(event.inaxes.i_comp)
        elif event.key == 'a':
            self.parent.PlotCompSourceArray(event.inaxes.i_comp)

    def OnForward(self, event):
        "turns the page forward"
        self.SetFirstEpoch(self.i_first_epoch + self.n_epochs)

    def OnUpdateUIBackward(self, event):
        event.Enable(self.CanBackward())

    def OnUpdateUIForward(self, event):
        event.Enable(self.CanForward())

    def SetFirstEpoch(self, i_first_epoch):
        self.i_first_epoch = i_first_epoch
        y = self._get_source_data()
        if i_first_epoch + self.n_epochs > self.n_epochs_in_data:
            elen = len(self.doc.sources.time)
            n_missing = self.i_first_epoch + self.n_epochs - self.n_epochs_in_data
            y = np.pad(y, ((0, 0), (0, elen * n_missing)), 'constant')

        for line, data in izip(self.lines, y):
            line.set_ydata(data)
        self.canvas.draw()
