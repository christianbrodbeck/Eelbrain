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
from .mpl_canvas import FigureCanvasPanel
from .history import Action, FileDocument, FileModel, FileFrame


COLOR = {True: (.5, 1, .5), False: (1, .3, .3)}


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
            ax.id = i
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
            self.model.toggle(event.inaxes.id)

    def OnCanvasKey(self, event):
        if not event.inaxes:
            return
        i = event.inaxes.id
        if event.key == 't':
            plot.Topomap(self.doc.components[i], w=10, sensorlabels='name',
                         title='# %i' % i)
        elif event.key == 'a':
            plot.Array(self.doc.sources[i], w=10, h=10, title='# %i' % i,
                       axtitle=False, interpolation='none')

    def OnPanelResize(self, event):
        w, h = event.GetSize()
        n_h = w // self.ax_size
        if n_h >= 2 and n_h != self.n_h:
            self.plot()

    def OnPointerEntersAxes(self, event):
        sb = self.GetStatusBar()
        if event.inaxes:
            sb.SetStatusText("#%i of %i ICA Components" %
                             (event.inaxes.id, len(self.doc.components)))
        else:
            sb.SetStatusText("%i ICA Components" % len(self.doc.components))

    def OnUpdateUIOpen(self, event):
        event.Enable(False)
