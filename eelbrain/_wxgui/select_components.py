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
from .._data_obj import NDVar, Ordered, fft, isfactor
from ..plot._topo import _ax_topomap
from .._wxutils import Icon, ID, REValidator
from .._utils.parse import POS_FLOAT_PATTERN
from .mpl_canvas import FigureCanvasPanel, CanvasFrame
from .text import TextFrame
from .history import Action, FileDocument, FileModel, FileFrame
from .frame import EelbrainDialog


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

    Parameters
    ----------
    path : str
        Path to the ICA file.
    ds : Dataset
        Dataset containing 'epochs' (mne Epochs), 'index' (Var describing
        epochs) and variables describing cases in epochs, used to plot
        condition averages.
    """
    def __init__(self, path, ds, sysname):
        FileDocument.__init__(self, path)

        self.ica = ica = mne.preprocessing.read_ica(path)
        self.accept = np.ones(self.ica.n_components_, bool)
        self.accept[ica.exclude] = False
        self.epochs = epochs = ds['epochs']
        self.epochs_ndvar = load.fiff.epochs_ndvar(epochs, sysname=sysname)
        self.ds = ds

        data = np.dot(ica.mixing_matrix_.T, ica.pca_components_[:ica.n_components_])
        self.components = NDVar(data, ('case', self.epochs_ndvar.sensor),
                                info={'meas': 'component', 'cmap': 'xpolar'})

        # sources
        data = ica.get_sources(epochs).get_data().swapaxes(0, 1)
        if 'index' in ds:
            epoch_index = ds['index']
        else:
            epoch_index = np.arange(len(epochs))
        epoch_dim = Ordered('epoch', epoch_index)
        self.sources = NDVar(data, ('case', epoch_dim, self.epochs_ndvar.time),
                             info={'meas': 'component', 'cmap': 'xpolar'})

        # publisher
        self._case_change_subscriptions = []

    def apply(self, inst, copy=True):
        return self.ica.apply(inst, copy=copy)

    def set_case(self, index, state):
        self.accept[index] = state
        self.ica.exclude = list(np.flatnonzero(np.invert(self.accept)))
        for func in self._case_change_subscriptions:
            func(index)

    def save(self):
        self.ica.save(self.path)

    def subscribe_to_case_change(self, callback):
        "callback(index)"
        self._case_change_subscriptions.append(callback)

    def unsubscribe_to_case_change(self, callback):
        "callback(index)"
        if callback in self._case_change_subscriptions:
            self._case_change_subscriptions.remove(callback)


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


class ContextMenu(wx.Menu):
    "Helper class for Menu to store component ID"
    def __init__(self, i):
        wx.Menu.__init__(self)
        self.i = i


class Frame(FileFrame):
    """
    Component Selection
    ===================

    Select components from ICA.

    * Click on components topographies to select/deselect them.
    * Use the context-menu (right click) for additional commands.


    *Keyboard shortcuts* in addition to the ones in the menu:

    =========== ============================================================
    Key         Effect
    =========== ============================================================
    t           topomap plot of the Component under the pointer
    a           array-plot of the source time course
    s           plot sources, starting with the component under the cursor
    f           plot the frequency spectrum for the selected component
    b           Butterfly plot of grand average (original and cleaned)
    B           Butterfly plot of condition averages
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

        # Buttons
        button = wx.Button(tb, ID.SHOW_SOURCES, "Sources")
        button.Bind(wx.EVT_BUTTON, self.OnShowSources)
        tb.AddControl(button)
        button = wx.Button(tb, ID.FIND_RARE_EVENTS, "Rare Events")
        button.Bind(wx.EVT_BUTTON, self.OnFindRareEvents)
        tb.AddControl(button)

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
        # re-Bind right click
        self.canvas.Unbind(wx.EVT_RIGHT_DOWN)
        self.canvas.Bind(wx.EVT_RIGHT_DOWN, self.OnRightDown)

        # attributes
        self.last_model = ""

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
        if event.button == 1:
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
            self.ShowSources(event.inaxes.i)
        elif event.key == 'f':
            self.PlotCompFFT(event.inaxes.i)
        elif event.key == 'b':
            self.PlotEpochButterfly(-1)
        elif event.key == 'B':
            self.PlotConditionAverages(self)

    def OnFindRareEvents(self, event):
        n_events = self.config.ReadInt("FindRareEvents/n_events", 5)
        threshold = self.config.ReadFloat("FindRareEvents/threshold", 2.)
        dlg = FindRareEventsDialog(self, threshold, n_events)
        rcode = dlg.ShowModal()
        dlg.Destroy()
        if rcode != wx.ID_OK:
            return
        threshold, n_events = dlg.GetValues()
        self.config.WriteInt("FindRareEvents/n_events", n_events)
        self.config.WriteFloat("FindRareEvents/threshold", threshold)

        def outliers(source):
            y = source - source.mean()
            y /= y.std()
            y **= 2
            ss = y.sum('time')
            idx = np.flatnonzero(ss.x > threshold * ss.std())
            return source.epoch.values[idx]

        res = []
        for i in xrange(len(self.doc.sources)):
            outl = outliers(self.doc.sources[i])
            if len(outl) <= n_events:
                res.append((i, outl))

        if len(res) == 0:
            wx.MessageBox("No rare events were found.", "No Rare Events Found",
                          style=wx.ICON_INFORMATION)
            return

        items = ["Components that have strong relative loading (SS > %g STD) "
                 "on %i or fewer epochs:" % (threshold, n_events), '']
        for i, epochs in res:
            items.append("# %i:  %s" % (i, ', '.join(map(str, epochs))))

        TextFrame(self, "Rare Events", '\n'.join(items))

    def OnPanelResize(self, event):
        w, h = event.GetSize()
        n_h = w // self.ax_size
        if n_h >= 2 and n_h != self.n_h:
            self.plot()

    def OnPlotCompSourceArray(self, event):
        self.PlotCompSourceArray(event.EventObject.i)

    def OnPlotCompTopomap(self, event):
        self.PlotCompTopomap(event.EventObject.i)

    def OnPointerEntersAxes(self, event):
        sb = self.GetStatusBar()
        if event.inaxes:
            sb.SetStatusText("#%i of %i ICA Components" %
                             (event.inaxes.i, len(self.doc.components)))
        else:
            sb.SetStatusText("%i ICA Components" % len(self.doc.components))

    def OnRankEpochs(self, event):
        i_comp = event.EventObject.i
        source = self.doc.sources[i_comp]
        y = source - source.mean()
        y /= y.std()
        y **= 2
        ss = y.sum('time').x  # ndvar has epoch as index

        # sort
        sort = np.argsort(ss)[::-1]
        ss = ss[sort]
        epoch_idx = source.epoch.values[sort]

        # text
        text = ["Component # %i, epochs SS in descending order:" % i_comp, ""]
        for pair in izip(epoch_idx, ss):
            text.append("% 4i: %.1f" % pair)

        TextFrame(self, "Component %i Epoch SS" % i_comp, '\n'.join(text))

    def OnRightDown(self, event):
        ax = self.canvas.MatplotlibEventAxes(event)
        if not ax:
            return

        # costruct menu
        menu = ContextMenu(ax.i)
        item = menu.Append(wx.ID_ANY, "Rank Epochs")
        self.Bind(wx.EVT_MENU, self.OnRankEpochs, item)
        menu.AppendSeparator()
        item = menu.Append(wx.ID_ANY, "Plot Topomap")
        self.Bind(wx.EVT_MENU, self.OnPlotCompTopomap, item)
        item = menu.Append(wx.ID_ANY, "Plot Source Array")
        self.Bind(wx.EVT_MENU, self.OnPlotCompSourceArray, item)

        # show menu
        pos = self.panel.CalcScrolledPosition(event.Position)
        self.PopupMenu(menu, pos)
        menu.Destroy()

    def OnShowSources(self, event):
        self.ShowSources(0)

    def OnUpdateUIOpen(self, event):
        event.Enable(False)

    def PlotCompFFT(self, i_comp):
        plot.UTS(fft(self.doc.sources[i_comp]).mean('epoch'), w=8,
                 title="# %i Frequency Spectrum" % i_comp)

    def PlotCompSourceArray(self, i_comp):
        plot.Array(self.doc.sources[i_comp], w=10, h=10, title='# %i' % i_comp,
                   axtitle=False, interpolation='none')

    def PlotCompTopomap(self, i_comp):
        plot.Topomap(self.doc.components[i_comp], w=10, sensorlabels='name',
                     title='# %i' % i_comp)

    def PlotConditionAverages(self, parent):
        "Prompt for model and plot condition averages"
        factors = [n for n, v in self.doc.ds.iteritems() if isfactor(v)]
        if len(factors) == 0:
            wx.MessageBox("The dataset that describes the epochs does not "
                          "contain any Factors that could be used to plot the "
                          "data by condition.", "No Factors in Dataset",
                          style=wx.ICON_ERROR)
            return
        elif len(factors) == 1:
            default = factors[0]
        else:
            default = self.last_model or factors[0]
        msg = "Specify the model (available factors: %s)" % ', '.join(factors)

        plot_model = None
        dlg = wx.TextEntryDialog(parent, msg, "Plot by Condition", default)
        while plot_model is None:
            if dlg.ShowModal() == wx.ID_OK:
                value = dlg.GetValue()
                use = [s.strip() for s in value.replace(':', '%').split('%')]
                invalid = [f for f in use if f not in factors]
                if invalid:
                    wx.MessageBox("The following are not valid factor names: %s"
                                  % (', '.join(invalid)), "Invalid Entry",
                                  wx.ICON_ERROR)
                else:
                    plot_model = '%'.join(use)
            else:
                dlg.Destroy()
                return
        dlg.Destroy()
        self.last_model = value

        ds = self.doc.ds.aggregate(plot_model, drop_bad=True)
        original = ds['epochs']
        cleaned = load.fiff.evoked_ndvar(map(self.doc.apply, original))
        vmax = 1.1 * max(abs(cleaned.min()), cleaned.max())
        for i in xrange(len(original)):
            name = ' '.join(ds[i, f] for f in use) + ' (n=%i)' % ds[i, 'n']
            plot.TopoButterfly([original[i], cleaned[i]], vmax=vmax,
                               title=name, axlabel=("Original", "Cleaned"))

    def PlotEpochButterfly(self, i_epoch):
        if i_epoch == -1:
            original_epoch = self.doc.epochs.average()
            name = "Epochs Average"
            vmax = None
        else:
            original_epoch = self.doc.epochs[i_epoch]
            name = "Epoch %i" % self.doc.sources.epoch[i_epoch]
            vmax = 2e-12
        clean_epoch = self.doc.apply(original_epoch)
        plot.TopoButterfly([original_epoch, clean_epoch], vmax=vmax,
                           title=name, axlabel=("Original", "Cleaned"))

    def ShowSources(self, i_first):
        SourceFrame(self, i_first)


class SourceFrame(CanvasFrame):
    """
    Component Source Time Course
    ============================

    Select components from ICA.

    * Click on components topographies to select/deselect them.


    *Keyboard shortcuts* in addition to the ones in the menu:

    =========== ============================================================
    Key         Effect
    =========== ============================================================
    t           topomap plot of the Component under the pointer
    a           array-plot of the source time course
    f           plot the frequency spectrum for the selected component
    b           butterfly plot of the epoch (original and cleaned)
    B           Butterfly plot of condition averages
    =========== ============================================================
    """
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
        self.n_comp = 0  # updated by plot
        self.n_comp_in_ica = len(self.doc.components)
        self.i_first = i_first
        self.n_epochs = 20
        self.i_first_epoch = 0
        self.n_epochs_in_data = len(self.doc.sources.epoch)
        self.y_scale = 5  # scale factor for y axis
        self._plot()

        # event bindings
        self.doc.subscribe_to_case_change(self.CaseChanged)
        self.Bind(wx.EVT_TOOL, self.OnUp, id=wx.ID_UP)
        self.Bind(wx.EVT_TOOL, self.OnDown, id=wx.ID_DOWN)
        self.Bind(wx.EVT_TOOL, self.OnBackward, id=wx.ID_BACKWARD)
        self.Bind(wx.EVT_TOOL, self.OnForward, id=wx.ID_FORWARD)
        self.canvas.mpl_connect('button_press_event', self.OnCanvasClick)
        self.canvas.mpl_connect('key_release_event', self.OnCanvasKey)
        self.Show()

    def _fill_toolbar(self, tb):
        self.up_button = tb.AddLabelTool(wx.ID_UP, "Up",
                                         Icon("tango/actions/go-up"))
        self.down_button = tb.AddLabelTool(wx.ID_DOWN, "Down",
                                           Icon("tango/actions/go-down"))
        self.back_button = tb.AddLabelTool(wx.ID_BACKWARD, "Back",
                                           Icon("tango/actions/go-previous"))
        self.next_button = tb.AddLabelTool(wx.ID_FORWARD, "Next",
                                           Icon("tango/actions/go-next"))

    def _get_source_data(self):
        "component by time"
        n_comp = self.n_comp
        n_comp_actual = self.n_comp_actual
        e_start = self.doc.sources.epoch[self.i_first_epoch]
        i_stop = self.i_first_epoch + self.n_epochs
        if i_stop >= len(self.doc.sources.epoch):
            e_stop = None
        else:
            e_stop = self.doc.sources.epoch[i_stop]
        data = self.doc.sources.sub(case=slice(self.i_first, self.i_first + n_comp),
                                    epoch=(e_start, e_stop))
        y = data.get_data(('case', 'epoch', 'time')).reshape((n_comp_actual, -1))
        if y.base is not None and data.x.base is not None:
            y = y.copy()
        start = n_comp - 1
        stop = -1 + (n_comp - n_comp_actual)
        y += np.arange(start * self.y_scale, stop * self.y_scale, -self.y_scale)[:, None]
        return y, data.epoch.x

    def _plot(self):
        # partition figure
        self.figure.clf()
        figheight = self.figure.get_figheight()
        n_comp = int(self.figure.get_figheight() // self.size)
        self.n_comp = n_comp
        # make sure no empty lines
        if self.i_first and self.n_comp_in_ica - self.i_first < n_comp:
            self.i_first = max(0, self.n_comp_in_ica - n_comp)
        # further layout-relevant properties
        n_comp_actual = min(self.n_comp_in_ica - self.i_first, n_comp)
        self.n_comp_actual = n_comp_actual
        elen = len(self.doc.sources.time)

        # topomaps
        axheight = self.size / figheight
        axwidth = self.size / self.figure.get_figwidth()
        left = axwidth / 2
        self.topo_plots = []
        self.topo_labels = []
        for i in xrange(n_comp_actual):
            i_comp = self.i_first + i
            ax = self.figure.add_axes((left, 1 - (i + 1) * axheight, axwidth, axheight))
            p = _ax_topomap(ax, [self.doc.components[i_comp]], None)
            text = ax.text(0, 0.5, "# %i" % i_comp, va='center', ha='right', color='k')
            ax.i = i
            ax.i_comp = i_comp
            self.topo_plots.append(p)
            self.topo_labels.append(text)

        # source time course data
        y, tick_labels = self._get_source_data()

        # axes
        left = 1.5 * axwidth
        bottom = 1 - n_comp * axheight
        ax = self.figure.add_axes((left, bottom, 1 - left, 1 - bottom),
                                  frameon=False, yticks=(),
                                  xticks=np.arange(elen / 2, elen * self.n_epochs, elen),
                                  xticklabels=tick_labels)
        ax.tick_params(bottom=False)
        ax.i = -1
        ax.i_comp = None

        # store canvas before plotting lines
        self.canvas.draw()
        self.canvas.store_canvas()

        # plot epochs
        self.lines = ax.plot(y.T, color=LINE_COLOR[True], clip_on=False)
        ax.set_ylim((-0.5 * self.y_scale, (n_comp - 0.5) * self.y_scale))
        ax.set_xlim((0, y.shape[1]))
        # line color
        reject_color = LINE_COLOR[False]
        for i in xrange(n_comp_actual):
            if not self.doc.accept[i + self.i_first]:
                self.lines[i].set_color(reject_color)
        # epoch demarcation
        for x in xrange(elen, elen * self.n_epochs, elen):
            ax.axvline(x, ls='--', c='k')
        # epoch labels


        self.ax_tc = ax
        self.canvas.draw()

    def CanBackward(self):
        return self.i_first_epoch > 0

    def CanDown(self):
        return self.i_first + self.n_comp < self.n_comp_in_ica

    def CanForward(self):
        return self.i_first_epoch + self.n_epochs < self.n_epochs_in_data

    def CanUp(self):
        return self.i_first > 0

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
        if event.inaxes:
            if event.inaxes.i_comp is None:
                i_comp = int(self.i_first + self.n_comp - ceil(event.ydata / self.y_scale + 0.5))
            else:
                i_comp = event.inaxes.i_comp
            self.model.toggle(i_comp)

    def OnCanvasKey(self, event):
        if event.key == 'down':
            if self.CanDown():
                self.OnDown(None)
        if event.key == 'up':
            if self.CanUp():
                self.OnUp(None)
        if event.key == 'right':
            if self.CanForward():
                self.OnForward(None)
        elif event.key == 'left':
            if self.CanBackward():
                self.OnBackward(None)
        elif event.key == 'B':
            self.parent.PlotConditionAverages(self)
        elif not event.inaxes:
            return
        elif event.inaxes.i_comp is None:
            if event.key == 'b':
                i_epoch = self.i_first_epoch + int(event.xdata // len(self.doc.sources.time))
                if i_epoch < len(self.doc.epochs):
                    self.parent.PlotEpochButterfly(i_epoch)
            return
        elif event.key == 't':
            self.parent.PlotCompTopomap(event.inaxes.i_comp)
        elif event.key == 'a':
            self.parent.PlotCompSourceArray(event.inaxes.i_comp)
        elif event.key == 'f':
            self.parent.PlotCompFFT(event.inaxes.i_comp)
        elif event.key == 'b':
            self.parent.PlotEpochButterfly(-1)

    def OnClose(self, event):
        self.doc.unsubscribe_to_case_change(self.CaseChanged)
        super(SourceFrame, self).OnClose(event)

    def OnDown(self, event):
        "turns the page backward"
        self.SetFirstComponent(self.i_first + self.n_comp)

    def OnForward(self, event):
        "turns the page forward"
        self.SetFirstEpoch(self.i_first_epoch + self.n_epochs)

    def OnUp(self, event):
        "turns the page backward"
        self.SetFirstComponent(self.i_first - self.n_comp)

    def OnUpdateUIBackward(self, event):
        event.Enable(self.CanBackward())

    def OnUpdateUIDown(self, event):
        event.Enable(self.CanDown())

    def OnUpdateUIForward(self, event):
        event.Enable(self.CanForward())

    def OnUpdateUIUp(self, event):
        event.Enable(self.CanUp())

    def SetFirstComponent(self, i_first):
        if i_first < 0:
            i_first = 0
        elif i_first >= self.n_comp_in_ica:
            i_first = self.n_comp_in_ica - 1

        n_comp_actual = min(self.n_comp_in_ica - i_first, self.n_comp)
        for i in xrange(n_comp_actual):
            p = self.topo_plots[i]
            i_comp = i_first + i
            p.set_data([self.doc.components[i_comp]], True)
            p.ax.i_comp = i_comp
            self.topo_labels[i].set_text("# %i" % i_comp)
            self.lines[i].set_color(LINE_COLOR[self.doc.accept[i_comp]])

        if n_comp_actual < self.n_comp:
            empty_data = self.doc.components[0].copy()
            empty_data.x.fill(0)
            for i in xrange(n_comp_actual, self.n_comp):
                p = self.topo_plots[i]
                p.set_data([empty_data])
                p.ax.i_comp = -1
                self.topo_labels[i].set_text("")
                self.lines[i].set_color('white')


        self.i_first = i_first
        self.n_comp_actual = n_comp_actual
        self.SetFirstEpoch(self.i_first_epoch)

    def SetFirstEpoch(self, i_first_epoch):
        self.i_first_epoch = i_first_epoch
        y, tick_labels = self._get_source_data()
        if i_first_epoch + self.n_epochs > self.n_epochs_in_data:
            elen = len(self.doc.sources.time)
            n_missing = self.i_first_epoch + self.n_epochs - self.n_epochs_in_data
            pad_time = elen * n_missing
        else:
            pad_time = 0

        if self.n_comp_actual < self.n_comp:
            pad_comp = self.n_comp - self.n_comp_actual
        else:
            pad_comp = 0

        if pad_time or pad_comp:
            y = np.pad(y, ((0, pad_comp), (0, pad_time)), 'constant')

        for line, data in izip(self.lines, y):
            line.set_ydata(data)
        self.ax_tc.set_xticklabels(tick_labels)
        self.canvas.draw()


class FindRareEventsDialog(EelbrainDialog):
    def __init__(self, parent, threshold, n_events, *args, **kwargs):
        super(FindRareEventsDialog, self).__init__(parent, wx.ID_ANY,
                                                   "Find Rare Events", *args,
                                                   **kwargs)
        sizer = wx.BoxSizer(wx.VERTICAL)

        # number of events
        sizer.Add(wx.StaticText(self, label="Max number of events:"))
        validator = REValidator("[1-9]\d*", "Invalid entry: {value}. Please "
                                "specify a number > 0.", False)
        ctrl = wx.TextCtrl(self, value=str(n_events), validator=validator)
        ctrl.SetHelpText("Max number of events to qualify as rare (positive integer)")
        ctrl.SelectAll()
        sizer.Add(ctrl)
        self.n_events_ctrl = ctrl

        # Threshold
        sizer.Add(wx.StaticText(self, label="Threshold for rare epoch SS (in STD):"))
        validator = REValidator(POS_FLOAT_PATTERN, "Invalid entry: {value}. Please "
                                "specify a number > 0.", False)
        ctrl = wx.TextCtrl(self, value=str(threshold), validator=validator)
        ctrl.SetHelpText("Epochs whose SS exceeds this value (in STD) are considered rare")
        ctrl.SelectAll()
        sizer.Add(ctrl)
        self.threshold_ctrl = ctrl

        # default button
        btn = wx.Button(self, wx.ID_DEFAULT, "Default Settings")
        sizer.Add(btn, border=2)
        btn.Bind(wx.EVT_BUTTON, self.OnSetDefault)

        # buttons
        button_sizer = wx.StdDialogButtonSizer()
        # ok
        btn = wx.Button(self, wx.ID_OK)
        btn.SetDefault()
        button_sizer.AddButton(btn)
        # cancel
        btn = wx.Button(self, wx.ID_CANCEL)
        button_sizer.AddButton(btn)
        # finalize
        button_sizer.Realize()
        sizer.Add(button_sizer)

        self.SetSizer(sizer)
        sizer.Fit(self)

    def GetValues(self):
        return (float(self.threshold_ctrl.GetValue()),
                int(self.n_events_ctrl.GetValue()))

    def OnSetDefault(self, event):
        self.threshold_ctrl.SetValue('2')
        self.n_events_ctrl.SetValue('5')
