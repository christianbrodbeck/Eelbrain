"""GUI for rejecting epochs"""

# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>

# Document:  represents data
# ChangeAction:  modifies Document
# Model:  creates ChangeActions and applies them to the History
# Frame:
#  - visualizaes Document
#  - listens to Document changes
#  - issues commands to Model

from __future__ import division

import math
import os
import time

import mne
import numpy as np
import wx

from .. import load, save, plot
from .._data_obj import Dataset, Factor, Var, Datalist, corr, asndvar, combine
from .._names import INTERPOLATE_CHANNELS
from .._info import BAD_CHANNELS
from ..plot._base import find_fig_vlims
from ..plot._nuts import _plt_bin_nuts
from ..plot._topo import _ax_topomap
from ..plot._utsnd import _ax_bfly_epoch
from .._utils.numpy_utils import full_slice
from .._wxutils import Icon, ID, logger, REValidator
from .app import get_app
from .frame import EelbrainFrame, EelbrainDialog
from .mpl_canvas import FigureCanvasPanel
from .history import History, Action


class ChangeAction(Action):
    """Action objects are kept in the history and can do and undo themselves"""

    def __init__(self, desc, index=None, old_accept=None, new_accept=None,
                 old_tag=None, new_tag=None, old_path=None, new_path=None,
                 old_bad_chs=None, new_bad_chs=None, old_interpolate=None,
                 new_interpolate=None):
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
        self.old_tag = old_tag
        self.new_path = new_path
        self.new_accept = new_accept
        self.new_tag = new_tag
        self.old_bad_chs = old_bad_chs
        self.new_bad_chs = new_bad_chs
        self.old_interpolate = old_interpolate
        self.new_interpolate = new_interpolate

    def do(self, doc):
        if self.index is not None:
            doc.set_case(self.index, self.new_accept, self.new_tag,
                         self.new_interpolate)
        if self.new_path is not None:
            doc.set_path(self.new_path)
        if self.new_bad_chs is not None:
            doc.set_bad_channels(self.new_bad_chs)

    def undo(self, doc):
        if self.index is not None:
            doc.set_case(self.index, self.old_accept, self.old_tag,
                         self.old_interpolate)
        if self.new_path is not None and self.old_path is not None:
            doc.set_path(self.old_path)
        if self.new_bad_chs is not None:
            doc.set_bad_channels(self.old_bad_chs)


class Document(object):
    """Represents data for the current state of the Document

    Data can be accesses through attributes, but should only be changed through
    the set_...() methods.

    Attributes
    ----------
    n_epochs : int
        The number of epochs.
    epochs : NDVar
        The raw epochs.
    accept : Var of bool
        Case status.
    tag : Factor
        Case tag.
    trigger : Var of int
        Case trigger value.
    blink : Datalist | None
        Case eye tracker artifact data.
    """

    def __init__(self, ds, data='meg', accept='accept', blink='blink',
                 tag='rej_tag', trigger='trigger', path=None, bad_chs=None,
                 allow_interpolation=True):
        """
        Parameters
        ----------
        ds : Dataset
            Dataset containing
            ...
        path : None | str
            Default location of the epoch selection file (used for save
            command). If the file exists, it is loaded as initial state.
        """
        if isinstance(ds, mne.Epochs):
            epochs = ds
            if not epochs.preload:
                err = ("Need Epochs with preloaded data (preload=True)")
                raise ValueError(err)
            ds = Dataset()
            ds[data] = epochs
            ds['trigger'] = Var(epochs.events[:, 2])

        data = asndvar(data, ds=ds)
        self.n_epochs = n = len(data)

        if not isinstance(accept, basestring):
            raise TypeError("accept needs to be a string")
        if accept not in ds:
            x = np.ones(n, dtype=bool)
            ds[accept] = Var(x)
        accept = ds[accept]

        if not isinstance(tag, basestring):
            raise TypeError("tag needs to be a string")
        if tag in ds:
            tag = ds[tag]
        else:
            tag = Factor([''], repeat=n, name=tag)
            ds.add(tag)

        if not isinstance(trigger, basestring):
            raise TypeError("trigger needs to be a string")
        if trigger in ds:
            trigger = ds[trigger]
        else:
            err = ("ds does not contain a variable named %r. The trigger "
                   "parameters needs to point to a variable in ds containing "
                   "trigger values." % trigger)
            raise KeyError(err)

        if INTERPOLATE_CHANNELS in ds:
            interpolate = ds[INTERPOLATE_CHANNELS]
            if not allow_interpolation and any(interpolate):
                raise ValueError("Dataset contains channel interpolation "
                                 "information but interpolation is turned off")
        else:
            interpolate = Datalist([[]] * ds.n_cases, INTERPOLATE_CHANNELS,
                                   'strlist')

        if isinstance(blink, basestring):
            if ds is not None:
                blink = ds.get(blink, None)
        elif blink is True:
            if 'edf' in ds.info:
                tmin = data.time.tmin
                tmax = data.time.tmax
                _, blink = load.eyelink.artifact_epochs(ds, tmin, tmax,
                                                        esacc=False)
            else:
                msg = ("No eye tracker data was found in ds.info['edf']. Use "
                       "load.eyelink.add_edf(ds) to add an eye tracker file "
                       "to a Dataset ds.")
                wx.MessageBox(msg, "Eye Tracker Data Not Found")
                blink = None
        elif blink is not None:
            raise TypeError("blink needs to be a string or None")

        # options
        self.allow_interpolation = allow_interpolation

        # data
        self.epochs = data
        self.accept = accept
        self.tag = tag
        self.interpolate = interpolate
        self.trigger = trigger
        self.blink = blink
        self.bad_channels = []
        self.good_channels = None

        # cache
        self._good_sensor_indices = {}

        # publisher
        self._bad_chs_change_subscriptions = []
        self._case_change_subscriptions = []
        self._path_change_subscriptions = []

        # finalize
        if bad_chs is not None:
            self.set_bad_channels_by_name(bad_chs)

        self.saved = True  # managed by the history
        self.path = path
        if path and os.path.exists(path):
            accept, tag, interpolate, bad_chs = self.read_rej_file(path)
            self.accept[:] = accept
            self.tag[:] = tag
            self.interpolate[:] = interpolate
            self.set_bad_channels_by_name(bad_chs)

    @property
    def bad_channel_names(self):
        return [self.epochs.sensor.names[i] for i in self.bad_channels]

    def good_data(self):
        "All cases, only good channels"
        if self.bad_channels:
            return self.epochs.sub(sensor=self.good_channels)
        else:
            return self.epochs

    def good_sensor_index(self, case):
        "Index of non-interpolated sensor relative to good sensors"
        if self.interpolate[case]:
            key = frozenset(self.interpolate[case])
            if key in self._good_sensor_indices:
                return self._good_sensor_indices[key]
            else:
                out = np.ones(len(self.epochs.sensor), bool)
                out[self.epochs.sensor.dimindex(self.interpolate[case])] = False
                if self.good_channels is not None:
                    out = out[self.good_channels]
                self._good_sensor_indices[key] = out
                return out

    def get_epoch(self, case, name):
        if self.bad_channels:
            return self.epochs.sub(case=case, sensor=self.good_channels, name=name)
        else:
            return self.epochs.sub(case=case, name=name)

    def get_grand_average(self):
        "Grand average of all accepted epochs"
        return self.epochs.sub(case=self.accept.x, sensor=self.good_channels,
                               name="Grand Average").mean('case')

    def set_bad_channels(self, indexes):
        """Set the channels to treat as bad (i.e., exclude)

        Parameters
        ----------
        bad_chs : collection of int
            Indices of channels to treat as bad.
        """
        indexes = sorted(indexes)
        if indexes == self.bad_channels:
            return
        self.bad_channels = indexes
        if indexes:
            self.good_channels = np.setdiff1d(np.arange(len(self.epochs.sensor)),
                                              indexes, True)
        else:
            self.good_channels = None
        self._good_sensor_indices.clear()
        for callback in self._bad_chs_change_subscriptions:
            callback()

    def set_bad_channels_by_name(self, names):
        self.set_bad_channels(self.epochs.sensor.dimindex(names))

    def set_case(self, index, state, tag, interpolate):
        if state is not None:
            self.accept[index] = state
        if tag is not None:
            self.tag[index] = tag
        if interpolate is not None:
            self.interpolate[index] = interpolate

        for func in self._case_change_subscriptions:
            func(index)

    def set_path(self, path):
        """Set the path

        Parameters
        ----------
        path : str
            Path under which to save. The extension determines the way file
            (*.pickled -> pickled Dataset; *.txt -> tsv)
        """
        root, ext = os.path.splitext(path)
        if ext == '':
            path = root + '.txt'

        self.path = path
        for func in self._path_change_subscriptions:
            func()

    def read_rej_file(self, path):
        "Read a file making sure it is compatible"
        _, ext = os.path.splitext(path)
        if ext == '.pickled':
            ds = load.unpickle(path)
        elif ext == '.txt':
            ds = load.tsv(path, delimiter='\t')
        else:
            raise ValueError("Unknown file extension for rejections: %r" % ext)

        # check file
        if ds.n_cases > self.n_epochs:
            app = get_app()
            cmd = app.message_box("The File contains more events than the data "
                                  "(%i vs %i). Truncate the file?"
                                  % (ds.n_cases, self.n_epochs),
                                  "Truncate the file?", wx.OK | wx.CANCEL |
                                  wx.CANCEL_DEFAULT | wx.ICON_WARNING)
            if cmd == wx.OK:
                ds = ds[:self.n_epochs]
            else:
                raise RuntimeError("Unequal number of cases")
        elif ds.n_cases < self.n_epochs:
            app = get_app()
            cmd = app.message_box("The rejection file contains fewer epochs "
                                  "than the data (%i vs %i). Load anyways "
                                  "(epochs missing from the file will be "
                                  "accepted)?" % (ds.n_cases, self.n_epochs),
                                  "Load partial file?", wx.OK | wx.CANCEL |
                                  wx.CANCEL_DEFAULT | wx.ICON_WARNING)
            if cmd == wx.OK:
                n_missing = self.n_epochs - ds.n_cases
                tail = Dataset(info = ds.info)
                tail['trigger'] = Var(self.trigger[-n_missing:])
                tail['accept'] = Var([True], repeat=n_missing)
                tail['tag'] = Factor(['missing'], repeat=n_missing)
                ds = combine((ds, tail))
            else:
                raise RuntimeError("Unequal number of cases")

        if not np.all(ds[self.trigger.name] == self.trigger):
            app = get_app()
            cmd = app.message_box("The file contains different triggers from "
                                  "the data. Ignore?",
                                  "Ignore trigger mismatch?",
                                  wx.OK | wx.CANCEL | wx.CANCEL_DEFAULT)
            if cmd == wx.OK:
                ds[self.trigger.name] = self.trigger
            else:
                raise RuntimeError("Trigger mismatch")

        accept = ds['accept']
        if 'rej_tag' in ds:
            tag = ds['rej_tag']
        else:
            tag = Factor([''], repeat=self.n_epochs, name='rej_tag')

        if INTERPOLATE_CHANNELS in ds:
            interpolate = ds[INTERPOLATE_CHANNELS]
            if not self.allow_interpolation and any(interpolate):
                app = get_app()
                cmd = app.message_box("The file contains channel interpolation "
                                      "instructions, but interpolation is "
                                      "disabled is the current session. Drop "
                                      "interpolation instructions?",
                                      "Clear Channel Interpolation "
                                      "Instructions?",
                                      wx.OK | wx.CANCEL | wx.CANCEL_DEFAULT)
                if cmd == wx.OK:
                    for l in interpolate:
                        del l[:]
                else:
                    raise RuntimeError("File with interpolation when "
                                       "Interpolation is disabled")
        else:
            interpolate = Datalist([[]] * self.n_epochs, INTERPOLATE_CHANNELS,
                                   'strlist')

        if BAD_CHANNELS in ds.info:
            bad_channels = self.epochs.sensor.dimindex(ds.info[BAD_CHANNELS])
        else:
            bad_channels = []

        return accept, tag, interpolate, bad_channels

    def save(self):
        # find dest path
        _, ext = os.path.splitext(self.path)

        # create Dataset to save
        info = {BAD_CHANNELS: self.bad_channel_names}
        ds = Dataset((self.trigger, self.accept, self.tag, self.interpolate),
                     info=info)

        if ext == '.pickled':
            save.pickle(ds, self.path)
        elif ext == '.txt':
            ds.save_txt(self.path)
        else:
            raise ValueError("Unsupported extension: %r" % ext)

    def subscribe_to_bad_channels_change(self, callback):
        "callback()"
        self._bad_chs_change_subscriptions.append(callback)

    def subscribe_to_case_change(self, callback):
        "callback(index)"
        self._case_change_subscriptions.append(callback)

    def subscribe_to_path_change(self, callback):
        "callback(path)"
        self._path_change_subscriptions.append(callback)



class Model(object):
    """Manages a document as well as its history"""

    def __init__(self, doc):
        self.doc = doc
        self.history = History(doc)

    def auto_reject(self, threshold=2e-12, method='abs', above=False,
                    below=True):
        """
        Marks epochs based on a threshold criterion

        Parameters
        ----------
        threshold : scalar
            The threshold value. Examples: 1.25e-11 to detect saturated
            channels; 2e-12: for conservative MEG rejection.
        method : 'abs' | 'p2p'
            How to apply the threshold. With "abs", the threshold is applied to
            absolute values. With 'p2p' the threshold is applied to
            peak-to-peak values.
        above, below: True, False, None
            How to mark segments that do (above) or do not (below) exceed the
            threshold: True->good; False->bad; None->don't change
        """
        args = ', '.join(map(str, (threshold, method, above, below)))
        logger.info("Auto-reject trials: %s" % args)

        x = self.doc.good_data()
        if method == 'abs':
            x_max = x.abs().max(('time', 'sensor'))
            sub_threshold = x_max <= threshold
        elif method == 'p2p':
            p2p = x.max('time') - x.min('time')
            max_p2p = p2p.max('sensor')
            sub_threshold = max_p2p <= threshold
        else:
            raise ValueError("Invalid method: %r" % method)

        accept = sub_threshold.copy()

        if below is False:
            accept[sub_threshold] = False
        elif below is None:
            accept[sub_threshold] = self.doc.accept.x[sub_threshold]
        elif below is not True:
            err = "below needs to be True, False or None, got %s" % repr(below)
            raise TypeError(err)

        if above is True:
            accept[sub_threshold == False] = True
        elif above is None:
            accept = np.where(sub_threshold, accept, self.doc.accept.x)
        elif above is not False:
            err = "above needs to be True, False or None, got %s" % repr(above)
            raise TypeError(err)

        index = np.where(self.doc.accept.x != accept)[0]
        old_accept = self.doc.accept[index]
        new_accept = accept[index]
        old_tag = self.doc.tag[index]
        new_tag = "%s_%s" % (method, threshold)
        desc = "Threshold-%s" % method
        action = ChangeAction(desc, index, old_accept, new_accept, old_tag,
                              new_tag)
        logger.info("Auto-rejecting %i trials based on threshold %s" %
                    (len(index), method))
        self.history.do(action)

    def clear(self):
        desc = "Clear"
        index = np.logical_not(self.doc.accept.x)
        old_tag = self.doc.tag[index]
        action = ChangeAction(desc, index, False, True, old_tag, 'clear')
        logger.info("Clearing %i rejections" % index.sum())
        self.history.do(action)

    def load(self, path):
        new_accept, new_tag, new_interpolate, new_bad_chs = self.doc.read_rej_file(path)

        # create load action
        action = ChangeAction("Load File", full_slice, self.doc.accept, new_accept,
                              self.doc.tag, new_tag, self.doc.path, path,
                              self.doc.bad_channels, new_bad_chs,
                              self.doc.interpolate, new_interpolate)
        self.history.do(action)
        self.history.register_save()

    def save(self):
        self.doc.save()
        self.history.register_save()

    def save_as(self, path):
        self.doc.set_path(path)
        self.save()

    def set_bad_channels(self, bad_channels, desc="Set bad channels"):
        "Set bad channels with a list of int"
        action = ChangeAction(desc, old_bad_chs=self.doc.bad_channels,
                              new_bad_chs=bad_channels)
        self.history.do(action)

    def set_case(self, i, state, tag=None, desc="Manual Change"):
        old_accept = self.doc.accept[i]
        if tag is None:
            old_tag = None
        else:
            old_tag = self.doc.tag[i]
        action = ChangeAction(desc, i, old_accept, state, old_tag, tag)
        self.history.do(action)

    def set_interpolation(self, case, ch_names):
        action = ChangeAction("Epoch %s interpolate %r" % (case, ', '.join(ch_names)),
                              case, old_interpolate=self.doc.interpolate[case],
                              new_interpolate=ch_names)
        self.history.do(action)

    def toggle_interpolation(self, case, ch_name):
        old_interpolate = self.doc.interpolate[case]
        new_interpolate = old_interpolate[:]
        if ch_name in new_interpolate:
            new_interpolate.remove(ch_name)
            desc = "Don't interpolate %s for %i" % (ch_name, case)
        else:
            new_interpolate.append(ch_name)
            desc = "Interpolate %s for %i" % (ch_name, case)
        action = ChangeAction(desc, case, old_interpolate=old_interpolate,
                              new_interpolate=new_interpolate)
        self.history.do(action)


class Frame(EelbrainFrame):  # control
    "View object of the epoch selection GUI"

    def __init__(self, parent, model, nplots, topo, mean, vlim,
                 plot_range, color, lw, mark, mcolor, mlw, antialiased, pos,
                 size, allow_interpolation):
        """View object of the epoch selection GUI

        Parameters
        ----------
        parent : wx.Frame
            Parent window.
        others :
            See TerminalInterface constructor.
        """
        config = wx.Config("Eelbrain")
        config.SetPath("SelectEpochs")

        if pos is None:
            pos = (config.ReadInt("pos_horizontal", -1),
                   config.ReadInt("pos_vertical", -1))
        else:
            pos_h, pos_v = pos
            config.WriteInt("pos_horizontal", pos_h)
            config.WriteInt("pos_vertical", pos_v)
            config.Flush()

        if size is None:
            size = (config.ReadInt("size_width", 800),
                    config.ReadInt("size_height", 600))
        else:
            w, h = pos
            config.WriteInt("size_width", w)
            config.WriteInt("size_height", h)
            config.Flush()

        super(Frame, self).__init__(parent, -1, "Select Epochs", pos, size)
        doc = model.doc
        history = model.history

        # bind events
        doc.subscribe_to_case_change(self.CaseChanged)
        doc.subscribe_to_path_change(self.UpdateTitle)
        doc.subscribe_to_bad_channels_change(self.ShowPage)
        history.subscribe_to_saved_change(self.UpdateTitle)

        self.config = config
        self.model = model
        self.doc = doc
        self.history = history
        self.allow_interpolation = allow_interpolation

        # setup figure canvas
        self.canvas = FigureCanvasPanel(self)
        self.figure = self.canvas.figure
        self.figure.subplots_adjust(left=.01, right=.99, bottom=.05,
                                    top=.95, hspace=.5)

        # Toolbar
        tb = self.CreateToolBar(wx.TB_HORIZONTAL)
        tb.SetToolBitmapSize(size=(32, 32))
        tb.AddLabelTool(wx.ID_SAVE, "Save",
                        Icon("tango/actions/document-save"), shortHelp="Save")
        tb.AddLabelTool(wx.ID_SAVEAS, "Save As",
                        Icon("tango/actions/document-save-as"),
                        shortHelp="Save As")
        tb.AddLabelTool(wx.ID_OPEN, "Load",
                        Icon("tango/actions/document-open"),
                        shortHelp="Open Rejections")
        tb.AddLabelTool(ID.UNDO, "Undo", Icon("tango/actions/edit-undo"),
                        shortHelp="Undo")
        tb.AddLabelTool(ID.REDO, "Redo", Icon("tango/actions/edit-redo"),
                        shortHelp="Redo")
        tb.AddSeparator()

        # --> select page
        txt = wx.StaticText(tb, -1, "Page:")
        tb.AddControl(txt)
        self.page_choice = wx.Choice(tb, -1)
        tb.AddControl(self.page_choice)
        tb.Bind(wx.EVT_CHOICE, self.OnPageChoice)

        # --> forward / backward
        self.back_button = tb.AddLabelTool(wx.ID_BACKWARD, "Back",
                                           Icon("tango/actions/go-previous"))
        self.next_button = tb.AddLabelTool(wx.ID_FORWARD, "Next",
                                           Icon("tango/actions/go-next"))
        tb.AddSeparator()

        # --> Bad Channels
        button = wx.Button(tb, ID.SET_BAD_CHANNELS, "Bad Channels")
        button.Bind(wx.EVT_BUTTON, self.OnSetBadChannels)
        tb.AddControl(button)

        # --> Thresholding
        button = wx.Button(tb, ID.THRESHOLD, "Threshold")
        button.Bind(wx.EVT_BUTTON, self.OnThreshold)
        tb.AddControl(button)

        # exclude channels
#         btn = wx.Button(tb, ID.EXCLUDE_CHANNELS, "Exclude Channel")
#         tb.AddControl(btn)
#         btn.Bind(wx.EVT_BUTTON, self.OnExcludeChannel)

        # right-most part
        if wx.__version__ >= '2.9':
            tb.AddStretchableSpace()
        else:
            tb.AddSeparator()

#         tb.AddLabelTool(wx.ID_HELP, 'Help', Icon("tango/apps/help-browser"))
#         self.Bind(wx.EVT_TOOL, self.OnHelp, id=wx.ID_HELP)

#         tb.AddLabelTool(ID.FULLSCREEN, "Fullscreen",
#                         Icon("tango/actions/view-fullscreen"))
#         self.Bind(wx.EVT_TOOL, self.OnShowFullScreen, id=ID.FULLSCREEN)

        # Grand-average plot
        button = wx.Button(tb, ID.GRAND_AVERAGE, "GA")
        button.SetHelpText("Plot the grand average of all accepted epochs")
        button.Bind(wx.EVT_BUTTON, self.OnPlotGrandAverage)
        tb.AddControl(button)

        tb.Realize()

        self.CreateStatusBar()

        # setup plot parameters
        self._vlims = find_fig_vlims([[self.doc.epochs]])
        if vlim is not None:
            for k in self._vlims:
                self._vlims[k] = (-vlim, vlim)
        if plot_range is None:
            plot_range = config.ReadBool('plot_range', True)
        self._mark = mark
        self._bfly_kwargs = {'color': color, 'lw': lw, 'mlw': mlw,
                             'antialiased': antialiased, 'vlims': self._vlims,
                             'plot_range': plot_range, 'mcolor': mcolor}
        self._topo_kwargs = {'vlims': self._vlims, 'title': None}
        self._SetLayout(nplots, topo, mean)

        # Bind Events ---
        self.Bind(wx.EVT_CLOSE, self.OnClose)
        self.Bind(wx.EVT_TOOL, self.OnBackward, id=wx.ID_BACKWARD)
        self.Bind(wx.EVT_TOOL, self.OnForward, id=wx.ID_FORWARD)
        self.Bind(wx.EVT_TOOL, self.OnOpen, id=ID.OPEN)
        self.Bind(wx.EVT_TOOL, self.OnSave, id=wx.ID_SAVE)
        self.Bind(wx.EVT_TOOL, self.OnSaveAs, id=wx.ID_SAVEAS)
        self.Bind(wx.EVT_TOOL, self.OnSetLayout, id=ID.SET_LAYOUT)
        self.canvas.mpl_connect('axes_leave_event', self.OnPointerLeaveAxes)
        self.canvas.mpl_connect('button_press_event', self.OnCanvasClick)
        self.canvas.mpl_connect('key_release_event', self.OnCanvasKey)
        self.canvas.mpl_connect('motion_notify_event', self.OnPointerMotion)

        # plot objects
        self._current_page_i = None
        self._epoch_idxs = None
        self._case_plots = None
        self._case_axes = None
        self._case_segs = None
        self._axes_by_idx = None
        self._topo_ax = None

        # Finalize
        self.ShowPage(0)
        self.UpdateTitle()

    def CanBackward(self):
        return self._current_page_i > 0

    def CanForward(self):
        return self._current_page_i < self._n_pages - 1

    def CanRedo(self):
        return self.history.can_redo()

    def CanSave(self):
        return bool(self.doc.path)

    def CanUndo(self):
        return self.history.can_undo()

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

        # update epoch plots
        axes = []
        for idx in index:
            if idx in self._axes_by_idx:
                ax = self._axes_by_idx[idx]
                ax_idx = ax.ax_idx
                h = self._case_plots[ax_idx]
                h.set_state(self.doc.accept[idx])
                # interpolated channels
                ch_index = h.epoch.sensor.channel_idx
                h.set_marked(INTERPOLATE_CHANNELS, [ch_index[ch] for ch in
                                                    self.doc.interpolate[idx]
                                                    if ch in ch_index])

                axes.append(ax)

        # update mean plot
        if self._plot_mean:
            mseg = self._get_page_mean_seg()
            self._mean_plot.set_data(mseg)
            axes.append(self._mean_ax)

        self.canvas.redraw(axes=axes)

    def OnBackward(self, event):
        "turns the page backward"
        self.ShowPage(self._current_page_i - 1)

    def OnCanvasClick(self, event):
        "called by mouse clicks"
        ax = event.inaxes
        if ax:
            logger.debug("Canvas click at ax.ax_idx=%i", ax.ax_idx)
            if ax.ax_idx >= 0:
                idx = ax.epoch_idx
                state = not self.doc.accept[idx]
                tag = "manual"
                desc = "Epoch %i %s" % (idx, state)
                self.model.set_case(idx, state, tag, desc)
            elif ax.ax_idx == -2:
                self.open_topomap()
        else:
            logger.debug("Canvas click outside axes")

    def OnCanvasKey(self, event):
        # GUI Control events
        if event.key == 'right':
            if self.CanForward():
                self.OnForward(None)
            return
        elif event.key == 'left':
            if self.CanBackward():
                self.OnBackward(None)
            return
        elif event.key == 'u':
            if self.CanUndo():
                self.OnUndo(None)
            return
        elif event.key == 'U':
            if self.CanRedo():
                self.OnRedo(None)
            return

        # plotting
        ax = event.inaxes
        if ax is None or ax.ax_idx == -2:
            return
        elif event.key == 't':
            self.PlotTopomap(ax.ax_idx, event.xdata)
        elif event.key == 'b':
            self.PlotButterfly(ax.ax_idx)
        elif event.key == 'c':
            self.PlotCorrelation(ax.ax_idx)
        elif ax.ax_idx == -1:
            return
        elif event.key == 'i':
            self.ToggleChannelInterpolation(ax, event)
        elif event.key == 'I':
            self.OnSetInterpolation(ax.epoch_idx)

    def OnClear(self, event):
        self.model.clear()

    def OnClose(self, event):
        "Ask to save unsaved changes"
        if event.CanVeto() and not self.history.is_saved():
            msg = ("The current document has unsaved changes. Would you like "
                   "to save them?")
            cap = ("Saved Unsaved Changes?")
            style = wx.YES | wx.NO | wx.CANCEL | wx.YES_DEFAULT
            cmd = wx.MessageBox(msg, cap, style)
            if cmd == wx.YES:
                if self.OnSave(event) != wx.ID_OK:
                    event.Veto()
                    return
            elif cmd == wx.CANCEL:
                event.Veto()
                return
            elif cmd != wx.NO:
                raise RuntimeError("Unknown answer: %r" % cmd)

        logger.debug("SelectEpochsFrame.OnClose(), saving window properties...")
        pos_h, pos_v = self.GetPosition()
        w, h = self.GetSize()

        self.config.WriteInt("pos_horizontal", pos_h)
        self.config.WriteInt("pos_vertical", pos_v)
        self.config.WriteInt("size_width", w)
        self.config.WriteInt("size_height", h)
        self.config.WriteBool("plot_range", self._bfly_kwargs['plot_range'])
        self.config.Flush()

        event.Skip()

    def OnForward(self, event):
        "turns the page forward"
        self.ShowPage(self._current_page_i + 1)

    def OnOpen(self, event):
        msg = ("Load the epoch selection from a file.")
        if self.doc.path:
            default_dir, default_name = os.path.split(self.doc.path)
        else:
            default_dir = ''
            default_name = ''
        wildcard = ("Tab Separated Text (*.txt)|*.txt|"
                    "Pickle (*.pickled)|*.pickled")
        dlg = wx.FileDialog(self, msg, default_dir, default_name,
                            wildcard, wx.FD_OPEN)
        rcode = dlg.ShowModal()
        dlg.Destroy()

        if rcode != wx.ID_OK:
            return rcode

        path = dlg.GetPath()
        try:
            self.model.load(path)
        except Exception as ex:
            msg = str(ex)
            title = "Error Loading Rejections"
            wx.MessageBox(msg, title, wx.ICON_ERROR)
            raise


    def OnPageChoice(self, event):
        "called by the page Choice control"
        page = event.GetSelection()
        self.ShowPage(page)

    def OnPlotGrandAverage(self, event):
        self.PlotGrandAverage()

    def OnPointerLeaveAxes(self, event):
        sb = self.GetStatusBar()
        sb.SetStatusText("", 0)

    def OnPointerMotion(self, event):
        "update view on mouse pointer movement"
        ax = event.inaxes
        if not ax:
            self.SetStatusText("")
            return

        # compose status text
        x = ax.xaxis.get_major_formatter().format_data(event.xdata)
        y = ax.yaxis.get_major_formatter().format_data(event.ydata)
        if ax.ax_idx >= 0:  # single trial plot
            self.SetStatusText('Epoch %i,   x = %s, y = %s,   interpolate %s'
                               % (ax.epoch_idx, x, y,
                                  ', '.join(self.doc.interpolate[ax.epoch_idx])))
        elif ax.ax_idx == -1:  # mean plot
            self.SetStatusText("Page average,   x = %s, y = %s" % (x, y))
        else:
            self.SetStatusText('x = %s, y = %s' % (x, y))

        # update topomap
        if self._plot_topo and ax.ax_idx > -2:  # topomap ax_idx is -2
            t = event.xdata
            tseg = self._get_ax_data(ax.ax_idx, t)
            _ax_topomap(self._topo_ax, [tseg], **self._topo_kwargs)
            self.canvas.redraw(axes=[self._topo_ax])

    def OnRedo(self, event):
        self.history.redo()

    def OnSave(self, event):
        if self.doc.path:
            self.model.save()
            return wx.ID_OK
        else:
            return self.OnSaveAs(event)

    def OnSaveAs(self, event):
        msg = ("Save the epoch selection to a file.")
        if self.doc.path:
            default_dir, default_name = os.path.split(self.doc.path)
        else:
            default_dir = ''
            default_name = ''
        wildcard = ("Tab Separated Text (*.txt)|*.txt|"
                    "Pickle (*.pickled)|*.pickled")
        dlg = wx.FileDialog(self, msg, default_dir, default_name, wildcard,
                            wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        rcode = dlg.ShowModal()
        if rcode == wx.ID_OK:
            path = dlg.GetPath()
            self.model.save_as(path)

        dlg.Destroy()
        return rcode

    def OnSetBadChannels(self, event):
        dlg = wx.TextEntryDialog(self, "Please enter bad channel names separated by "
                                 "comma (e.g., \"MEG 003, MEG 010\"):", "Set Bad "
                                 "Channels", ', '.join(self.doc.bad_channel_names))
        while True:
            if dlg.ShowModal() == wx.ID_OK:
                try:
                    names_in = filter(None, (s.strip() for s in
                                             dlg.GetValue().split(',')))
                    names = self.doc.epochs.sensor._normalize_sensor_names(names_in)
                    break
                except ValueError as exception:
                    msg = wx.MessageDialog(self, str(exception), "Invalid Entry",
                                           wx.OK | wx.ICON_ERROR)
                    msg.ShowModal()
                    msg.Destroy()
            else:
                dlg.Destroy()
                return
        dlg.Destroy()
        bad_channels = self.doc.epochs.sensor.dimindex(names)
        self.model.set_bad_channels(bad_channels)

    def OnSetInterpolation(self, epoch):
        "Show Dialog for channel interpolation for this epoch (index)"
        old = self.doc.interpolate[epoch]
        dlg = wx.TextEntryDialog(self, "Please enter channel names separated by "
                                 "comma (e.g., \"MEG 003, MEG 010\"):", "Set "
                                 "Channels for Interpolation", ', '.join(old))
        while True:
            if dlg.ShowModal() == wx.ID_OK:
                try:
                    names = filter(None, (s.strip() for s in dlg.GetValue().split(',')))
                    new = self.doc.epochs.sensor._normalize_sensor_names(names)
                    break
                except ValueError as exception:
                    msg = wx.MessageDialog(self, str(exception), "Invalid Entry",
                                           wx.OK | wx.ICON_ERROR)
                    msg.ShowModal()
                    msg.Destroy()
            else:
                dlg.Destroy()
                return
        dlg.Destroy()
        if new != old:
            self.model.set_interpolation(epoch, new)

    def OnSetLayout(self, event):
        caption = "Set Plot Layout"
        msg = ("Number of epoch plots for square layout (e.g., '10') or \n"
               "exact n_rows and n_columns (e.g., '5 4'). Add 'nomean' to \n"
               "turn off plotting the page mean at the bottom right (e.g., "
               "'3 nomean').")
        default = ""
        dlg = wx.TextEntryDialog(self, msg, caption, default)
        while True:
            if dlg.ShowModal() == wx.ID_OK:
                nplots = None
                topo = True
                mean = True
                err = []

                # separate options from layout
                value = dlg.GetValue()
                items = value.split(' ')
                options = []
                while not items[-1].isdigit():
                    options.append(items.pop(-1))

                # extract options
                for option in options:
                    if option == 'nomean':
                        mean = False
                    elif option == 'notopo':
                        topo = False
                    else:
                        err.append('Unknown option: "%s"' % option)

                # extract layout info
                if len(items) == 1 and items[0].isdigit():
                    nplots = int(items[0])
                elif len(items) == 2 and all(item.isdigit() for item in items):
                    nplots = tuple(int(item) for item in items)
                else:
                    value_ = ' '.join(items)
                    err = 'Invalid layout specification: "%s"' % value_

                # if all ok: break
                if nplots and not err:
                    break

                # error
                caption = 'Invalid Layout Entry: "%s"' % value
                err.append('Please read the instructions and try again.')
                msg = '\n'.join(err)
                style = wx.OK | wx.ICON_ERROR
                wx.MessageBox(msg, caption, style)
            else:
                dlg.Destroy()
                return

        dlg.Destroy()
        self.SetLayout(nplots, topo, mean)

    def OnSetMarkedChannels(self, event):
        "mark is represented in sensor names"
        dlg = wx.TextEntryDialog(self, "Please enter channel names separated by "
                                 "comma (e.g., \"MEG 003, MEG 010\"):", "Set Marked"
                                 "Channels", ', '.join(self._mark))
        while True:
            if dlg.ShowModal() == wx.ID_OK:
                try:
                    names_in = filter(None, (s.strip() for s in
                                             dlg.GetValue().split(',')))
                    names = self.doc.epochs.sensor._normalize_sensor_names(names_in)
                    break
                except ValueError as exception:
                    msg = wx.MessageDialog(self, str(exception), "Invalid Entry",
                                           wx.OK | wx.ICON_ERROR)
                    msg.ShowModal()
                    msg.Destroy()
            else:
                dlg.Destroy()
                return
        dlg.Destroy()
        self.SetPlotStyle(mark=names)

    def OnSetVLim(self, event):
        default = str(self._vlims.values()[0][1])
        dlg = wx.TextEntryDialog(self, "New Y-axis limit:", "Set Y-Axis Limit",
                                 default)

        if dlg.ShowModal() == wx.ID_OK:
            value = dlg.GetValue()
            try:
                vlim = abs(float(value))
            except Exception as exception:
                msg = wx.MessageDialog(self, str(exception), "Invalid Entry",
                                       wx.OK | wx.ICON_ERROR)
                msg.ShowModal()
                msg.Destroy()
                raise
            self.SetVLim(vlim)
        dlg.Destroy()

    def OnThreshold(self, event):
        method = self.config.Read("Threshold/method", "p2p")
        mark_above = self.config.ReadBool("Threshold/mark_above", True)
        mark_below = self.config.ReadBool("Threshold/mark_below", False)
        threshold = self.config.ReadFloat("Threshold/threshold", 2e-12)

        dlg = ThresholdDialog(self, method, mark_above, mark_below, threshold)
        if dlg.ShowModal() == wx.ID_OK:
            threshold = dlg.GetThreshold()
            method = dlg.GetMethod()
            mark_above = dlg.GetMarkAbove()
            if mark_above:
                above = False
            else:
                above = None

            mark_below = dlg.GetMarkBelow()
            if mark_below:
                below = True
            else:
                below = None

            self.model.auto_reject(threshold, method, above, below)

            self.config.Write("Threshold/method", method)
            self.config.WriteBool("Threshold/mark_above", mark_above)
            self.config.WriteBool("Threshold/mark_below", mark_below)
            self.config.WriteFloat("Threshold/threshold", threshold)
            self.config.Flush()

        dlg.Destroy()

    def OnTogglePlotRange(self, event):
        plot_range = event.IsChecked()
        self.SetPlotStyle(plot_range=plot_range)

    def OnUndo(self, event):
        self.history.undo()

    def OnUpdateUIBackward(self, event):
        event.Enable(self.CanBackward())

    def OnUpdateUIForward(self, event):
        event.Enable(self.CanForward())

    def OnUpdateUIOpen(self, event):
        event.Enable(True)

    def OnUpdateUIPlotRange(self, event):
        event.Enable(True)
        event.Check(self._bfly_kwargs['plot_range'])

    def OnUpdateUIRedo(self, event):
        event.Enable(self.CanRedo())

    def OnUpdateUISave(self, event):
        event.Enable(self.CanSave())

    def OnUpdateUISaveAs(self, event):
        event.Enable(self.CanSave())

    def OnUpdateUISetLayout(self, event):
        event.Enable(True)

    def OnUpdateUISetMarkedChannels(self, event):
        event.Enable(True)

    def OnUpdateUISetVLim(self, event):
        event.Enable(True)

    def OnUpdateUIUndo(self, event):
        event.Enable(self.CanUndo())

    def PlotCorrelation(self, ax_index):
        if ax_index == -1:
            seg = self._mean_seg
            name = 'Page Mean Neighbor Correlation'
        else:
            epoch_idx = self._epoch_idxs[ax_index]
            seg = self._case_segs[ax_index]
            name = 'Epoch %i Neighbor Correlation' % epoch_idx
        plot.Topomap(corr(seg, name=name), sensorlabels='name')

    def PlotButterfly(self, ax_index):
        epoch = self._get_ax_data(ax_index)
        plot.TopoButterfly(epoch, vmax=self._vlims)

    def PlotGrandAverage(self):
        epoch = self.doc.get_grand_average()
        plot.TopoButterfly(epoch)

    def PlotTopomap(self, ax_index, time):
        tseg = self._get_ax_data(ax_index, time)
        plot.Topomap(tseg, vmax=self._vlims)

    def SetLayout(self, nplots=(6, 6), topo=True, mean=True):
        """Determine the layout of the Epochs canvas

        Parameters
        ----------
        nplots : int | tuple of 2 int
            Number of epoch plots per page. Can be an ``int`` to produce a
            square layout with that many epochs, or an ``(n_rows, n_columns)``
            tuple.
        topo : bool
            Show a topomap plot of the time point under the mouse cursor.
        mean : bool
            Show a plot of the page mean at the bottom right of the page.
        """
        self._SetLayout(nplots, topo, mean)
        self.ShowPage(0)

    def _SetLayout(self, nplots, topo, mean):
        if topo is None:
            topo = self.config.ReadBool('Layout/show_topo', True)
        else:
            topo = bool(topo)
            self.config.WriteBool('Layout/show_topo', topo)

        if mean is None:
            mean = self.config.ReadBool('Layout/show_mean', True)
        else:
            mean = bool(mean)
            self.config.WriteBool('Layout/show_mean', mean)

        if nplots is None:
            nrow = self.config.ReadInt('Layout/n_rows', 6)
            ncol = self.config.ReadInt('Layout/n_cols', 6)
            nax = ncol * nrow
            n_per_page = nax - bool(topo) - bool(mean)
        else:
            if isinstance(nplots, int):
                if nplots == 1:
                    mean = False
                elif nplots < 1:
                    raise ValueError("nplots needs to be >= 1; got %r" % nplots)
                nax = nplots + bool(mean) + bool(topo)
                nrow = math.ceil(math.sqrt(nax))
                ncol = int(math.ceil(nax / nrow))
                nrow = int(nrow)
                n_per_page = nplots
            else:
                nrow, ncol = nplots
                nax = ncol * nrow
                if nax == 1:
                    mean = False
                    topo = False
                elif nax == 2:
                    mean = False
                elif nax < 1:
                    err = ("nplots=%s: Need at least one plot." % str(nplots))
                    raise ValueError(err)
                n_per_page = nax - bool(topo) - bool(mean)
            self.config.WriteInt('Layout/n_rows', nrow)
            self.config.WriteInt('Layout/n_cols', ncol)
        self.config.Flush()

        self._plot_mean = mean
        self._plot_topo = topo

        # prepare segments
        n = self.doc.n_epochs
        self._nplots = (nrow, ncol)
        self._n_per_page = n_per_page
        self._n_pages = n_pages = int(math.ceil(n / n_per_page))

        # get a list of IDS for each page
        self._segs_by_page = []
        for i in xrange(n_pages):
            start = i * n_per_page
            stop = min((i + 1) * n_per_page, n)
            self._segs_by_page.append(range(start, stop))

        # update page selector
        pages = []
        for i in xrange(n_pages):
            istart = self._segs_by_page[i][0]
            if i == n_pages - 1:
                pages.append('%i: %i..%i' % (i, istart, self.doc.n_epochs))
            else:
                pages.append('%i: %i...' % (i, istart))
        self.page_choice.SetItems(pages)

    def SetPlotStyle(self, **kwargs):
        """Select channels to mark in the butterfly plots.

        Parameters
        ----------
        plot_range : bool
            In the epoch plots, plot the range of the data (instead of plotting
            all sensor traces). This makes drawing of pages quicker, especially
            for data with many sensors (default ``True``).
        color : None | matplotlib color
            Color for primary data (default is black).
        lw : scalar
            Linewidth for normal sensor plots.
        mark : None | str | list of str
            Sensors to plot as individual traces with a separate color.
        mcolor : matplotlib color
            Color for marked traces.
        mlw : scalar
            Line width for marked sensor plots.
        antialiased : bool
            Perform Antialiasing on epoch plots (associated with a minor speed
            cost).
        """
        self._SetPlotStyle(**kwargs)
        self.ShowPage()

    def _SetPlotStyle(self, **kwargs):
        "See .SetPlotStyle()"
        for key, value in kwargs.iteritems():
            if key == 'vlims':
                err = ("%r is an invalid keyword argument for this function"
                       % key)
                raise TypeError(err)
            elif key == 'mark':
                self._mark = value
            elif key == 'plot_range':
                self._bfly_kwargs[key] = value
            elif key in self._bf_kwargs:
                self._bfly_kwargs[key] = value
            else:
                raise KeyError(repr(key))

    def SetVLim(self, vlim):
        """Set the value limits (butterfly plot y axes and topomap colormaps)

        Parameters
        ----------
        vlim : scalar | (scalar, scalar)
            For symmetric limits the positive vmax, for asymmetric limits a
            (vmin, vmax) tuple.
        """
        for p in self._case_plots:
            p.set_ylim(vlim)
        self._mean_plot.set_ylim(vlim)
        if np.isscalar(vlim):
            vlim = (-vlim, vlim)
        for key in self._vlims:
            self._vlims[key] = vlim
        self.canvas.draw()

    def ShowPage(self, page=None):
        "Dislay a specific page (start counting with 0)"
        wx.BeginBusyCursor()
        t0 = time.time()
        if page is None:
            page = self._current_page_i
        else:
            self._current_page_i = page
            self.page_choice.Select(page)

        self.figure.clf()
        nrow, ncol = self._nplots

        # segment plots
        self._epoch_idxs = self._segs_by_page[page]
        self._case_plots = []
        self._case_axes = []
        self._case_segs = []
        self._axes_by_idx = {}
        mark = None
        for i, epoch_idx in enumerate(self._epoch_idxs):
            case = self.doc.get_epoch(epoch_idx, 'Epoch %i' % epoch_idx)
            if mark is None:
                mark = [case.sensor.channel_idx[ch] for ch in self._mark
                        if ch in case.sensor.channel_idx]
            state = self.doc.accept[epoch_idx]
            ax = self.figure.add_subplot(nrow, ncol, i + 1, xticks=[0], yticks=[])
            h = _ax_bfly_epoch(ax, case, mark, state, **self._bfly_kwargs)
            # mark interpolated channels
            if self.doc.interpolate[epoch_idx] and not self._bfly_kwargs['plot_range']:
                chs = [case.sensor.channel_idx[ch]
                       for ch in self.doc.interpolate[epoch_idx]
                       if ch in case.sensor.channel_idx]
                h.set_marked(INTERPOLATE_CHANNELS, chs)
            # mark eye tracker artifacts
            if self.doc.blink is not None:
                _plt_bin_nuts(ax, self.doc.blink[epoch_idx],
                              color=(0.99, 0.76, 0.21))

            ax.ax_idx = i
            ax.epoch_idx = epoch_idx
            self._case_plots.append(h)
            self._case_axes.append(ax)
            self._case_segs.append(case)
            self._axes_by_idx[epoch_idx] = ax

        # mean plot
        if self._plot_mean:
            plot_i = nrow * ncol
            self._mean_ax = self.figure.add_subplot(nrow, ncol, plot_i)
            self._mean_ax.ax_idx = -1
            self._mean_seg = self._get_page_mean_seg()
            self._mean_plot = _ax_bfly_epoch(self._mean_ax, self._mean_seg,
                                             mark, **self._bfly_kwargs)

        # topomap
        if self._plot_topo:
            plot_i = nrow * ncol - self._plot_mean
            self._topo_ax = self.figure.add_subplot(nrow, ncol, plot_i)
            self._topo_ax.ax_idx = -2
            self._topo_ax.set_axis_off()

        self.canvas.draw()
        self.canvas.store_canvas()

        dt = time.time() - t0
        logger.debug('Page draw took %.1f seconds.', dt)
        wx.EndBusyCursor()

    def ToggleChannelInterpolation(self, ax, event):
        if self._bfly_kwargs['plot_range']:
            wx.MessageBox("To assign channels for interpolation turn off "
                          "plotting as range (in the View menu)",
                          "Turn off Range Plotting", wx.OK)
            return
        elif not self.allow_interpolation:
            wx.MessageBox("Interpolation is disabled for this session",
                          "Interpolation disabled", wx.OK)
            return
        plt = self._case_plots[ax.ax_idx]
        locs = plt.epoch.sub(time=event.xdata).x
        sensor = np.argmin(np.abs(locs - event.ydata))
        sensor_name = plt.epoch.sensor.names[sensor]
        self.model.toggle_interpolation(ax.epoch_idx, sensor_name)

    def UpdateTitle(self):
        is_modified = not self.doc.saved

        self.OSXSetModified(is_modified)

        if self.doc.path:
            title = os.path.basename(self.doc.path)
            if is_modified:
                title = '* ' + title
        else:
            title = 'Unsaved'
        self.SetTitle(title)

    def _get_page_mean_seg(self):
        page_segments = self._segs_by_page[self._current_page_i]
        page_index = np.zeros(self.doc.n_epochs, dtype=bool)
        page_index[page_segments] = True
        index = np.logical_and(page_index, self.doc.accept.x)
        mseg = self.doc.get_epoch(index, "Page Average").mean('case')
        return mseg

    def _get_ax_data(self, ax_index, time=None):
        if ax_index == -1:
            seg = self._mean_seg
            epoch_name = 'Page Average'
            sensor_idx = None
        elif ax_index >= 0:
            epoch_idx = self._epoch_idxs[ax_index]
            epoch_name = 'Epoch %i' % epoch_idx
            seg = self._case_segs[ax_index]
            sensor_idx = self.doc.good_sensor_index(epoch_idx)
        else:
            raise ValueError("ax_index needs to be >= -1, not %s" % ax_index)

        if time is not None:
            name = '%s, %.3f s' % (epoch_name, time)
            return seg.sub(time=time, sensor=sensor_idx, name=name)
        elif sensor_idx is None:
            return seg
        else:
            return seg.sub(sensor=sensor_idx)


class TerminalInterface(object):
    """
    Keyboard
    --------
    right-arrow:
        Go to next page.
    left-arrow:
        Go to previous page.
    b:
        Butterfly plot of the currently displayed epoch.
    c:
        Pairwise sensor correlation plot or the current epoch.
    t:
        Topomap plot of the currently displayed time point.
    u:
        Undo.
    shift-u:
        Redo.
    """
    def __init__(self, ds, data='meg', accept='accept', blink='blink',
                 tag='rej_tag', trigger='trigger',
                 path=None, nplots=None, topo=None, mean=None,
                 vlim=None, plot_range=None, color='k', lw=0.5, mark=[],
                 mcolor='r', mlw=0.8, antialiased=True, pos=None, size=None,
                 allow_interpolation=True):
        # Documented in eelbrain.gui
        bad_chs = None
        self.doc = Document(ds, data, accept, blink, tag, trigger, path,
                            bad_chs, allow_interpolation)
        self.model = Model(self.doc)
        self.history = self.model.history

        app = get_app()

        self.frame = Frame(None, self.model, nplots, topo, mean, vlim,
                           plot_range, color, lw, mark, mcolor, mlw,
                           antialiased, pos, size, allow_interpolation)
        self.frame.Show()
        app.SetTopWindow(self.frame)
        if not app.IsMainLoopRunning():
            logger.info("Entering MainLoop for Epoch Selection GUI")
            app.MainLoop()


class ThresholdDialog(EelbrainDialog):

    _methods = (('absolute', 'abs'),
                ('peak-to-peak', 'p2p'))

    def __init__(self, parent, method, mark_above, mark_below, threshold):
        title = "Threshold Criterion Rejection"
        wx.Dialog.__init__(self, parent, wx.ID_ANY, title)
        sizer = wx.BoxSizer(wx.VERTICAL)

        txt = "Mark epochs based on a threshold criterion"
        ctrl = wx.StaticText(self, wx.ID_ANY, txt)
        sizer.Add(ctrl)

        choices = tuple(m[0] for m in self._methods)
        ctrl = wx.RadioBox(self, wx.ID_ANY, "Method", choices=choices)
        selection = [m[1] for m in self._methods].index(method)
        ctrl.SetSelection(selection)
        sizer.Add(ctrl)
        self.method_ctrl = ctrl

        ctrl = wx.CheckBox(self, wx.ID_ANY, "Reject all epochs exceeding the "
                           "threshold")
        ctrl.SetValue(mark_above)
        sizer.Add(ctrl)
        self.mark_above_ctrl = ctrl

        ctrl = wx.CheckBox(self, wx.ID_ANY, "Accept all epochs not exceeding "
                           "the threshold")
        ctrl.SetValue(mark_below)
        sizer.Add(ctrl)
        self.mark_below_ctrl = ctrl

        float_pattern = "^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$"
        msg = ("Invalid entry for threshold: {value}. Need a floating\n"
               "point number.")
        validator = REValidator(float_pattern, msg, False)
        ctrl = wx.TextCtrl(self, wx.ID_ANY, str(threshold),
                           validator=validator)
        ctrl.SetHelpText("Threshold value (positive scalar)")
        ctrl.SelectAll()
        sizer.Add(ctrl)
        self.threshold_ctrl = ctrl

        # buttons
        button_sizer = wx.StdDialogButtonSizer()

        btn = wx.Button(self, wx.ID_OK)
        btn.SetDefault()
        button_sizer.AddButton(btn)

        btn = wx.Button(self, wx.ID_CANCEL)
        button_sizer.AddButton(btn)

        button_sizer.Realize()
        sizer.Add(button_sizer)

        self.SetSizer(sizer)
        sizer.Fit(self)

    def GetMarkAbove(self):
        return self.mark_above_ctrl.IsChecked()

    def GetMarkBelow(self):
        return self.mark_below_ctrl.IsChecked()

    def GetMethod(self):
        index = self.method_ctrl.GetSelection()
        value = self._methods[index][1]
        return value

    def GetThreshold(self):
        text = self.threshold_ctrl.GetValue()
        value = float(text)
        return value
