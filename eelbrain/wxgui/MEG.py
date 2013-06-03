'''
Created on Feb 17, 2012

@author: christian
'''


import logging
import os
import time

import numpy as np
import wx


import ID
from .. import plot
from ..plot._base import eelfigure
from ..plot.utsnd import _ax_bfly_epoch
from ..plot.nuts import _plt_bin_nuts
from .. import load, save, ui
from ..vessels.data import dataset, var, corr
from ..vessels import process
from ..wxutils import mpl_canvas, Icon


__all__ = ['SelectEpochs', 'pca']



class SelectEpochs(eelfigure):
    """
    Interfaces showing individual cases in an ndvar as butterfly plots, with
    the option to interactively manipulate a boolean list (i.e., select cases).

    LMB on any butterfly plot:
        (de-)select this case.
    't' on any butterfly plot:
        Open a topomap for the current time point.
    'c' on any epoch butterfly plot:
        Open a plot with the correlation of each channel with its neighbors.
    """
    def __init__(self, ds, data='MEG', target='accept', blink=True,
                 path=None,
                 nplots=(6, 6), plotsize=(3, 1.5), fill=True, ROI=None,
                 mean=True, topo=True, ylim=None, aa=False, dpi=50):
        """
        Plots all cases in the collection segment and allows visual selection
        of cases. The selection can be retrieved through the get_selection
        Method.

        Parameters
        ----------
        ds : dataset
            dataset on which to perform the selection.
        data : str | ndvar
            Epoch data as case by sensor by time ndvar (or its name in ds).
        target : str | var
            Boolean variable indicating for each epoch whether it is accepted
            (True) or rejected (False). If a str with no corresponding variable
            in ds, a new variable is created.
        blink : bool | str | blink epochs (see load.eyelink)
            Overlay eye tracker blink data on the epoch plots. If
            True and ds contains eyetracker information ('t_edf' variable and
            .info['edf'], as added by ``load.eyelink.events(path, ds=ds)``),
            the blink epochs are extracted automatically. Can also be str (name
            in dataset) or blink epochs directly.
        path : None | str
            Path to the rejection file. If the file already exists, its values
            are read and applied immediately. If the file does not exist, this
            will be the default path for saving the file (valid extensions are
            .txt and .pickled).
        nplots : 1 | 2 | (i, j)
            Number of plots (including topo plots and mean).
        fill : bool
            Only show the range in the butterfly plots, instead of all traces.
            This is faster for data with many channels.
        ROI : None | index for sensor dim
            Sensors to plot as individual traces over the range (ignored if
            range==False).
        mean : bool
            Plot the page mean on each page.
        topo : bool
            Show a dynamically updated topographic plot at the bottom right.
        ylim : scalar
            y-limit of the butterfly plots.
        aa : bool
            Antialiasing (matplolibt parameter).


        Examples
        --------
        >>> SelectEpochs(my_dataset)
        [... visual selection of cases ...]
        >>> cases = my_dataset['reject'] == False
        >>> pruned_dataset = my_dataset[cases]

        """
    # interpret plotting args
        # variable keeping track of selection
        if isinstance(data, basestring):
            data = ds[data]
        self._data = data

        if isinstance(blink, basestring):
            blink = ds.get(blink, None)
        elif (blink is True) and ('edf' in ds.info):
            tmin = data.time.tmin
            tmax = data.time.tmax
            _, blink = load.eyelink.artifact_epochs(ds, tmin=tmin, tmax=tmax,
                                                    esacc=False)
        self._blink = blink

        if np.prod(nplots) == 1:
            nplots = (1, 1)
            mean = False
            topo = False
        elif np.prod(nplots) == 2:
            mean = False
            if nplots == 2:
                nplots = (1, 2)

        if isinstance(target, basestring):
            self._target_name = target
            if target in ds:
                target = ds[target]
            else:
                x = np.ones(ds.n_cases, dtype=bool)
                target = var(x, name=target)
                ds.add(target)
        else:
            self._target_name = target.name
        self._target = target
        self._plot_mean = mean
        self._plot_topo = topo
        self._topo_fig = None
        self._path = path

    # prepare segments
        self._nplots = nplots
        n_per_page = self._n_per_page = np.prod(nplots) - bool(topo) - bool(mean)
        n_pages = ds.n_cases // n_per_page + bool(ds.n_cases % n_per_page)
        self._n_pages = n_pages

        # get a list of IDS for each page
        self._segs_by_page = []
        for i in xrange(n_pages):
            start = i * n_per_page
            stop = min((i + 1) * n_per_page, ds.n_cases)
            self._segs_by_page.append(range(start, stop))

    # init wx frame
        title = "SelectEpochs -> %r" % target.name
        figsize = (plotsize[0] * nplots[0], plotsize[1] * nplots[1])
        super(self.__class__, self).__init__(title=title, figsize=figsize, dpi=dpi)

    # setup figure
        self.figure.subplots_adjust(left=.01, right=.99, bottom=.05, top=.95,
                                    hspace=.5)
        # connect canvas
        self.canvas.mpl_connect('button_press_event', self._on_click)
        self.canvas.mpl_connect('key_release_event', self._on_key)
        if self._is_wx:
            self.canvas.mpl_connect('axes_leave_event', self._on_leave_axes)

    # compile plot kwargs:
        self._bfly_kwargs = {'plot_range': fill, 'plot_traces': ROI}
        if ylim is None:
            ylim = data.properties.get('ylim', None)
        if ylim:
            self._bfly_kwargs['ylim'] = ylim

    # finalize
        self._ds = ds
        if path and os.path.exists(path):
            self._load_selection(path)

        self.show_page(0)
        self._frame.store_canvas()
        self._show()

    def _fill_toolbar(self, tb):
        tb.AddSeparator()

        # --> select page
        txt = wx.StaticText(tb, -1, "Page:")
        tb.AddControl(txt)
        pages = []
        for i in xrange(self._n_pages):
            istart = self._segs_by_page[i][0]
            pages.append('%i: %i...' % (i, istart))
        c = self._page_choice = wx.Choice(tb, -1, choices=pages)
        tb.AddControl(c)
        tb.Bind(wx.EVT_CHOICE, self._OnPageChoice)

        # forward / backward
        tb.AddLabelTool(wx.ID_BACKWARD, "Back", Icon("tango/actions/go-previous"))
        tb.Bind(wx.EVT_TOOL, self._OnBackward, id=wx.ID_BACKWARD)
        tb.AddLabelTool(wx.ID_FORWARD, "Next", Icon("tango/actions/go-next"))
        tb.Bind(wx.EVT_TOOL, self._OnForward, id=wx.ID_FORWARD)
        if self._n_pages < 2:
            tb.EnableTool(wx.ID_FORWARD, False)
            tb.EnableTool(wx.ID_BACKWARD, False)
        tb.AddLabelTool(wx.ID_REFRESH, "Refresh", Icon("tango/actions/view-refresh"))
        tb.Bind(wx.EVT_TOOL, self._Refresh, id=wx.ID_REFRESH)
        tb.AddSeparator()

        # Thresholding
        btn = wx.Button(tb, ID.THRESHOLD, "Threshold")
        tb.AddControl(btn)
        btn.Bind(wx.EVT_BUTTON, self._OnThreshold)

        # save rejection
        btn = wx.Button(tb, ID.SAVE_REJECTION, "Save")
        btn.SetHelpText("Save the epoch selection to a file.")
        tb.AddControl(btn)
        btn.Bind(wx.EVT_BUTTON, self._OnSaveRejection)

    def _get_page_mean_seg(self):
        seg_IDs = self._segs_by_page[self._current_page_i]
        index = np.zeros(self._ds.n_cases, dtype=bool)
        index[seg_IDs] = True
        index *= self._target
        mseg = self._data[index].summary()
        return mseg

    def _get_statusbar_text(self, event):
        "called by parent class to get figure-specific status bar text"
        ax = event.inaxes
        if ax and (ax.ID > -2):  # topomap ID is -2
            t = event.xdata
            tseg = self._get_topo_seg(ax, t)
            if ax.ID >= 0:  # single trial plot
                txt = 'Segment %i,   ' % ax.segID + '%s'
            elif  ax.ID == -1:  # mean plot
                txt = "Page average,   %s"
            # update internal topomap plot
            plot.topo._ax_topomap(self._topo_ax, [tseg])
            self._frame.redraw(axes=[self._topo_ax])
            # update external topomap plot
            if self._topo_fig:
                pass

            return txt
        else:
            return '%s'

    def _get_topo_seg(self, ax, t):
        name = '%%s, %.3f s' % t
        ax_id = ax.ID
        if ax_id == -1:
            tseg = self._mean_seg.subdata(time=t, name=name % 'Page Average')
        elif ax_id >= 0:
            seg = self._case_segs[ax_id]
            tseg = seg.subdata(time=t, name=name % 'Segment %i' % ax.segID)
        else:
            raise IndexError("ax_id needs to be >= -1, not %i" % ax_id)

        self._tseg = tseg
        return tseg

    def _update_mean(self):
        mseg = self._get_page_mean_seg()
        self._mean_handle.update_data(mseg)
        self._frame.redraw(axes=[self._mean_ax])

    def set_ax_state(self, axID, state):
        ax = self._case_axes[axID]
        h = self._case_handles[axID]
        h.set_state(state)
        ax._epoch_state = state

        self._frame.redraw(artists=[h.ax])
        self._update_mean()

    def invert_selection(self, axID):
        "ID refers to ax-ID in the display"
        # find current selection
        ax = self._case_axes[axID]
        epochID = ax.segID
        state = not self._target[epochID]
        self._target[epochID] = state

        # update plot
        self.set_ax_state(axID, state)

    def load_selection(self, path):
        self._load_selection(path)
        self._Refresh(None)

    def _load_selection(self, path):
        _, ext = os.path.splitext(path)
        if ext == '.pickled':
            ds = load.unpickle(path)
            if np.all(ds['eventID'] == self._ds['eventID']):
                self._target[:] = ds['accept']
            else:
                err = ("Event IDs don't match")
                raise ValueError(err)
        elif ext == '.txt':
            self._target[:] = load.txt.var(path)
        else:
            raise ValueError("Unknown file extension for rejections: %r" % ext)

    def open_topomap(self):
        if self._topo_fig:
            pass
        else:
            fig = plot.topo.topomap(self._tseg, sensors='name')
            self._topo_fig = fig

    def save_rejection(self, path):
        """Save the rejection list as a file

        path : str
            The extension def

        """
        # find dest path
        root, ext = os.path.splitext(path)
        if ext == '':
            ext = '.pickled'
        path = root + ext

        # create dataset to save
        accept = self._target
        if accept.name != 'accept':
            accept = accept.copy('accept')
        ds = dataset(self._ds['eventID'], accept)

        if ext == '.pickled':
            save.pickle(ds, path)
        elif ext == '.txt':
            ds.export(path, fmt='%s')
        else:
            raise ValueError("Unsupported extension: %r" % ext)

    def set_ylim(self, y):
        for ax in self._case_axes:
            ax.set_ylim(-y, y)
        self._mean_ax.set_ylim(-y, y)
        self.canvas.draw()
        self._bfly_kwargs['ylim'] = y

    def show_page(self, page):
        "Dislay a specific page (start counting with 0)"
        t0 = time.time()
        self._current_page_i = page
        self._page_choice.Select(page)

        self.figure.clf()
        nx, ny = self._nplots
        seg_IDs = self._segs_by_page[page]

        # segment plots
        self._case_handles = []
        self._case_axes = []
        self._case_segs = []
        for i, ID in enumerate(seg_IDs):
            case = self._data[ID]
            state = self._target[ID]
            ax = self.figure.add_subplot(nx, ny, i + 1, xticks=[0], yticks=[])  # , 'axis_off')
            ax._epoch_state = state
#            ax.set_axis_off()
            h = _ax_bfly_epoch(ax, case, xlabel=None, ylabel=None, state=state,
                               **self._bfly_kwargs)
            if self._blink is not None:
                _plt_bin_nuts(ax, self._blink[ID])

            ax.ID = i
            ax.segID = ID
            self._case_handles.append(h)
            self._case_axes.append(ax)
            self._case_segs.append(case)


        # mean plot
        if self._plot_mean:
            ax = self._mean_ax = self.figure.add_subplot(nx, ny, nx * ny)
            ax.ID = -1

            mseg = self._mean_seg = self._get_page_mean_seg()
            self._mean_handle = _ax_bfly_epoch(ax, mseg, **self._bfly_kwargs)

        # topomap
        if self._plot_topo:
            ax = self._topo_ax = self.figure.add_subplot(nx, ny, nx * ny - self._plot_mean)
            ax.ID = -2
            ax.set_axis_off()

        self.canvas.draw()
        dt = time.time() - t0
        logging.debug('Page draw took %.1f seconds.', dt)

    def _OnBackward(self, event):
        "turns the page backward"
        if self._current_page_i == 0:
            self.show_page(self._n_pages - 1)
        else:
            self.show_page(self._current_page_i - 1)

    def _on_click(self, event):
        "called by mouse clicks"
        logging.debug('click: ')
        ax = event.inaxes
        if ax:
            if ax.ID >= 0:
                self.invert_selection(ax.ID)
            elif ax.ID == -2:
                self.open_topomap()

    def _on_key(self, event):
        ax = event.inaxes
        ax_id = getattr(ax, 'ID', None)
        if (event.key == 't'):
            if (ax_id < 0) and (ax_id != -2):
                return
            tseg = self._get_topo_seg(ax, t=event.xdata)
            plot.topo.topomap(tseg, sensors='name')
        elif (event.key == 'c'):
            if (ax_id < 0) and (ax_id != -2):
                return
            seg = self._case_segs[ax_id]
            cseg = corr(seg, name='Epoch %i Neighbor Correlation' % ax.segID)
            plot.topo.topomap(cseg)

    def _OnForward(self, event):
        "turns the page forward"
        if self._current_page_i < self._n_pages - 1:
            self.show_page(self._current_page_i + 1)
        else:
            self.show_page(0)

    def _on_leave_axes(self, event):
        sb = self.GetStatusBar()
        sb.SetStatusText("", 0)

    def _OnPageChoice(self, event):
        "called by the page Choice control"
        page = self._page_choice.GetSelection()
        self.show_page(page)

    def _OnSaveRejection(self, event):
        msg = ("Save the epoch selection to a file.")
        if self._path:
            default_dir, default_name = os.path.split(self._path)
        else:
            default_dir = ''
            default_name = ''
        wildcard = "Pickle (*.pickled)|*.pickled|Text (*.txt)|*.txt"
        dlg = wx.FileDialog(self._frame, msg, default_dir, default_name,
                            wildcard, wx.FD_SAVE)
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            self.save_rejection(path)

    def _Refresh(self, event):
        "updates the states of the segments on the current page"
        for ax in self._case_axes:
            state = self._target[ax.segID]
            if state != ax._epoch_state:
                self.set_ax_state(ax.ID, state)

    def save_all_pages(self, fname=None):
        if fname is None:
            msg = ("Save all pages. Index is inserted before extension. "
                   "If no extension is provided, pdf is used.")
            fname = ui.ask_saveas("title", msg, None)
            if not fname:
                return

        head, tail = os.path.split(fname)
        if os.path.extsep in tail:
            root, ext = os.path.splitext(tail)
        else:
            root = fname
            ext = '.pdf'
        fntemp = os.path.join(head, root) + '_%i' + ext

        i = self.current_page_i
        self.figure.savefig(fntemp % i)
        pages = range(self.n)
        pages.remove(i)
        prog = ui.progress_monitor(self.n - 1, "Saving Figures", "Saving Figures")
        for i in pages:
            self.show_page(i)
            self.figure.savefig(fntemp % i)
            prog.advance()
        prog.terminate()

    def _OnThreshold(self, event):
        threshold = None
        above = False
        below = True

        msg = "What value should be used to threshold the data?"
        while threshold is None:
            dlg = wx.TextEntryDialog(self._frame, msg, "Choose Threshold", "2e-12")
            if dlg.ShowModal() == wx.ID_OK:
                value = dlg.GetValue()
                try:
                    threshold = float(value)
                except ValueError:
                    ui.message("Invalid Entry", "%r is not a valid entry. Need "
                               "a float." % value, '!')
            else:
                return

        process.mark_by_threshold(self._ds, DV=self._data,
                                  threshold=threshold, above=above,
                                  below=below, target=self._target)

        self._Refresh(event)



class pca(mpl_canvas.CanvasFrame):
    def __init__(self, dataset, Y='MEG', nplots=(7, 10), dpi=50, figsize=(20, 12)):
        """
        Performs PCA and opens a GUI for removing individual components.

        Y : ndvar | str
            dependent variable
        timecourse : int
            number of (randomly picked) segments for whichthe time-course is
            displayed

        """
        if isinstance(Y, basestring):
            Y = dataset[Y]

        self._dataset = dataset
        self._Y = Y

    # prepare plots:
        self._topo_kwargs = {}

    # wx stuff
        parent = wx.GetApp().shell
        title = "PCA of %r" % Y.name
        mpl_canvas.CanvasFrame.__init__(self, parent, title, statusbar=False,
                                        figsize=figsize, dpi=dpi)
        # connect
        self.canvas.mpl_connect('button_press_event', self.OnClick)
        # figure
        self.figure.subplots_adjust(left=.01, right=.99, bottom=.05, top=.95,
                                    hspace=.2)

    # do the PCA
        pca = self.pca = process.PCA(Y)

    # plot the components
        self._components = []
        self._rm_comp = []
        npy, npx = nplots
        title_temp = '%i'
        for i in xrange(np.prod(nplots)):
            name = title_temp % i
            comp = pca.get_component(i)
            ax = self.figure.add_subplot(npy, npx, i + 1, xticks=[], yticks=[])
            ax.Id = i
            plot.topo._ax_topomap(ax, [comp])
            ax.set_title(name)
            ax.set_frame_on(1)
            ax.set_axis_on()
            self._components.append(comp)

    # finalize
        self.canvas.store_canvas()
        self.Show()

    def _init_FillToolBar(self, tb):
        tb.AddSeparator()

        # remove
        btn = wx.Button(tb, ID.PCA_REMOVE, "Remove Selected Components")
        tb.AddControl(btn)
        self.Bind(wx.EVT_BUTTON, self.OnRemove, id=ID.PCA_REMOVE)

        mpl_canvas.CanvasFrame._init_FillToolBar(self, tb)

    def OnClick(self, event):
        ax = event.inaxes
        if ax:
            Id = ax.Id
            if Id in self._rm_comp:
                self._rm_comp.remove(Id)
                ax.set_axis_bgcolor('white')
            else:
                self._rm_comp.append(Id)
                ax.set_axis_bgcolor('r')
            self.canvas.redraw(axes=[ax])

    def OnRemove(self, event):
        target = None
        rm = sorted(self._rm_comp)
        while not target:
            dlg = wx.TextEntryDialog(self, "What name should the new ndvar be assigned in the dataset?",
                                     "Choose Name for New Variable", "%s" % self._Y.name)
            if dlg.ShowModal() == wx.ID_OK:
                newname = str(dlg.GetValue())
                if newname in self._dataset:
                    msg = ("The dataset already contains an item named %r. "
                           "Should it be replaced? The item is:\n\n%r" %
                            (newname, self._dataset[newname]))
                    answer = ui.ask("Replace %r?" % newname, msg)
                    if answer is True:
                        target = newname
                    elif answer is None:
                        pass
                    else:
                        return
                else:
                    target = newname
            else:
                return

        # if we made it down here, remove the component:
        self._dataset[target] = self.pca.subtract(rm, name=target)

        self.Close()




