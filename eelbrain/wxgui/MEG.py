'''
Created on Feb 17, 2012

@author: christian
'''


import logging
import os

import numpy as np
import wx
import mdp

from eelbrain import plot
from eelbrain.vessels import data as _data
from eelbrain.vessels import process

import ID
from eelbrain import ui
from eelbrain.wxutils import mpl_canvas
from eelbrain.wxutils import Icon


__all__ = ['select_cases_butterfly', 'pca']



class select_cases_butterfly(mpl_canvas.CanvasFrame):
    def __init__(self, dataset, data='MEG', target='reject', nplots=(6,6), plotsize=(3,1.5),
                 mean=True, topo=True, ylim=None, aa=False, dpi=50):
        """
        Plots all cases in the collection segment and allows visual selection
        of cases. The selection can be retrieved through the get_selection 
        Method.  
        
        Arguments
        ----------
        
        dataset : dataset
            dataset on which to perform the selection.
        
        Example::
        
            >>> p = select_cases_butterfly(my_dataset)
            [... visual selection of cases ...]
            >>> cases = p.get_selection()
            >>> pruned_egment = my_segment.subdata(cases)
            
        """
    # interpret plotting args
        # variable keeping track of selection
        if isinstance(data, basestring):
            data = dataset[data]
        self._data = data
        
        if isinstance(target, basestring):
            try:
                target = dataset[target]
            except KeyError:
                x = np.zeros(dataset.N, dtype=bool)
                target = _data.var(x, name=target)
                dataset.add(target)
        self._target = target
        
    # prepare segments
        self._nplots = nplots
        n_per_page = self._n_per_page = np.prod(nplots) - 2
        n_pages = dataset.N // n_per_page + bool(dataset.N % n_per_page)
        self._n_pages = n_pages
        
        # get a list of IDS for each page
        self._segs_by_page = []
        for i in xrange(n_pages):
            start = i * n_per_page
            stop = min((i+1) * n_per_page, dataset.N)
            self._segs_by_page.append(range(start, stop))
        
    # init wx frame
        parent = wx.GetApp().shell
        title = "select_cases_butterfly -> %r" % target.name
        figsize = (plotsize[0] * nplots[0], plotsize[1] * nplots[1])
        mpl_canvas.CanvasFrame.__init__(self, parent, title, figsize=figsize, 
                                        dpi=dpi)
        # connect
        self.canvas.mpl_connect('button_press_event', self.OnClick)
        self.canvas.mpl_connect('axes_leave_event', self.OnLeaveAxes)
        # figure
        self.figure.subplots_adjust(left=.01, right=.99, bottom=.05, top=.95, 
                                    hspace=.5)
                
    # compile plot kwargs:
        self._bfly_kwargs = {'extrema': True}
        if ylim is None:
            ylim = data.properties.get('ylim', None)
        if ylim:
            self._bfly_kwargs['ylim'] = ylim
        
    # finalize
        self._dataset = dataset
        self.show_page(0)
        self.canvas.store_canvas()
        self.Show()
    
    def _init_FillToolBar(self, tb):
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
        tb.Bind(wx.EVT_CHOICE, self.OnPageChoice)
        
        # forward / backward
        tb.AddLabelTool(wx.ID_BACKWARD, "Back", Icon("tango/actions/go-previous"))
        self.Bind(wx.EVT_TOOL, self.OnBackward, id=wx.ID_BACKWARD)
        tb.AddLabelTool(wx.ID_FORWARD, "Next", Icon("tango/actions/go-next"))
        self.Bind(wx.EVT_TOOL, self.OnForward, id=wx.ID_FORWARD)
        if self._n_pages < 2:
            tb.EnableTool(wx.ID_FORWARD, False)
            tb.EnableTool(wx.ID_BACKWARD, False)
        tb.AddLabelTool(wx.ID_REFRESH, "Refresh", Icon("tango/actions/view-refresh"))
        self.Bind(wx.EVT_TOOL, self.OnRefresh, id=wx.ID_REFRESH)
        tb.AddSeparator()
        
        # Thresholding
        btn = wx.Button(tb, ID.THRESHOLD, "Threshold")
        tb.AddControl(btn)
        self.Bind(wx.EVT_BUTTON, self.OnThreshold, id=ID.THRESHOLD)
        
        mpl_canvas.CanvasFrame._init_FillToolBar(self, tb)
    
    def _get_page_mean_seg(self):
        seg_IDs = self._segs_by_page[self._current_page_i]
        index = np.zeros(self._dataset.N, dtype=bool)
        index[seg_IDs] = True
        index *= (self._target == False)
        mseg = self._data[index].mean()
        return mseg 
    
    def _update_mean(self):
        mseg = self._get_page_mean_seg()
        data = mseg.get_data(('time', 'sensor'))[0]
        T = mseg.time
        T_len = len(T)
        Ylen = T_len * 2 + 1
        Y = np.empty((Ylen, 2))
        # data
        Y[:T_len, 1] = data.min(1)
        Y[2*T_len:T_len:-1, 1] = data.max(1)
        # T
        Y[:T_len, 0] = T
        Y[2*T_len:T_len:-1, 0] = T
        # border regions
        Y[T_len,:] = Y[T_len+1,:]
        self._mean_handle[0].set_paths([Y])
        
        # update figure
        self.canvas.redraw_ax(self._mean_ax)
    
    def set_ax_state(self, axID, state):
        ax = self._case_axes[axID]
        h = self._case_handles[axID]
        if state:
            h.set_facecolors('r')
        else:
            h.set_facecolors('k')
        ax._epoch_state = state
        
        self.canvas.redraw_ax(ax)
#        self.canvas_panel.redraw_artist(ax, h)
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
    
    def show_page(self, page):
        "Dislay a specific page (start counting with 0)"
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
            case = self._data.get_epoch(ID)
            state = self._target[ID]
            if state:
                color = 'r'
            else:
                color = 'k'
            ax = self.figure.add_subplot(nx, ny, i+1, xticks=[0], yticks=[])#, 'axis_off')
            ax._epoch_state = state
#            ax.set_axis_off()
            h = plot.uts._ax_butterfly(ax, case, color=color, antialiased=False,
                                       title=False, xlabel=None, ylabel=None,
                                       **self._bfly_kwargs)[0]
            ax.ID = i
            ax.segID = ID
            self._case_handles.append(h)
            self._case_axes.append(ax)
            self._case_segs.append(case)
        
        
        # mean plot
        ax = self._mean_ax = self.figure.add_subplot(nx, ny, nx*ny)
        ax.ID = -1
        
        mseg = self._mean_seg = self._get_page_mean_seg()
        self._mean_handle = plot.uts._ax_butterfly(ax, mseg, color='k', **self._bfly_kwargs)
        
        # topomap
        ax = self._topo_ax = self.figure.add_subplot(nx, ny, nx*ny - 1)
        ax.ID = -2
        ax.set_axis_off()
        
        self.canvas.draw()
    
    def OnBackward(self, event):
        "turns the page forward"
        if self._current_page_i == 0:
            self.show_page(self._n_pages - 1)
        else:
            self.show_page(self._current_page_i - 1)
    
    def OnClick(self, event):
        "called by mouse clicks"
        logging.debug('click: ')
        ax = event.inaxes
        if ax and ax.ID >= 0:
            self.invert_selection(ax.ID)

    def OnForward(self, event):
        "turns the page backwards"
        if self._current_page_i < self._n_pages - 1:
            self.show_page(self._current_page_i + 1)
        else:
            self.show_page(0)
    
    def OnLeaveAxes(self, event):
        sb = self.GetStatusBar()
        sb.SetStatusText("", 0)
    
    def OnMotionStatusBarUpdate(self, event):
        """
        called for mouse motion
        http://matplotlib.sourceforge.net/examples/user_interfaces/wxcursor_demo.html
        
        """
        ax = event.inaxes
        if ax:
            t = event.xdata
            if ax.ID >= 0:
                seg = self._case_segs[ax.ID]
                tseg = seg.subdata(time=t)
                name = 'Segment %i' % ax.segID
            elif  ax.ID == -1:
                tseg = self._mean_seg.subdata(time=t)
                name = "Page average"
            elif ax.ID == -2:
                return
            # update status bar
            sb = self.GetStatusBar()
            txt = "%s, t = %.3f s" % (name, t)
            sb.SetStatusText(txt, 0)
            # update plot
            plot.topo._ax_topomap(self._topo_ax, tseg)
            self.canvas.redraw_ax(self._topo_ax)
    
    def OnPageChoice(self, event):
        "called by the page Choice control"
        page = self._page_choice.GetSelection()
        self.show_page(page)
    
    def OnRefresh(self, event):
        "updates the states of the segments on the current page"
        for ax in self._case_axes:
            state = self._target[ax.segID]
            if state != ax._epoch_state:
                self.set_ax_state(ax.ID, state)
    
    def OnFileSave(self, event):
        dialog = wx.FileDialog(self, "Save Figure (If no extension is provided, pdf is used).", 
                               style=wx.FD_SAVE|wx.FD_OVERWRITE_PROMPT)
#                               wildcard="Current Page|*.pdf|All Pages (*_n.*)|*.gif")
        if dialog.ShowModal() == wx.ID_OK:
            path = dialog.GetPath()
#            wc = dialog.GetFilterIndex()
            wc = 0
            path, filename = os.path.split(path)
            if wc == 0: # save only current page
                if not '.' in filename:
                    filename += '.pdf'
                self.figure.savefig(os.path.join(path, filename))
            elif wc == 1: # save all pages
                fn_temp = os.path.join('{dir}', '{root}_{n}.{ext}')
                if '.' in filename:
                    filename, ext = filename.split('.')
                else:
                    ext = 'pdf'
                i = self.current_page_i
                self.figure.savefig(fn_temp.format(dir=path, root=filename, n=i, ext=ext))
                pages = range(self.n)
                pages.remove(i)
                prog = ui.progress(self.n-1, "Saving Figures", "Saving Figures")
                for i in pages:
                    self.show_page(i)
                    self.figure.savefig(fn_temp.format(dir=path, root=filename, n=i, ext=ext))
                    prog.advance()
                prog.terminate()
            else:
                logging.error(" invalid wildcard: %s"%wc)
    
    def OnThreshold(self, event):
        """
        above: True, False, None
            how to mark segments that exceed the threshold: True->good; 
            False->bad; None->don't change
        below:
            same as ``above`` but for segments that do not exceed the threshold
        
        """
        threshold = None
        above = True
        below = False
        
        msg = "What value should be used to threshold the data?"
        while threshold is None:
            dlg = wx.TextEntryDialog(self, msg, "Choose Threshold", "2e-12")
            if dlg.ShowModal() == wx.ID_OK:
                value = dlg.GetValue()
                try:
                    threshold = float(value)
                except ValueError:
                    ui.message("Invalid Entry", "%r is not a valid entry. Need "
                               "a float." % value, '!')
            else:
                return
        
        process.mark_by_threshold(self._dataset, DV=self._data, 
                                  threshold=threshold, above=above, 
                                  below=below, target=self._target)
        
        self.OnRefresh(event)
    


class pca(mpl_canvas.CanvasFrame):
    def __init__(self, dataset, Y='MEG', nplots=(7,10), dpi=50, figsize=(20,12)):
        if isinstance(Y, basestring):
            Y = dataset[Y]
        
        self._dataset = dataset
        self._Y = Y
        
    # prepare plots:
        self._nplots = nplots
        self._ncomp = np.prod(nplots)
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
        npy, npx = self._nplots
        title_temp = '%i'
        for i in xrange(self._ncomp):
            name = title_temp % i
            comp = pca.get_component(i)
            ax = self.figure.add_subplot(npy, npx, i+1, xticks=[], yticks=[])
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
            self.canvas.redraw_ax(ax)
    
    def OnRemove(self, event):
        target = None
        rm = sorted(self._rm_comp)
        while not target:
            dlg = wx.TextEntryDialog(self, "What name should the new ndvar be assigned in the dataset?",
                                     "Choose Name for New Variable", "%s"%self._Y.name)
            if dlg.ShowModal() == wx.ID_OK:
                newname = str(dlg.GetValue())
                if newname in self._dataset:
                    msg = ("The dataset already contains an item named %r. "
                           "Should it be replaced? The item is:\n\n%r" %
                            (newname, self._dataset[newname]))
                    answer = ui.ask("Replace %r?"%newname, msg)
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



        
