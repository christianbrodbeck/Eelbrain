# from copy import deepcopy
import os, logging

import wx

from eelbrain.wxutils import mpl_canvas


from eelbrain import ui
from eelbrain.wxutils import Icon
from ...utils.basic import toList



class ListViewerFrame(mpl_canvas.CanvasFrame):
    """

    methods for user:

    .set_zoom    change the extent of the data displayed (using event address)

    """
    def __init__(self, visualizers, id= -1,
                 x=1, y=5, texsave=True,
                 zi=None, za=None, zp=10,  # zoom (index, address, pad
                 size=(16, 8), dpi=50,
                 ):
        visualizers = self.visualizers = toList(visualizers)
#        self.visualizers.reverse()

        title = "List Viewer"
        self.texsave = texsave

        N = visualizers[0].N
        if not all([v.N == N for v in visualizers[1:]]):
            raise ValueError("visualizers do not provide the same N of segments")

        # distribute segments into pages:
        y = min((y, N))
        self.shape = x, y
        n_per_page = self.n_per_page = x * y
        i_lists = []
        for i in range(0, N, n_per_page):
            i_lists.append(range(i, i + min(n_per_page, N - i)))
        self.i_lists = i_lists
        self.n = len(i_lists)



        self._zoom_events = zi
        self._zoom_address = za
        self._zoom_pad = zp
        self._zoom_window = None

    # init wx frame
        parent = wx.GetApp().shell
        title = "List Viewer"
        mpl_canvas.CanvasFrame.__init__(self, parent, title, size, dpi,
                                        statusbar=True)
        # figure
        self.figure.subplots_adjust(left=.01, right=.99, bottom=.05, top=.95,
                                 hspace=.5)

    # finalize
#        self._dataset = dataset
        self.show_page(0)
#        self.canvas.store_canvas()
        self.Show()

    def _init_FillToolBar(self, tb):
        # --> select page
        txt = wx.StaticText(tb, -1, "Page:")
        tb.AddControl(txt)
        pages = [str(i) for i in range(self.n)]
        c = self.page_choice = wx.Choice(tb, -1, choices=pages)
        tb.AddControl(c)
        tb.Bind(wx.EVT_CHOICE, self.OnPageChoice)

        tb.AddLabelTool(wx.ID_BACKWARD, "Back", Icon("tango/actions/go-previous"))
        self.Bind(wx.EVT_TOOL, self.OnBackward, id=wx.ID_BACKWARD)
        tb.AddLabelTool(wx.ID_FORWARD, "Next", Icon("tango/actions/go-next"))
        self.Bind(wx.EVT_TOOL, self.OnForward, id=wx.ID_FORWARD)
        if self.n < 2:
            tb.EnableTool(wx.ID_FORWARD, False)
            tb.EnableTool(wx.ID_BACKWARD, False)
        tb.AddLabelTool(wx.ID_REFRESH, "Refresh", Icon("tango/actions/view-refresh"))
        self.Bind(wx.EVT_TOOL, self.OnRefresh, id=wx.ID_REFRESH)


        # --> Gauge
#        gauge = self.gauge = wx.Gauge(tb, -1, size=(50, -1))
#        tb.AddControl(gauge)

        # --> options
        """
        tb.AddSeparator()
        checkbox = wx.CheckBox(tb, wx_base.ID_SHOW_SOURCE, "Show Source")
        tb.AddControl(checkbox)
        self.Bind(wx.EVT_CHECKBOX, self.OnCheckBox)

        tb.AddLabelTool(wx.ID_ZOOM_IN, "Zoom In", Icon("mine.zoom_in"))
        tb.AddLabelTool(wx.ID_ZOOM_OUT, "Zoom Out", Icon("mine.zoom_out"))
        self.Bind(wx.EVT_TOOL, self.OnZoom, id=wx.ID_ZOOM_IN)
        self.Bind(wx.EVT_TOOL, self.OnZoom, id=wx.ID_ZOOM_OUT)
        """

        mpl_canvas.CanvasFrame._init_FillToolBar(self, tb)

    def __repr__(self):
        temp = "ListViewerFrame({Vs})"
        Vs = ', '.join(v.__repr__() for v in self.visualizers)
        return temp.format(Vs=Vs)

    def set_window(self, tstart, tend):
        """
        Set the zoom based on time points

        **See also:** .set_zoom()

        """
        self._zoom_window = (tstart, tend)
        self._zoom_events = None
        self.OnRefresh(None)

    def set_zoom(self, events_ds=None, address=None, pad=10):
        """
        Set the time range of the data that is displayed based on events.

        events_ds :
            index of vizualizer/dataset containing events (None to
            reset zoom and show all data)
        address :
            event-address of events to display
        pad: scalar
            time to display outside of events


        **See also:** .set_window()

        """
        if events_ds is not None:
            v = self.visualizers[events_ds]
            ds = v.dataset
            if ds.properties['data_type'] != 'event':
                ui.message("'%s' is not an event dataset!" % ds.name, "", '!')
                return
        self._zoom_events = events_ds
        self._zoom_address = address
        self._zoom_pad = pad
        self._zoom_window = None
        self.OnRefresh(None)

    def show_page(self, page):
        if page >= self.n:
            raise ValueError("Page %s not available (%s pages)" % (page, self.n))
        else:
            #
            x, y = self.shape
            self.current_page_i = page
            self.page_choice.SetSelection(page)
            # fig
            indexes = self.i_lists[page]
            self.figure.clf()
            self.axes = []
#            self.gauge.SetRange(len(indexes))
            for i, index in enumerate(indexes):  # loop through list entries
                n_col = i % x
                n_row = i % self.n_per_page / x
                i_plot = n_row * x + n_col + 1
                ax = self.figure.add_subplot(y, x, i_plot)
                ax.x_fmt = 't = %.3g s'

                name = None
                for v in self.visualizers:
                    v.set_segment(index)
                    if name == None:
                        name = v.segment_name

                # store stuff for subclasses (GUIs)
                self.axes.append(ax)
                ax.segment_id = v.segment_id

                # determine zoom t1 and t2
                if self._zoom_events is not None:
                    v = self.visualizers[self._zoom_events]
                    time = v.dataset.experiment.variables.get('time')
                    duration = v.dataset.experiment.variables.get('duration')
                    if self._zoom_address is None:
                        seg = v._seg
                    else:
                        seg = v._seg[self._zoom_address]

                    t1 = seg[0][time] - self._zoom_pad
                    t2 = seg[-1][time] + self._zoom_pad
                    if duration in seg.varlist:
                        t2 += seg[-1][duration]
                elif self._zoom_window:
                    t1, t2 = self._zoom_window
                else:
                    t1 = None
                    t2 = None

                # plot
                logging.debug("plotting %s: t1=%s, t2=%s" % (name, t1, t2))
                for v in self.visualizers:
                    v.toax(ax, t1, t2)

                # xlim
                if t1 is None:
                    t1 = v._seg.tstart
                if t2 is None:
                    t2 = v._seg.tend
#                else:
#                    t1 = min(t1, segment.tstart)
#                    t2 = max(t2, segment.tend)
                ax.set_yticks([])
#                ax.set_ylabel(name, rotation='vertical', color='r')
                ax.set_xlim(t1, t2)

                # name
                if self.texsave:
                    name = name.replace('_', ' ')
                ax.text(0.01, 0.05, name, color='r', horizontalalignment='left',
                        verticalalignment='bottom',
                        transform=ax.transAxes)
                # prog
#                self.gauge.SetValue(i+1)
#                self.gauge.Update()
#            self.gauge.Pulse()
            self.canvas.draw()
#            self.gauge.SetValue(0)

    def OnBackward(self, event):
        if self.current_page_i == 0:
            self.show_page(self.n - 1)
        else:
            self.show_page(self.current_page_i - 1)

    def OnForward(self, event):
        if self.current_page_i < self.n - 1:
            self.show_page(self.current_page_i + 1)
        else:
            self.show_page(0)

    def OnRefresh(self, event):
        self.show_page(self.current_page_i)

    def OnPageChoice(self, event):
        page = int(event.GetString())
        self.show_page(page)

    def OnSave(self, event):
        dialog = wx.FileDialog(self, "Save Figure (If no extension is provided, pdf is used).",
                               style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
#                               wildcard="Current Page|*.pdf|All Pages (*_n.*)|*.gif")
        if dialog.ShowModal() == wx.ID_OK:
            path = dialog.GetPath()
#            wc = dialog.GetFilterIndex()
            wc = 0
            path, filename = os.path.split(path)
            if wc == 0:  # save only current page
                if not '.' in filename:
                    filename += '.pdf'
                self.figure.savefig(os.path.join(path, filename))
            elif wc == 1:  # save all pages
                fn_temp = os.path.join('{dir}', '{root}_{n}.{ext}')
                if '.' in filename:
                    filename, ext = filename.split('.')
                else:
                    ext = 'pdf'
                i = self.current_page_i
                self.figure.savefig(fn_temp.format(dir=path, root=filename, n=i, ext=ext))
                pages = range(self.n)
                pages.remove(i)
                prog = ui.progress(self.n - 1, "Saving Figures", "Saving Figures")
                for i in pages:
                    self.show_page(i)
                    self.figure.savefig(fn_temp.format(dir=path, root=filename, n=i, ext=ext))
                    prog.advance()
                prog.terminate()
            else:
                logging.error(" invalid wildcard: %s" % wc)



