import wx

from ..frame import EelbrainFrame
from ... import plot


class StatsResultsFrame(EelbrainFrame):

    ALL_COLUMNS = ("effect", "p", "sig", "tstart", "tstop", "duration",
                   "location", "hemi", "n_sources")

    def __init__(self, parent, res, ds, data):
        super().__init__(parent, id=wx.ID_ANY)
        self.res = res
        self.ds = ds
        self.data = data
        self.columns = [i for i in self.ALL_COLUMNS if i in res.clusters]
        self.InitUI()
        self.Raise()
        self.Show()

    def InitUI(self):
        self.sizer = wx.BoxSizer()
        self.clist = wx.ListCtrl(self, style=wx.LC_REPORT)
        for i, col in enumerate(self.columns):
            self.clist.InsertColumn(i, col)
        for i, cl in enumerate(self.res.clusters.itercases()):
            self.clist.InsertItem(i, cl["id"])
            for idx, col in enumerate(self.columns):
                if col in ("tstart", "tstop", "p", "duration"):
                    val = "{:0.3f}".format(cl[col])
                else:
                    val = str(cl[col])
                self.clist.SetStringItem(i, idx, val)

        self.sizer.Add(self.clist)
        self.SetSizer(self.sizer)
        self.sizer.Layout()
        self.sizer.Fit(self)
        self.Bind(wx.EVT_LIST_ITEM_ACTIVATED, self.OnClusterClick, self.clist)

    def OnClusterClick(self, evt):
        cl_idx = evt.GetItem().GetId()
        info = self._cluster_info_for_plot(cl_idx)
        if self.data.ndim == 2:
            self._plot_temporal(info)
        elif self.data.ndim == 3:
            self._plot_spatiotemporal(info)

    def _cluster_info_for_plot(self, cl_idx):
        cl_ds = self.res.clusters[cl_idx:cl_idx + 1]
        cluster = self.res.clusters[cl_idx]
        plot_model = cluster.get("effect") or self.res.x
        if " x " in plot_model:
            plot_model = plot_model.replace(" x ", " % ")
        return dict(
            cluster=cluster, plot_model=plot_model,
            cluster_ds=cl_ds
        )

    def _plot_temporal(self, info):
        p = plot.UTSStat(self.data, ds=self.ds, x=info["plot_model"],
                         pmax=1.0, ptrend=1.0, clusters=info["cluster_ds"])

    def _plot_spatiotemporal(self, info):
        cl = info["cluster"]
        # mask to any source that participates in the cluster at any time
        cl_mask = cl["cluster"].any("time")
        ts = self.data[cl_mask].mean("source")  # average over source to plot timecourse
        p = plot.UTSStat(ts, ds=self.ds, x=info["plot_model"], pmax=1.0,
                         ptrend=1.0, clusters=info["cluster_ds"])
        spatial = cl["cluster"].mean("time")
        b = plot.brain.cluster(spatial, hemi=cl["hemi"], surf="pial")