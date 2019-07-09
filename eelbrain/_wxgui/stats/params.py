import wx

from .utils import TitleSizer, TextEntryWithLabel


class TestParams(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent)
        self.InitWidgets()
        self.InitUI()

    def InitWidgets(self):
        self.tstart = TextEntryWithLabel(self, wx.VERTICAL, "Start Time (s)")
        self.tstop = TextEntryWithLabel(self, wx.VERTICAL, "Stop Time (s)")
        self.samples = TextEntryWithLabel(self, wx.VERTICAL, "Permutations", "10000")
        self.mintime = TextEntryWithLabel(self, wx.VERTICAL, "Min. Cluster Time (s)", "0.02")
        self.minsource = TextEntryWithLabel(self, wx.VERTICAL, "Min. Cluster Sources", "25")
        self.sig = SignificanceType(self)
        self.title = TitleSizer(self, "Test Parameters")
        self.Bind(wx.EVT_RADIOBOX, self.sig.OnTypeChange, self.sig.choice)

    def InitUI(self):
        self.sizer = wx.GridBagSizer(hgap=10, vgap=10)
        self.sizer.Add(self.title, pos=(0, 0), span=(1, 2))
        self.sizer.Add(self.tstart, pos=(1, 0), span=(1, 2))
        self.sizer.Add(self.tstop, pos=(1, 2), span=(1, 2))
        self.sizer.Add(self.samples, pos=(1, 4), span=(1, 2))
        self.sizer.Add(self.mintime, pos=(2, 0), span=(1, 2))
        self.sizer.Add(self.minsource, pos=(2, 2), span=(1, 2))
        self.sizer.Add(self.sig, pos=(3, 0), span=(1, 6))
        self.SetSizer(self.sizer)
        # hide minsource at first, because 'temporal' is default choice
        # in SpatiotemporalSettings
        self.toggle_minsource()

    def get_test_kwargs(self):
        kwargs = dict()
        kwargs["tstart"] = float(self.tstart.GetValue())
        kwargs["tstop"] = float(self.tstop.GetValue())
        kwargs["samples"] = int(self.samples.GetValue())
        kwargs["mintime"] = float(self.mintime.GetValue())
        kwargs["minsource"] = int(self.minsource.GetValue())
        kwargs["tfce"] = self.sig.is_tfce()
        kwargs["pmin"] = float(self.sig.get_pmin())
        return kwargs

    def toggle_minsource(self, evt=None):
        self.minsource.Toggle()
        self.sizer.Layout()


class SignificanceType(wx.BoxSizer):
    def __init__(self, parent):
        super().__init__(wx.VERTICAL)
        self.choice = wx.RadioBox(parent, choices=["p-value", "TFCE"])
        self.pval = wx.TextCtrl(parent, value="0.05")
        self.Add(self.choice)
        self.Add(self.pval)

    def OnTypeChange(self, evt):
        if self.choice.GetSelection() == 1:
            self.pval.Hide()
        else:
            self.pval.Show()

    def is_tfce(self):
        return self.choice.GetSelection() == 1

    def get_pmin(self):
        if self.is_tfce():
            return None
        return float(self.pval.GetValue())


class SpatiotemporalSettings(wx.BoxSizer):
    def __init__(self, parent):
        super().__init__(wx.VERTICAL)
        title = TitleSizer(parent, "Permutation Test Type")
        self.choice = wx.RadioBox(parent, choices=["Temporal", "Spatiotemporal"])
        self.Add(title, 0, wx.BOTTOM, 5)
        self.Add(self.choice)

    def is_temporal(self):
        return self.choice.GetStringSelection() == "Temporal"
