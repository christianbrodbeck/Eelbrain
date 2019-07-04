import wx

from ..frame import EelbrainFrame
from .roi import RegionOfInterest
from .info import InfoPanel
from .params import TestParams, SpatiotemporalSettings
from .model import TestModelInfo
from ... import testnd, set_parc


class StatsFrame(EelbrainFrame):

    add_params = dict(proportion=0, flag=wx.EXPAND | wx.ALL, border=10)

    def __init__(self, parent, loader=None, ds=None):
        super().__init__(parent, id=wx.ID_ANY)
        self.loader = loader
        self.ds = ds
        self.roi_info = None
        self.InitUI()
        self.Show()
        self.Raise()

    def InitUI(self):
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.info_panel = InfoPanel(self, self.loader, self.ds)
        self.sizer.Add(self.info_panel)
        self.spatiotemp = SpatiotemporalSettings(self)
        self.sizer.Add(self.spatiotemp, **self.add_params)
        self.test_model = TestModelInfo(self, self.loader)
        self.sizer.Add(self.test_model, **self.add_params)
        self.test_params = TestParams(self)
        self.sizer.Add(self.test_params, **self.add_params)
        self.roi_dialog = wx.Button(self, label="Select ROI")
        self.sizer.Add(self.roi_dialog, **self.add_params)
        self.submit = wx.Button(self, label="Run Test")
        self.sizer.Add(self.submit, **self.add_params)
        self.Bind(wx.EVT_BUTTON, self.run_test, self.submit)
        self.Bind(wx.EVT_RADIOBOX, self.test_params.toggle_minsource,
                  self.spatiotemp.choice)
        self.Bind(wx.EVT_BUTTON, self.OnROIMenuClick, self.roi_dialog)
        self.SetSizer(self.sizer)
        self.sizer.Layout()
        self.sizer.Fit(self)

    def get_test_kwargs(self):
        kwargs = dict()
        kwargs.update(self.test_model.get_test_kwargs())
        kwargs.update(self.test_params.get_test_kwargs())
        return kwargs

    def get_data_for_test(self):
        info = self.roi_info
        if info is None:
            raise RuntimeError("No ROI information provided.")
        subjects = self.info_panel.subj_sel.get_selected_subjects()
        idx = self.ds["subject"].isin(subjects)
        ds = self.ds.sub(idx)
        src = ds["src"]
        src = set_parc(src, info["atlas"])
        if self.spatiotemp.is_temporal():
            data = src.summary(source=info["labels"])
        else:
            data = src.sub(source=info["labels"])
        return ds, data

    def run_test(self, evt):
        # TODO: handle correction over multiple regions
        test_type = self.test_model.get_test_type()
        if test_type == "ANOVA":
            test_func = testnd.anova
        elif test_type == "t-test":
            test_func = testnd.ttest_rel
        ds, data = self.get_data_for_test()
        kwargs = self.get_test_kwargs()
        if self.spatiotemp.is_temporal():
            del kwargs["minsource"]
        res = test_func(data, ds=ds, match="subject", **kwargs)
        print(res)

    def OnROIMenuClick(self, evt):
        with RegionOfInterest(self, self.ds["src"]) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self.roi_info = dlg.get_roi_info()
            else:
                pass
            dlg.Destroy()
