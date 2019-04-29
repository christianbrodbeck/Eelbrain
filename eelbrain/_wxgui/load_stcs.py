"""GUI to detect and load stc files and experimental conditions

Prompts user for experiment design information, and upon submission
loads stcs into an ``eelbrain.Dataset`` via a ``DatasetSTCLoader``
instance.
"""

import os
import wx

from .frame import EelbrainFrame
from .._io.mrat import DatasetSTCLoader


TEST_MODE = False


class STCLoaderFrame(EelbrainFrame):

    add_params = dict(proportion=0, flag=wx.EXPAND | wx.ALL, border=10)

    def __init__(self, parent):
        super().__init__(parent, wx.ID_ANY, "Find and Load STCs")
        self.loader = None
        self.factor_name_ctrls = []
        self.InitUI()
        self.Show()
        self.Center()
        self.Raise()

    def InitUI(self):
        self.SetMinSize((500, -1))
        self.status = self.CreateStatusBar(1)
        self.sizer = wx.BoxSizer(wx.VERTICAL) # top-level sizer
        data_title = TitleSizer(self, "MEG/MRI Information")
        self.sizer.Add(data_title, **self.add_params)
        dir_label = wx.StaticText(self, label=".stc Data Directory")
        self.dir_ctl = wx.DirPickerCtrl(self)
        if self.dir_ctl.HasTextCtrl():
            self.dir_ctl.SetTextCtrlProportion(5)
            self.dir_ctl.SetPickerCtrlProportion(1)
        vsizer = wx.BoxSizer(wx.VERTICAL)
        vsizer.Add(dir_label, 0, wx.BOTTOM, 2)
        vsizer.Add(self.dir_ctl, 0, wx.EXPAND)
        self.sizer.Add(vsizer, **self.add_params)
        self.mri_panel = MRIPanel(self)
        self.sizer.Add(self.mri_panel, **self.add_params)
        self.design_title = TitleSizer(self, "Experiment Design")
        self.sizer.Add(self.design_title, **self.add_params)
        self.design_title.ctl.Hide()
        self.factor_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer.Add(self.factor_sizer, **self.add_params)
        bottom_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.submit = wx.Button(self, wx.ID_ANY, "Load Data")
        self.submit.Disable()
        bottom_sizer.Add(self.submit, 0, wx.ALIGN_RIGHT)
        self.sizer.Add(bottom_sizer, 0, wx.ALIGN_RIGHT | wx.ALL, 10)
        self.sizer.Layout()
        self.SetSizer(self.sizer)
        self.sizer.Fit(self)

        self.Bind(wx.EVT_DIRPICKER_CHANGED, self.OnDirChange, self.dir_ctl)
        self.Bind(wx.EVT_BUTTON, self.OnSubmit, self.submit)

    def OnDirChange(self, dir_picker_evt):
        """Create dataset loader and display level/factor names"""
        self.factor_sizer.Clear(True)
        path = dir_picker_evt.GetPath()
        try:
            self.loader = DatasetSTCLoader(path)
        except:
            self.status.SetStatusText("Error loading data from that directory.")
            return
        self.DisplayLevels(self.loader.levels)
        self.submit.Enable()

    def DisplayLevels(self, levels):
        """Show level names and factor name input for each factor"""
        self.design_title.ctl.Show()
        self.factor_name_ctrls = []
        for i, lvls in enumerate(levels):
            panel = FactorPanel(self, lvls, i)
            # self.factor_sizer.Add(panel, 0, wx.EXPAND | wx.RIGHT, 30)
            self.factor_sizer.Add(panel, 0, wx.EXPAND | wx.RIGHT, 30)
            self.factor_name_ctrls.append(panel.factor_ctl)
        self.factor_sizer.Layout()
        self.sizer.Layout()
        self.sizer.Fit(self)

    def _get_factor_names(self):
        return [c.GetValue() for c in self.factor_name_ctrls]

    def _get_stc_kwargs(self):
        kw = dict()
        kw["subjects_dir"] = self.mri_panel.mri_dir.GetPath()
        kw["subject"] = self.mri_panel.mri_subj.GetValue()
        kw["src"] = self.mri_panel.mri_src.GetValue()
        return kw

    def OnSubmit(self, evt):
        names = self._get_factor_names()
        self.loader.set_factor_names(names)
        stc_kw = self._get_stc_kwargs()
        _ = self.loader.make_dataset(**stc_kw)
        # Launch Stats GUI, passing ds to constructor


class FactorPanel(wx.Panel):
    def __init__(self, parent, levels, idx):
        super().__init__(parent)
        self.factor_ctl = wx.TextCtrl(self, value="factor_%d" % idx)
        level_names = ["- " + i for i in levels]
        level_ctl = wx.StaticText(self)
        level_ctl.SetLabel("\n".join(level_names))
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.factor_ctl)
        sizer.Add(level_ctl, 0, wx.EXPAND | wx.TOP, 10)
        sizer.Layout()
        self.SetSizer(sizer)


class MRIPanel(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent)
        sizer = wx.StaticBoxSizer(wx.VERTICAL, self, "")
        # check for subjects dir in environment
        sdir = os.environ.get("SUBJECTS_DIR")
        if sdir is None:
            sdir = "/Applications/freesurfer/subjects"
        # directory control for MRI subjects dir, default freesurfer
        dir_ctl = wx.DirPickerCtrl(self, path=sdir)
        if dir_ctl.HasTextCtrl():
            dir_ctl.SetTextCtrlProportion(5)
            dir_ctl.SetPickerCtrlProportion(1)
        # text control for MRI subject, default 'fsaverage'
        subj_ctl = wx.TextCtrl(self, value="fsaverage")
        # dropdown to choose source space sampling, default ico-4
        srcs = ["ico-%d" % i for i in range(2, 7)] + ["oct-%d" % i for i in range(2, 7)] + ["all"]
        src_ctl = wx.ComboBox(self, choices=srcs, value="ico-4")
        # attach controls to panel for use in loader
        self.mri_dir = dir_ctl
        self.mri_subj = subj_ctl
        self.mri_src = src_ctl
        sizer.Add(wx.StaticText(self, label="MRI Directory"), 0, wx.BOTTOM, 2)
        sizer.Add(dir_ctl, 0, wx.EXPAND | wx.BOTTOM, 5)
        hs = wx.BoxSizer(wx.HORIZONTAL)
        for label, ctl in zip(("Subject", "Source Space"),
                              (subj_ctl, src_ctl)):
            vs = wx.BoxSizer(wx.VERTICAL)
            vs.Add(wx.StaticText(self, label=label), 0)
            vs.Add(ctl, 0)
            hs.Add(vs, 0, wx.RIGHT, 15)
        sizer.Add(hs, 0, wx.BOTTOM, 5)
        sizer.Layout()
        self.SetSizer(sizer)


class TitleSizer(wx.BoxSizer):
    def __init__(self, parent, title):
        super().__init__(wx.HORIZONTAL)
        self.ctl = wx.StaticText(parent, label=title)
        font = wx.Font(14, wx.DEFAULT, wx.NORMAL, wx.BOLD)
        self.ctl.SetFont(font)
        self.Add(self.ctl)
