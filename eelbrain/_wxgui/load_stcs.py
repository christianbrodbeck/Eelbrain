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
    def __init__(self, parent):
        super().__init__(parent, wx.ID_ANY, "Find and Load STCs")
        self.loader = None
        self.factor_name_ctrls = []
        self.InitUI()
        self.Show()
        self.Center()
        self.Raise()

    def InitUI(self):
        dir_label = wx.StaticText(self, label="Data directory")
        dir_ctl = wx.DirPickerCtrl(self)
        if dir_ctl.HasTextCtrl():
            dir_ctl.SetTextCtrlProportion(5)
            dir_ctl.SetPickerCtrlProportion(1)
        self.dir_ctl = dir_ctl
        vsizer = wx.BoxSizer(wx.VERTICAL)
        vsizer.Add(dir_label, 0, wx.BOTTOM, 2)
        vsizer.Add(dir_ctl, 0, wx.EXPAND)
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(vsizer, 0, wx.EXPAND | wx.ALL, 10)
        self.sizer.Add(self._create_mri_form(), 0,  wx.EXPAND | wx.ALL, 10)
        self.factor_sizer = wx.BoxSizer(wx.HORIZONTAL)
        design_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.design_title = wx.StaticText(self, label="Experiment Structure")
        font = wx.Font(18, wx.DEFAULT, wx.NORMAL, wx.BOLD)
        self.design_title.SetFont(font)
        design_sizer.Add(self.design_title)
        self.sizer.Add(design_sizer, 0, wx.EXPAND | wx.ALL, 10)
        self.design_title.Hide()
        self.sizer.Add(self.factor_sizer, 0, wx.EXPAND | wx.ALL, 10)
        bottom_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.submit = wx.Button(self, wx.ID_ANY, "Load Data")
        self.submit.Disable()
        bottom_sizer.Add(self.submit, 0, wx.ALIGN_RIGHT)
        self.sizer.Add(bottom_sizer, 0, wx.ALIGN_RIGHT | wx.ALL, 10)
        self.sizer.Layout()
        self.SetSizer(self.sizer)
        self.sizer.Fit(self)

        self.Bind(wx.EVT_DIRPICKER_CHANGED, self.OnDirChange, dir_ctl)
        self.Bind(wx.EVT_BUTTON, self.OnSubmit, self.submit)

    def _create_mri_form(self):
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
        # attach controls to frame for use in loader
        self.mri_dir = dir_ctl
        self.mri_subj = subj_ctl
        self.mri_src = src_ctl
        ms = wx.BoxSizer(wx.VERTICAL)
        ms.Add(wx.StaticText(self, label="MRI Directory"), 0, wx.BOTTOM, 2)
        ms.Add(dir_ctl, 0, wx.EXPAND)
        sizer.Add(ms, 0, wx.BOTTOM, 5)
        hs = wx.BoxSizer(wx.HORIZONTAL)
        for label, ctl in zip(("Subject", "Source Space"),
                              (subj_ctl, src_ctl)):
            vs = wx.BoxSizer(wx.VERTICAL)
            vs.Add(wx.StaticText(self, label=label), 0)
            vs.Add(ctl, 0)
            hs.Add(vs, 0, wx.RIGHT, 15)
        sizer.Add(hs)
        sizer.Layout()
        return sizer

    def OnDirChange(self, dir_picker_evt):
        """Create dataset loader and display level/factor names"""
        path = dir_picker_evt.GetPath()
        self.loader = DatasetSTCLoader(path)
        self.DisplayLevels(self.loader.levels)
        self.submit.Enable()

    def DisplayLevels(self, levels):
        """Show level names and factor name input for each factor"""
        self.design_title.Show()
        self.factor_name_ctrls = []
        self.factor_sizer.Clear(True)
        for i, lvls in enumerate(levels):
            sizer = self._create_factor_sizer(lvls, i)
            self.factor_sizer.Add(sizer, 0, wx.EXPAND | wx.RIGHT, 30)
        self.factor_sizer.Layout()
        self.sizer.Layout()
        self.sizer.Fit(self)

    def _create_factor_sizer(self, level_names, idx):
        sizer = wx.BoxSizer(wx.VERTICAL)
        fctl = wx.TextCtrl(self, value="factor_%d" % idx, style=wx.TE_CENTER)
        self.factor_name_ctrls.append(fctl)
        ctl = wx.StaticText(self)
        level_names = ["- " + i for i in level_names]
        ctl.SetLabel("\n".join(level_names))
        sizer.Add(fctl)
        sizer.Add(ctl, 1, wx.EXPAND | wx.TOP, 10)
        return sizer

    def _get_factor_names(self):
        return [c.GetValue() for c in self.factor_name_ctrls]

    def _get_stc_kwargs(self):
        kw = dict()
        kw["subjects_dir"] = self.mri_dir.GetPath()
        kw["subject"] = self.mri_subj.GetValue()
        kw["src"] = self.mri_src.GetValue()
        return kw

    def OnSubmit(self, evt):
        names = self._get_factor_names()
        self.loader.set_factor_names(names)
        stc_kw = self._get_stc_kwargs()
        ds = self.loader.make_dataset(**stc_kw)
        # Launch Stats GUI, passing ds to constructor
