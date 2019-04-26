import wx

from .frame import EelbrainFrame
from .._io.mrat import DatasetSTCLoader


class STCLoaderFrame(EelbrainFrame):
    def __init__(self, parent):
        super().__init__(parent, wx.ID_ANY, "Find and Load STCs", size=(500, 350))
        self.loader = None
        self.factor_name_ctrls = []
        self.InitUI()
        self.Show()
        self.Center()
        self.Raise()

    def InitUI(self):
        dir_label = wx.StaticText(self, label="Choose your data directory")
        dir_ctl = wx.DirPickerCtrl(self)
        if dir_ctl.HasTextCtrl():
            dir_ctl.SetTextCtrlProportion(5)
            dir_ctl.SetPickerCtrlProportion(1)
        vsizer = wx.BoxSizer(wx.VERTICAL)
        vsizer.Add(dir_label, 0, wx.BOTTOM, 5)
        vsizer.Add(dir_ctl, 0, wx.EXPAND)
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(vsizer, 0, wx.EXPAND | wx.ALL, 20)
        self.factor_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer.Add(self.factor_sizer, 1, wx.EXPAND | wx.ALL, 20)
        bottom_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.submit = wx.Button(self, wx.ID_ANY, "Load Data")
        self.submit.Disable()
        bottom_sizer.Add(self.submit, 0, wx.ALIGN_RIGHT)
        self.sizer.Add(bottom_sizer, flag=wx.ALIGN_RIGHT | wx.ALL, border=20)
        self.sizer.Layout()
        self.SetSizer(self.sizer)

        self.Bind(wx.EVT_DIRPICKER_CHANGED, self.OnDirChange, dir_ctl)
        self.Bind(wx.EVT_BUTTON, self.OnSubmit, self.submit)

    def OnDirChange(self, dir_picker_evt):
        path = dir_picker_evt.GetPath()
        self.loader = DatasetSTCLoader(path)
        self.DisplayLevels(self.loader.levels)
        self.submit.Enable()

    def DisplayLevels(self, levels):
        for i, lvls in enumerate(levels):
            sizer = self._create_factor_sizer(lvls, i)
            self.factor_sizer.Add(sizer, 0, wx.EXPAND | wx.RIGHT, 30)
        self.sizer.Layout()

    def _create_factor_sizer(self, level_names, idx):
        sizer = wx.BoxSizer(wx.VERTICAL)
        fctl = wx.TextCtrl(self, value="factor_%d" % idx)
        self.factor_name_ctrls.append(fctl)
        ctl = wx.StaticText(self)
        level_names = ["- " + i for i in level_names]
        ctl.SetLabel("\n".join(level_names))
        sizer.Add(fctl, 1, wx.EXPAND | wx.TOP, 15)
        sizer.Add(ctl, 1, wx.EXPAND | wx.TOP, 10)
        return sizer

    def _get_factor_names(self):
        return [c.GetValue() for c in self.factor_name_ctrls]

    def OnSubmit(self, evt):
        names = self._get_factor_names()
        self.loader.set_factor_names(names)
        ds = self.loader.make_dataset()
        # Launch Stats GUI, passing ds to constructor
