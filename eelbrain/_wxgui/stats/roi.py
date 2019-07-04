import wx
from mne import read_labels_from_annot

from .utils import TitleSizer


class RegionOfInterest(wx.Dialog):
    ATLASES = {
        "Brodmann": "PALS_B12_Brodmann",
        "Desikan-Killiany": "aparc",
        "Lobes": "PALS_B12_Lobes"
    }

    add_params = dict(flag=wx.ALL | wx.EXPAND, border=5)

    def __init__(self, parent, src_ndvar):
        super().__init__(parent, wx.OK | wx.CANCEL)
        self.subject = src_ndvar.source.subject
        self.subjects_dir = src_ndvar.source.subjects_dir
        self.InitUI()

    def InitUI(self):
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        content_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.col1 = wx.BoxSizer(wx.VERTICAL)
        self.atlas = wx.RadioBox(self, choices=list(self.ATLASES.keys()),
                                 majorDimension=1)
        atlas_title = TitleSizer(self, "Atlas")
        self.col1.Add(atlas_title)
        self.col1.Add(self.atlas)
        self.regions = wx.CheckListBox(self, choices=[])
        self.mult_corr = wx.CheckBox(self, label="Correct Across ROIs")
        self.col1.Add(self.mult_corr, 0, wx.TOP, 20)
        content_sizer.Add(self.col1, 1, **self.add_params)
        content_sizer.Add(self.regions, 2, **self.add_params)
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        ok = wx.Button(self, id=wx.ID_OK, label="OK")
        cancel = wx.Button(self, id=wx.ID_CANCEL, label="Cancel")
        button_sizer.Add(ok)
        button_sizer.Add(cancel)
        self.sizer.Add(content_sizer, 0, wx.BOTTOM, 30)
        self.sizer.Add(button_sizer, 0, wx.BOTTOM, 10)
        self.SetSizer(self.sizer)
        self.sizer.Layout()
        self.sizer.Fit(self)
        self.Bind(wx.EVT_RADIOBOX, self.OnAtlasChange, self.atlas)
        self.Bind(wx.EVT_CHECKLISTBOX, self.OnRegionSelect, self.regions)
        self.OnAtlasChange(None)
        self.OnRegionSelect(None)

    def OnRegionSelect(self, evt):
        regions = self.regions.GetCheckedStrings()
        if len(regions) < 2:
            self.mult_corr.SetValue(False)
            self.mult_corr.Disable()
        else:
            self.mult_corr.Enable()

    def OnAtlasChange(self, evt):
        atlas = self.ATLASES[self.atlas.GetStringSelection()]
        labels = read_labels_from_annot(self.subject, atlas,
                                        subjects_dir=self.subjects_dir)
        label_names = [lbl.name for lbl in labels]
        label_names = list(filter(lambda x: "?" not in x, label_names))
        self.regions.Clear()
        self.regions.AppendItems(label_names)

    def get_roi_info(self):
        info = dict()
        info["atlas"] = self.ATLASES[self.atlas.GetStringSelection()]
        info["labels"] = self.regions.GetCheckedStrings()
        info["corr"] = self.mult_corr.IsChecked()
        return info
