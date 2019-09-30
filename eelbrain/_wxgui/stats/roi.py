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

    def __init__(self, parent, src_ndvar, state=None):
        super().__init__(parent, wx.OK | wx.CANCEL)
        self.subject = src_ndvar.source.subject
        self.subjects_dir = src_ndvar.source.subjects_dir
        self.InitWidgets()
        self.InitUI()
        self.SetState(state)

    def InitWidgets(self):
        self.atlas = wx.RadioBox(self, choices=list(self.ATLASES.keys()),
                                 majorDimension=1)
        self.regions = wx.CheckListBox(self, choices=[])
        self.mult_corr = wx.CheckBox(self, label="Correct Across ROIs")
        self.ok = wx.Button(self, id=wx.ID_OK, label="OK")
        self.cancel = wx.Button(self, id=wx.ID_CANCEL, label="Cancel")
        self.Bind(wx.EVT_RADIOBOX, self.OnAtlasChange, self.atlas)
        self.Bind(wx.EVT_CHECKLISTBOX, self.OnRegionSelect, self.regions)

    def InitUI(self):
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        content_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.col1 = wx.BoxSizer(wx.VERTICAL)
        atlas_title = TitleSizer(self, "Atlas")
        self.col1.Add(atlas_title)
        self.col1.Add(self.atlas)
        self.col1.Add(self.mult_corr, 0, wx.TOP, 20)
        content_sizer.Add(self.col1, 1, **self.add_params)
        content_sizer.Add(self.regions, 2, **self.add_params)
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        button_sizer.Add(self.ok)
        button_sizer.Add(self.cancel)
        self.sizer.Add(content_sizer, 0, wx.BOTTOM, 30)
        self.sizer.Add(button_sizer, 0, wx.BOTTOM, 10)
        self.SetSizer(self.sizer)
        self.sizer.Layout()
        self.sizer.Fit(self)
        self.OnAtlasChange(None)
        self.OnRegionSelect(None)

    def SetState(self, state=None):
        if state:
            atlas_inv = {v: k for k, v in self.ATLASES.items()}
            self.atlas.SetStringSelection(atlas_inv[state["atlas"]])
            self.OnAtlasChange(None)
            self.regions.SetCheckedStrings(list(state["labels"]))
            self.OnRegionSelect(None)
            self.mult_corr.SetValue(state["corr"])

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

    def get_roi_status_string(self):
        info = self.get_roi_info()
        status = "ATLAS: {}\n".format(self.atlas.GetStringSelection())
        status += "LABELS: {}".format(",".join(info["labels"]))
        if info["corr"]:
            status += "\nCorrect Across ROIs"
        return status
