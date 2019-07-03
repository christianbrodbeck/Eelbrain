import wx
from mne import read_labels_from_annot

from ..frame import EelbrainFrame
from ..load_stcs import FactorPanel, TitleSizer
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
        self.submit = wx.Button(self, label="Go")
        self.sizer.Add(self.submit)
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
        ds = self.ds.sub("subject.isin(subjects)")
        src = ds["src"]
        src = set_parc(src, info["atlas"])
        if self.spatiotemp.is_temporal():
            data = src.summary(source=info["label"])
        else:
            data = src.sub(source=info["label"])
        return ds, data

    def run_test(self, evt):
        # TODO: handle multiple correction over regions
        test_type = self.test_model.get_test_type()
        if test_type == "ANOVA":
            test_func = testnd.anova
        elif test_type == "t-test":
            test_func = testnd.ttest_rel
        ds, data = self.get_data_for_test()
        kwargs = self.get_test_kwargs()
        res = test_func(data, ds=ds, match="subject", **kwargs)
        print(res)

    def OnROIMenuClick(self, evt):
        with RegionOfInterest(self, self.ds["src"]) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self.roi_info = dlg.get_roi_info()
            else:
                pass
            dlg.Destroy()


class SubjectsSelector(wx.CheckListBox):
    def __init__(self, parent, subjects=[], **kwargs):
        super().__init__(parent, choices=subjects, style=wx.LC_LIST, **kwargs)
        self.subjects = subjects
        self.SetCheckedItems(range(len(self.subjects)))

    def get_selected_subjects(self):
        return self.GetCheckedStrings()


class InfoPanel(wx.Panel):

    add_params = dict(proportion=0, flag=wx.EXPAND | wx.ALL, border=10)

    def __init__(self, parent, loader=None, ds=None):
        super().__init__(parent)
        self.loader = loader
        self.ds = ds
        self.sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.subj_sel = SubjectsSelector(self, list(set(self.ds["subject"])))
        col1 = wx.BoxSizer(wx.VERTICAL)
        col1.Add(TitleSizer(self, "Subjects"), 0, wx.BOTTOM, 5)
        col1.Add(self.subj_sel)
        self.sizer.Add(col1, **self.add_params)
        col2 = wx.BoxSizer(wx.VERTICAL)
        col2.Add(TitleSizer(self, "Design"), 0, wx.BOTTOM, 5)
        self.factor_sizer = wx.BoxSizer(wx.HORIZONTAL)
        for i, level_names in enumerate(self.loader.levels):
            panel = FactorPanel(self, level_names, i, editable=False)
            panel.factor_ctl.SetLabel(self.loader.factors[i])
            panel.factor_ctl.SetFont(wx.Font.Bold(panel.factor_ctl.GetFont()))
            self.factor_sizer.Add(panel, 0, wx.EXPAND | wx.RIGHT, 30)
        col2.Add(self.factor_sizer)
        self.sizer.Add(col2, **self.add_params)
        col3 = wx.BoxSizer(wx.VERTICAL)
        col3.Add(TitleSizer(self, "Data"), 0, wx.BOTTOM, 5)
        src = self.ds["src"]
        n_src = src.source.vertices[0].shape[0] + src.source.vertices[0].shape[0]
        src_label = "{} sources".format(n_src)
        col3.Add(wx.StaticText(self, label=src_label))
        time_label = "{:0.2f} to {:0.2f} s, step={:0.3f} s".format(
            src.time.times[0],
            src.time.tstop, src.time.tstep
        )
        col3.Add(wx.StaticText(self, label=time_label))
        self.sizer.Add(col3, **self.add_params)
        self.sizer.Layout()
        self.SetSizer(self.sizer)


class TestParams(wx.Panel):

    def __init__(self, parent):
        super().__init__(parent)
        self.InitUI()

    def InitUI(self):
        self.sizer = wx.GridBagSizer(hgap=10, vgap=10)
        title = TitleSizer(self, "Test Parameters")
        self.tstart = TextEntryWithLabel(self, wx.VERTICAL, "Start Time (s)")
        self.tstop = TextEntryWithLabel(self, wx.VERTICAL, "Stop Time (s)")
        self.samples = TextEntryWithLabel(self, wx.VERTICAL, "Permutations", "10000")
        self.mintime = TextEntryWithLabel(self, wx.VERTICAL, "Min. Cluster Time (s)", "0.02")
        self.minsource = TextEntryWithLabel(self, wx.VERTICAL, "Min. Cluster Sources", "25")
        self.sig = SignificanceType(self)
        self.sizer.Add(title, pos=(0, 0), span=(1, 3))
        self.sizer.Add(self.tstart, pos=(1, 0), span=(1, 2))
        self.sizer.Add(self.tstop, pos=(1, 2), span=(1, 2))
        self.sizer.Add(self.samples, pos=(1, 4), span=(1, 2))
        self.sizer.Add(self.mintime, pos=(2, 0), span=(1, 2))
        self.sizer.Add(self.minsource, pos=(2, 2), span=(1, 2))
        self.sizer.Add(self.sig, pos=(3, 0), span=(1, 6))
        self.SetSizer(self.sizer)
        self.Bind(wx.EVT_RADIOBOX, self.sig.OnTypeChange, self.sig.choice)
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


class TextEntryWithLabel(wx.BoxSizer):
    label_add_params = {
        wx.HORIZONTAL: dict(proportion=0, flag=wx.RIGHT, border=10),
        wx.VERTICAL: dict(proportion=0, flag=wx.BOTTOM, border=2)
    }

    def __init__(self, parent, orientation=wx.HORIZONTAL, label="", value=""):
        super().__init__(orientation)
        self.label = wx.StaticText(parent, label=label)
        self.field = wx.TextCtrl(parent, value=value)
        self.Add(self.label, **self.label_add_params[orientation])
        self.Add(self.field)

    def GetValue(self):
        return self.field.GetValue()

    def Toggle(self):
        if self.field.IsShown():
            self.field.Hide()
        else:
            self.field.Show()
        if self.label.IsShown():
            self.label.Hide()
        else:
            self.label.Show()


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


class TestModelInfo(wx.Panel):
    def __init__(self, parent, loader):
        super().__init__(parent)
        self.loader = loader
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(TitleSizer(self, "Test/Model Definition"), 0, wx.BOTTOM, 5)
        row1 = wx.BoxSizer(wx.HORIZONTAL)
        self.test_type = wx.RadioBox(self, choices=["ANOVA", "t-test"])
        row1.Add(self.test_type)
        self.sizer.Add(row1)
        self.anova_def = ANOVAModel(self, self.loader.factors)
        fld = {f: l for f, l in zip(self.loader.factors, self.loader.levels)}
        self.ttest_def = TTestModel(self, fld)
        self.sizer.Add(self.anova_def)
        self.sizer.Add(self.ttest_def)
        self.sizer.Hide(self.ttest_def, recursive=True)
        self.SetSizer(self.sizer)

        self.Bind(wx.EVT_RADIOBOX, self.OnTestTypeChange, self.test_type)

    def OnTestTypeChange(self, evt):
        test_type = self.get_test_type()
        if test_type == "ANOVA":
            self.sizer.Show(self.anova_def, recursive=True)
            self.sizer.Hide(self.ttest_def, recursive=True)
        elif test_type == "t-test":
            self.sizer.Show(self.ttest_def, recursive=True)
            self.sizer.Hide(self.anova_def, recursive=True)
        self.sizer.Layout()

    def get_test_type(self):
        return self.test_type.GetStringSelection()

    def get_test_kwargs(self):
        test_type = self.get_test_type()
        if test_type == "ANOVA":
            return self.anova_def.get_test_kwargs()
        elif test_type == "t-test":
            return self.ttest_def.get_test_kwargs()


class ANOVAModel(wx.BoxSizer):
    def __init__(self, parent, factors=[], **kwargs):
        super().__init__(wx.HORIZONTAL)
        self.model = wx.CheckListBox(parent, choices=factors,
                                     style=wx.LC_LIST, **kwargs)
        self.Add(self.model)
        self.Layout()

    def get_test_kwargs(self):
        kwargs = dict()
        kwargs["x"] = " * ".join(self.model.GetCheckedStrings())
        return kwargs


class TTestModel(wx.BoxSizer):

    add_params = dict(proportion=0, flag=wx.RIGHT, border=5)

    def __init__(self, parent, factor_level_dict):
        super().__init__(wx.HORIZONTAL)
        self.fld = factor_level_dict
        factor_sizer = wx.BoxSizer(wx.VERTICAL)
        self.factor = wx.RadioBox(parent, choices=list(self.fld.keys()),
                                  style=wx.RA_SPECIFY_COLS, majorDimension=1)
        factor_sizer.Add(wx.StaticText(parent, label="Select Factor"))
        factor_sizer.Add(self.factor)
        self.Add(factor_sizer, **self.add_params)
        level1_sizer = wx.BoxSizer(wx.VERTICAL)
        level2_sizer = wx.BoxSizer(wx.VERTICAL)
        self.level1 = wx.ComboBox(parent, choices=[])
        self.level2 = wx.ComboBox(parent, choices=[])
        level1_sizer.Add(wx.StaticText(parent, label="Condition 1"))
        level2_sizer.Add(wx.StaticText(parent, label="Condition 2"))
        level1_sizer.Add(self.level1)
        level2_sizer.Add(self.level2)
        self.Add(level1_sizer, **self.add_params)
        self.Add(level2_sizer, **self.add_params)
        self.OnFactorChange(None)
        self.Layout()
        parent.Bind(wx.EVT_RADIOBOX, self.OnFactorChange, self.factor)

    def OnFactorChange(self, evt):
        factor = self.factor.GetStringSelection()
        levels = self.fld[factor]
        for cb in (self.level1, self.level2):
            cb.Clear()
            cb.AppendItems(levels)

    def get_test_kwargs(self):
        kwargs = dict()
        kwargs["x"] = self.factor.GetStringSelection()
        kwargs["c0"] = self.level1.GetValue()
        kwargs["c1"] = self.level2.GetValue()
        return kwargs


class SpatiotemporalSettings(wx.BoxSizer):
    def __init__(self, parent):
        super().__init__(wx.VERTICAL)
        title = TitleSizer(parent, "Permutation Test Type")
        self.choice = wx.RadioBox(parent, choices=["Temporal", "Spatiotemporal"])
        self.Add(title, 0, wx.BOTTOM, 5)
        self.Add(self.choice)

    def is_temporal(self):
        return self.choice.GetStringSelection() == "Temporal"


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
