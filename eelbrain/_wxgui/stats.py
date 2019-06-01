import wx

from .frame import EelbrainFrame
from .load_stcs import FactorPanel, TitleSizer


class StatsFrame(EelbrainFrame):

    add_params = dict(proportion=0, flag=wx.EXPAND | wx.ALL, border=10)

    def __init__(self, parent, loader=None, ds=None):
        super().__init__(parent, id=wx.ID_ANY)
        self.loader = loader
        self.ds = ds
        self.InitUI()
        self.Show()
        self.Raise()

    def InitUI(self):
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        panel = InfoPanel(self, self.loader, self.ds)
        self.sizer.Add(panel)
        self.spatiotemp = SpatiotemporalSettings(self)
        self.sizer.Add(self.spatiotemp, **self.add_params)
        self.test_model = TestModelInfo(self, self.loader)
        self.sizer.Add(self.test_model, **self.add_params)
        self.test_params = TestParams(self)
        self.sizer.Add(self.test_params, **self.add_params)
        self.SetSizer(self.sizer)
        self.sizer.Layout()
        self.sizer.Fit(self)

    def get_test_kwargs(self, evt):
        kwargs = dict()
        kwargs.update(self.test_model.get_test_kwargs())
        kwargs.update(self.test_params.get_test_kwargs())
        print(kwargs)


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
        self.tstart = TextEntryWithLabel(self, wx.VERTICAL, "Start Time")
        self.tstop = TextEntryWithLabel(self, wx.VERTICAL, "Stop Time")
        self.samples = TextEntryWithLabel(self, wx.VERTICAL, "Permutations", "10000")
        self.sig = SignificanceType(self)
        self.sizer.Add(title, pos=(0, 0), span=(1, 3))
        self.sizer.Add(self.tstart, pos=(1, 0), span=(1, 2))
        self.sizer.Add(self.tstop, pos=(1, 2), span=(1, 2))
        self.sizer.Add(self.samples, pos=(1, 4), span=(1, 2))
        self.sizer.Add(self.sig, pos=(2, 0), span=(1, 6))
        self.SetSizer(self.sizer)
        self.Bind(wx.EVT_RADIOBOX, self.sig.OnTypeChange, self.sig.choice)

    def get_test_kwargs(self):
        kwargs = dict()
        kwargs["tstart"] = self.tstart.GetValue()
        kwargs["tstop"] = self.tstop.GetValue()
        kwargs["samples"] = self.samples.GetValue()
        kwargs["tfce"] = self.sig.is_tfce()
        kwargs["pmin"] = self.sig.get_pmin()
        return kwargs


class TextEntryWithLabel(wx.BoxSizer):
    label_add_params = {
        wx.HORIZONTAL: dict(proportion=0, flag=wx.RIGHT, border=10),
        wx.VERTICAL: dict(proportion=0, flag=wx.BOTTOM, border=2)
    }

    def __init__(self, parent, orientation=wx.HORIZONTAL, label="", value=""):
        super().__init__(orientation)
        label = wx.StaticText(parent, label=label)
        self.field = wx.TextCtrl(parent, value=value)
        self.Add(label, **self.label_add_params[orientation])
        self.Add(self.field)

    def GetValue(self):
        return self.field.GetValue()


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
        test_type = self.test_type.GetStringSelection()
        if test_type == "ANOVA":
            self.sizer.Show(self.anova_def, recursive=True)
            self.sizer.Hide(self.ttest_def, recursive=True)
        elif test_type == "t-test":
            self.sizer.Show(self.ttest_def, recursive=True)
            self.sizer.Hide(self.anova_def, recursive=True)
        self.sizer.Layout()

    def get_test_kwargs(self):
        test_type = self.test_type.GetStringSelection()
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

