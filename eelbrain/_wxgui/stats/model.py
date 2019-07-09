import wx

from. utils import TitleSizer


class TestModelInfo(wx.Panel):
    def __init__(self, parent, loader):
        super().__init__(parent)
        self.loader = loader
        self.InitWidgets()
        self.InitUI()

    def InitWidgets(self):
        self.test_type = wx.RadioBox(self, choices=["ANOVA", "t-test"])
        self.anova_def = ANOVAModel(self, self.loader.factors)
        fld = {f: l for f, l in zip(self.loader.factors, self.loader.levels)}
        self.ttest_def = TTestModel(self, fld)
        self.Bind(wx.EVT_RADIOBOX, self.OnTestTypeChange, self.test_type)

    def InitUI(self):
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        row1 = wx.BoxSizer(wx.HORIZONTAL)
        row1.Add(self.test_type)
        self.sizer.Add(TitleSizer(self, "Test/Model Definition"), 0, wx.BOTTOM, 5)
        self.sizer.Add(row1)
        self.sizer.Add(self.anova_def)
        self.sizer.Add(self.ttest_def)
        self.sizer.Hide(self.ttest_def, recursive=True)
        self.SetSizer(self.sizer)

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

