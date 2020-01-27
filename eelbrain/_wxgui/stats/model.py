import wx

from. utils import TitleSizer, ValidationException


class TestModelInfo(wx.Panel):
    def __init__(self, parent, loader):
        super().__init__(parent)
        self.loader = loader
        self.InitWidgets()
        self.InitUI()

    def InitWidgets(self):
        self.test_type = wx.RadioBox(self, choices=["ANOVA", "t-test"])
        self.anova_def = ANOVAModel(self, self.loader.factors, self.loader.levels)
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

    def validate(self):
        kwargs = self.get_test_kwargs()
        test_type = self.get_test_type()
        if test_type == "ANOVA":
            if not kwargs["x"]:
                raise ValidationException("Select at least one ANOVA factor.")
        elif test_type == "t-test":
            if kwargs["c0"] == "" or kwargs["c1"] == "":
                raise ValidationException("Select both levels for the t-test.")


class ANOVAFactor(wx.BoxSizer):
    def __init__(self, parent, name, levels):
        super().__init__(wx.VERTICAL)
        self.name = name
        self.levels = levels
        self.factor_select = wx.CheckBox(parent, label=name)
        self.Add(self.factor_select)
        self.level_select = wx.CheckListBox(parent, choices=levels, style=wx.LC_LIST)
        self.level_select.Disable()
        self.Add(self.level_select)
        parent.Bind(wx.EVT_CHECKBOX, self.OnFactorToggle, self.factor_select)
        parent.Bind(wx.EVT_CHECKLISTBOX, self.OnLevelToggle, self.level_select)

    def is_selected(self):
        return self.factor_select.IsChecked()

    def OnFactorToggle(self, evt):
        if evt.IsChecked():
            self.level_select.Enable()
            self.level_select.SetCheckedItems(range(len(self.levels)))
        else:
            self.level_select.SetCheckedItems([])
            self.level_select.Disable()

    def OnLevelToggle(self, evt):
        idx = evt.GetInt()
        items = self.level_select.GetCheckedItems()
        if len(items) < 2:
            self.level_select.Check(idx)
            message = "You must select at least 2 levels to include a factor."
            wx.MessageBox(message, "Invalid ANOVA Model", wx.OK)


class ANOVAModel(wx.BoxSizer):
    def __init__(self, parent, factors, levels):
        super().__init__(wx.HORIZONTAL)
        self.factor_boxes = []
        for fact, levs in zip(factors, levels):
            factor_box = ANOVAFactor(parent, fact, levs)
            self.factor_boxes.append(factor_box)
            self.Add(factor_box)
        self.Layout()

    def get_test_kwargs(self):
        kwargs = dict()
        factors = [i.name for i in self.factor_boxes if i.is_selected()]
        kwargs["x"] = " * ".join(factors)
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
        parent.Bind(wx.EVT_COMBOBOX, self.OnLevelChange, self.level1)
        parent.Bind(wx.EVT_COMBOBOX, self.OnLevelChange, self.level2)

    def OnFactorChange(self, evt):
        factor = self.factor.GetStringSelection()
        levels = self.fld[factor]
        for cb in (self.level1, self.level2):
            cb.Clear()
            cb.AppendItems(levels)

    def OnLevelChange(self, evt):
        kwargs = self.get_test_kwargs()
        if kwargs["c0"] == kwargs["c1"]:
            message = "Levels for t-test must be distinct."
            wx.MessageBox(message, "Invalid t-test Model", wx.OK)
            evt.GetEventObject().SetValue("")

    def get_test_kwargs(self):
        kwargs = dict()
        kwargs["x"] = self.factor.GetStringSelection()
        kwargs["c0"] = self.level1.GetValue()
        kwargs["c1"] = self.level2.GetValue()
        return kwargs

