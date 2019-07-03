import wx


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


class TitleSizer(wx.BoxSizer):
    def __init__(self, parent, title):
        super().__init__(wx.HORIZONTAL)
        self.ctl = wx.StaticText(parent, label=title)
        font = wx.Font(14, wx.DEFAULT, wx.NORMAL, wx.BOLD)
        self.ctl.SetFont(font)
        self.Add(self.ctl)


class FactorPanel(wx.Panel):
    """Panel to display a factor and its level names"""
    def __init__(self, parent, levels, idx, editable=True):
        super(FactorPanel, self).__init__(parent)
        if editable:
            self.factor_ctl = wx.TextCtrl(self, value="factor_%d" % idx)
        else:
            self.factor_ctl = wx.StaticText(self, label="factor_%d" % idx)
            self.factor_ctl.SetFont(wx.Font.Bold(self.factor_ctl.GetFont()))
        level_names = ["- " + i for i in levels]
        level_ctl = wx.StaticText(self)
        level_ctl.SetLabel("\n".join(level_names))
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.factor_ctl)
        sizer.Add(level_ctl, 0, wx.EXPAND | wx.TOP, 10)
        sizer.Layout()
        self.SetSizer(sizer)
