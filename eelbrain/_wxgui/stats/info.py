import wx

from .utils import TitleSizer, FactorPanel, ValidationException


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
        # create menu to select subjects for analysis
        self.subj_sel = SubjectsSelector(self, list(set(self.ds["subject"])))
        # three columns
        col1 = wx.BoxSizer(wx.VERTICAL)
        col2 = wx.BoxSizer(wx.VERTICAL)
        col3 = wx.BoxSizer(wx.VERTICAL)
        # add subject menu and title to first column
        col1.Add(TitleSizer(self, "Subjects"), 0, wx.BOTTOM, 5)
        col1.Add(self.subj_sel)
        # create sizer for experiment design (factors/levels)
        self.factor_sizer = wx.BoxSizer(wx.HORIZONTAL)
        for i, level_names in enumerate(self.loader.levels):
            panel = FactorPanel(self, level_names, i, editable=False)
            panel.factor_ctl.SetLabel(self.loader.factors[i])
            panel.factor_ctl.SetFont(wx.Font.Bold(panel.factor_ctl.GetFont()))
            self.factor_sizer.Add(panel, 0, wx.EXPAND | wx.RIGHT, 30)
        # add design and title to second column
        col2.Add(TitleSizer(self, "Design"), 0, wx.BOTTOM, 5)
        col2.Add(self.factor_sizer)
        # derive brain data descriptions from source NDVar
        src = self.ds["src"]
        n_src = src.source.vertices[0].shape[0] + src.source.vertices[0].shape[0]
        src_label = "{} sources".format(n_src)
        time_label = "{} to {} ms, step={} ms".format(
            int(src.time.times[0] * 1000),
            int(src.time.tstop * 1000), int(src.time.tstep * 1000)
        )
        # add brain data description and title to third column
        col3.Add(TitleSizer(self, "Data"), 0, wx.BOTTOM, 5)
        col3.Add(wx.StaticText(self, label=src_label))
        col3.Add(wx.StaticText(self, label=time_label))
        # add columns to sizer
        self.sizer.Add(col1, **self.add_params)
        self.sizer.Add(col2, **self.add_params)
        self.sizer.Add(col3, **self.add_params)
        self.sizer.Layout()
        self.SetSizer(self.sizer)

    def validate(self):
        subjects = self.subj_sel.get_selected_subjects()
        if len(subjects) == 0:
            raise ValidationException("You must select subjects for analysis.")
        return True

