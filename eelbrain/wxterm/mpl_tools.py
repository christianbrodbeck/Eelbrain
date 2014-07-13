import logging, tempfile, os

import matplotlib.pyplot as P
import wx

from .. import fmtxt
from .._utils import ui
from .._wxutils import Icon


class PyplotManager(wx.MiniFrame):
    copy_as_file = True  # copy as file or use mpl copy function
    def __init__(self, parent, pos=wx.DefaultPosition):
        wx.MiniFrame.__init__(self, parent, -1, "PyplotManager",
                              pos=pos,  # wx.Point
                              size=wx.Size(100, 500),
                              style=wx.DEFAULT_FRAME_STYLE)

        panel = self.panel = wx.Panel(self, -1)
#        sizer = self.sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer = self.sizer = wx.GridBagSizer(3, 2)
        panel.SetSizer(sizer)

#        button = wx.Button(panel, -1, "Update")
#        sizer.Add(button, 0, wx.EXPAND)
        button = wx.BitmapButton(panel, -1, Icon("tango/actions/view-refresh"))
        sizer.Add(button, (0, 0))
        button.Bind(wx.EVT_BUTTON, self.OnUpdate)

#        button = wx.Button(panel, -1, "Close All")
#        sizer.Add(button, 0, wx.EXPAND)
        button = wx.BitmapButton(panel, -1, Icon("tango/status/image-missing"))
        sizer.Add(button, (0, 1))
        button.Bind(wx.EVT_BUTTON, self.OnCloseAll)

        # ## SELECT
        ch = self.select = wx.Choice(panel, -1, size=(100, -1),
                                     choices=["Select"],
                                     name="Select")
        sizer.Add(ch, (1, 0), (1, 2))
        ch.Bind(wx.EVT_LEFT_DOWN, self.update_copy_fignums)
        ch.Bind(wx.EVT_CHOICE, self.OnFigureSelect)
        # ##

        # ## COPY
        ch = self.copy = wx.Choice(panel, -1, size=(100, -1),
                                   choices=["Copy"],
                                   name="Copy")
        sizer.Add(ch, (2, 0), (1, 2))
        ch.Bind(wx.EVT_LEFT_DOWN, self.update_copy_fignums)
        ch.Bind(wx.EVT_CHOICE, self.OnFigureSelect)
        # ##

        button = wx.BitmapButton(panel, -1, Icon("copy/textab"))
        sizer.Add(button, (3, 0))
        button.Bind(wx.EVT_BUTTON, self.OnCopyTextab)


        sizer.Fit(panel)
        self.Bind(wx.EVT_CLOSE, self.OnCloseWindow)

        x, y = sizer.GetMinSize()
        x += 10
        y += 20
        self.SetSize((x, y))  # x was 100


#    def figure(self, num=None, **kwargs):
#        """
#        could provide function to the shell to create a customized figure
#        frame in the future
#
#        """
#        if num is None:
#            num = 1
#            used = self.figures.keys()
#            while num in used:
#                num += 1
#        else:
#            if num in self.figures:
#                return self.figures[num]
#        # if not found create new figure
#        fig = self.P_figure(num, **kwargs)
#        self.figures[num] = fig
#
#        self.update_figurelist()
#        return fig

    def update_copy_fignums(self, event):
        sender = event.GetEventObject()
        allnums = [str(f.num) for f in P._pylab_helpers.Gcf.get_all_fig_managers()]
        sender.SetItems([sender.Name] + allnums)
        event.Skip()

    def OnFigureSelect(self, event):
        sender = event.GetEventObject()
        choice = event.GetString()
        if choice.isdigit():
            num = int(choice)
            if sender == self.select:
                P.figure(num)
            elif sender == self.copy:
                fig = P.figure(num)
                if self.copy_as_file:
                    if wx.TheClipboard.Open():
                        # save to temp file
                        fd, path = tempfile.mkstemp('.pdf', text=True)
#                        os.write(fd, pdf)
                        os.close(fd)
                        logging.debug("Temporary file created at: %s" % path)
                        fig.savefig(path)
                        # copy path
            #            shutil.copyfileobj(f, 'bar.txt')
                        do = wx.FileDataObject()
                        do.AddFile(path)
                        wx.TheClipboard.SetData(do)
                        wx.TheClipboard.Close()


                else:
                    fig.set_facecolor((1, 1, 1))
                    P.draw()
                    canvas = fig.canvas
                    if hasattr(canvas, 'Copy_to_Clipboard'):
                        canvas.Copy_to_Clipboard()
                    else:
                        logging.warning("Could Not Copy Figure %s To ClipBoard" % num)
            else:
                raise ValueError("URK!")
            sender.SetSelection(0)

    def OnUpdate(self, event):
        P.draw()
        if P.get_backend() == 'WXAgg':
            P.show()

    def OnCopyTextab(self, event=None):
        try:
            fmtxt.copy_pdf()
        except Exception as e:
            ui.message("Error in Copy Tex", str(e), '!')

    def OnCloseAll(self, event):
        "Close all open figures"
        self.Parent.CloseAllPlots()

    def OnCloseWindow(self, event):
        "Hides the window instead of destroying it"
        logging.debug("P_Mgr.OnCloseWindow()")
        self.Show(False)
#        self.Destroy()
