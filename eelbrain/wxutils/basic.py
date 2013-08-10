'''Some WxPython utilities'''
# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import wx
from wx.lib.dialogs import ScrolledMessageDialog

from . import icons

__all__ = ['Icon', 'key_mod', 'show_text_dialog']


# store icons once loaded for repeated access
cache = {}
iconcache = {}

def Icon(path, asicon=False):  # , size=None):
    if asicon:
        if path not in iconcache:
            iconcache[path] = icons.catalog[path].GetIcon()
        return iconcache[path]
    else:
        if path not in cache:
            cache[path] = icons.catalog[path].GetBitmap()
        return cache[path]


def key_mod(event):
    """
    takes KeyPressEvent, returns
    mac:    [control, command(meta), alt]
    else:   [control, 0, alt]

    """
    meta_down = event.MetaDown()
    control_down = event.ControlDown()
    alt_down = event.AltDown()
    return [control_down, meta_down, alt_down]
    # FIXME: platform-specific modifier key handling:
#    cmd = event.CmdDown() # Control for PC and Unix, Command on Macs
#    if "wxMac" in wx.PlatformInfo:
#        mod = [control_down, cmd, alt_down]
#    else:
#        mod = [control_down, False, alt_down]
#    return mod


def show_text_dialog(parent, text, caption):
    "Create and show a ScrolledMessageDialog"
    style = wx.CAPTION | wx.CLOSE_BOX | wx.RESIZE_BORDER | wx.SYSTEM_MENU
    dlg = ScrolledMessageDialog(parent, text, caption, style=style)
    font = wx.Font(12, wx.MODERN, wx.NORMAL, wx.NORMAL, False, u'Inconsolata')
    dlg.text.SetFont(font)

    n_lines = dlg.text.GetNumberOfLines()
    line_text = dlg.text.GetLineText(0)
    w, h = dlg.text.GetTextExtent(line_text)
    dlg.text.SetSize((w + 100, (h + 3) * n_lines + 50))

    dlg.Fit()
    dlg.Show()
    return dlg
