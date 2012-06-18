'''
Created on Feb 26, 2012

@author: christian
'''


import wx

import icons



# store icons once loaded for repeated access
cache = {}
iconcache = {}

def Icon(path, asicon=False):#, size=None):
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

