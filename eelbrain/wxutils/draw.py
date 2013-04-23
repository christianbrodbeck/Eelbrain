'''
Created on Apr 22, 2013

@author: Christian M Brodbeck
'''
import numpy as np
import wx



def draw_text(text, face='Scheherazade', w=128, h=128):
    """
    Use wxPython's text rendering engine to convert unicode text to an image
    in the form of a numpy array.

    Parameters
    ----------
    text : unicode
        The text.
    face : str
        The font name.
    w, h : int
        The desired width and height of the array.
    """
    bmp = wx.EmptyBitmap(w, h)
    dc = wx.MemoryDC()
    dc.SelectObject(bmp)
    dc.Clear()

    font = wx.Font(42, family=wx.FONTFAMILY_UNKNOWN, style=wx.FONTSTYLE_NORMAL,
                   weight=wx.FONTWEIGHT_NORMAL, face=face)
    dc.SetFont(font)

    tw, th = dc.GetTextExtent(text)

    if tw > w or th > h:
        err = ("Text dimensions (w=%i, h=%i) exceed image resolution "
               "(w=%i, h=%i)." % (tw, th, w, h))
        raise ValueError(err)

    dc.DrawText(text, (w - tw) / 2, (h - th) / 2)
    dc.SelectObject(wx.NullBitmap)

    im = bmp.ConvertToImage()
    a = np.fromstring(im.GetData(), dtype='uint8').reshape((w, h, 3))
    a = 255 - a[:, :, 0]
    return a
