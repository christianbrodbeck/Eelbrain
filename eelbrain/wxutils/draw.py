'''
Created on Apr 22, 2013

@author: Christian M Brodbeck
'''
import numpy as np
from scipy.io import savemat
import wx



def draw_text(text, face='Scheherazade', size=42, spo2=True, w=None, h=None):
    """
    Use wxPython's text rendering engine to convert unicode text to an image
    in the form of a numpy array.

    Parameters
    ----------
    text : unicode
        The text.
    face : str
        The font name.
    size : int
        Font size.
    spo2 : bool
        If the shape is determined automatically (with shape=None), use a
        Square with a side length that its a Power Of 2.
    w, h : None | int
        Specify the desired width and/or height for the output array.
        If None, the shape is determined based on the text size.
    """
    if (w is not None or h is not None) and spo2:
        err = ("spo2 can not be true when providing custom width and/ir "
               "height values.")
        raise TypeError(err)

    font = wx.Font(size, family=wx.FONTFAMILY_UNKNOWN, style=wx.FONTSTYLE_NORMAL,
                   weight=wx.FONTWEIGHT_NORMAL, face=face)

    dc = wx.MemoryDC()
    dc.SetFont(font)
    res = 512
    dc.SelectObject(wx.EmptyBitmap(res, res))

    tw, th = dc.GetTextExtent(text)
    if w is None:
        w = tw + 2
    if h is None:
        h = th + 2

    if spo2:
        x = 2
        while (x < w) or (x < h):
            x *= 2
        w, h = x, x
    elif tw > w or th > h:
        err = ("Text dimensions (w=%i, h=%i) exceed image resolution "
               "(w=%i, h=%i)." % (tw, th, w, h))
        raise ValueError(err)

    bmp = wx.EmptyBitmap(w, h)
    dc.SelectObject(bmp)
    dc.Clear()

    dc.DrawText(text, (w - tw) / 2, (h - th) / 2)

    im = bmp.ConvertToImage()
    a = np.fromstring(im.GetData(), dtype='uint8').reshape((h, w, 3))
    a = 255 - a[:, :, 0]
    return a



class UnicodeImages(object):
    """
    Build a dictionary with image arrays depicting unicode strings, ensuring
    the same string is not stored twice.
    """
    def __init__(self, face='Scheherazade', size=42, spo2=True):
        """
        Parameters
        ----------
        draw_text parameters
        """
        self._arrays = {}
        self._codes = {}
        self._face = face
        self._size = size
        self._spo2 = spo2

    def code(self, text):
        """
        Retrieve the code for the text.

        Parameters
        ----------
        text : unicode
            The unicode string to be converted to an image.

        Returns
        -------
        code : int
            The code assigned to the unicode object.
        """
        if text in self._codes:
            code = self._codes[text]
        else:
            im = draw_text(text, face=self._face, size=self._size,
                           spo2=self._spo2)
            code = len(self._codes)
            str_code = 'i%i' % code
            self._arrays[str_code] = im
            self._codes[text] = code

        return code

    def save(self, fname):
        "Save as matlab *.m file."
        savemat(fname, self._arrays, do_compression=True)
