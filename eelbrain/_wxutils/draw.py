'''
Created on Apr 22, 2013

@author: Christian M Brodbeck
'''
import numpy as np
from scipy.io import savemat
import wx

from .._wxgui import get_app


def draw_text(text, face='Scheherazade', size=42, spo2=False, w=None, h=None,
              fg=None, bg=None):
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
    fg : 0-255 RGB tuple
        Text foreground color.
    bg : 0-255 RGB tuple
        Text background color.

    Returns
    -------
    array : (h, w, 3) | (h, w)
        If fg and/or bg are specified an RGB array, otherwise a monochromatic
        array.
    """
    if (w is not None or h is not None) and spo2:
        err = ("spo2 can not be true when providing custom width and/ir "
               "height values.")
        raise TypeError(err)
    mono = not (fg or bg)

    get_app()
    font = wx.Font(size, family=wx.FONTFAMILY_UNKNOWN, style=wx.FONTSTYLE_NORMAL,
                   weight=wx.FONTWEIGHT_NORMAL, face=face)

    # device context for drawing
    dc = wx.MemoryDC()
    dc.SetFont(font)

    # determine scales using temporary bitmap
    res = 512
    dc.SelectObject(wx.EmptyBitmap(res, res))
    tw, th = dc.GetTextExtent(text)
    if w is None:
        w = tw
    if h is None:
        h = th

    if spo2:
        x = 2
        while (x < w) or (x < h):
            x *= 2
        w, h = x, x
    elif tw > w or th > h:
        err = ("Text dimensions (w=%i, h=%i) exceed image resolution "
               "(w=%i, h=%i)." % (tw, th, w, h))
        raise ValueError(err)

    # draw in nea bitmap
    bmp = wx.EmptyBitmap(w, h)
    dc.SelectObject(bmp)
    if fg is not None:
        dc.SetTextForeground(fg)
    if bg is not None:
        dc.SetTextBackground(bg)
        brush = wx.Brush(bg, wx.SOLID)
        dc.SetBackground(brush)
    dc.Clear()

    dc.DrawText(text, (w - tw) / 2, (h - th) / 2)

    im = bmp.ConvertToImage()
    a = np.fromstring(im.GetData(), dtype='uint8').reshape((h, w, 3))
    if mono:
        return 255 - a[:, :, 0]
    else:
        return a


def draw_paragraph(lines, face='Scheherazade', size=42, align='center'):
    """
    Parameters
    ----------
    lines : iterator
        Iterator over unicode strings, each of which will be added as a
        separate line.
    face : str
        The font name.
    size : int
        Font size.
    align : 'left' | 'center' | 'right'
        How to align lines
    """
    ims = [draw_text(line, face=face, size=size, spo2=False) for line in lines]
    shape = np.array([im.shape for im in ims])
    h = np.sum(shape[:, 0])
    w = np.max(shape[:, 1])
    out = np.zeros((h, w), dtype='uint8')
    y = 0
    for im in ims:
        im_h, im_w = im.shape
        y0 = y
        y += im_h
        if align == 'left':
            x = 0
            x1 = im_w
        elif align == 'center':
            x = (w - im_w) // 2
            x1 = x + im_w
        elif align == 'right':
            x = w - im_w
            x1 = w
        else:
            raise ValueError("align=%r" % align)

        out[y0:y, x:x1] = im

    return out


class UnicodeImages(object):
    """
    Build a dictionary with image arrays depicting unicode strings, ensuring
    the same string is not stored twice.
    """
    def __init__(self, face='Scheherazade', size=42, spo2=False):
        """
        Parameters
        ----------
        draw_text parameters
        """
        self.arrays = {}
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
            self.arrays[str_code] = im
            self._codes[text] = code

        return code

    def save(self, fname):
        "Save as matlab *.m file."
        savemat(fname, self.arrays, do_compression=True)
