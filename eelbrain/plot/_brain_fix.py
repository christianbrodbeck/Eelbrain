# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Fix up surfer.Brain"""
from matplotlib.image import imsave
import surfer

from ..fmtxt import Image


class Brain(surfer.Brain):

    def image(self, name, format='png', alt=None):
        """Create an FMText Image from a screenshot

        Parameters
        ----------
        name : str
            Name for the file (without extension; default is 'image').
        format : str
            File format (default 'png').
        alt : None | str
            Alternate text, placeholder in case the image can not be found
            (HTML `alt` tag).
        """
        im = self.screenshot('rgba', True)
        return Image.from_array(im, name, format, alt)

    def save_image(self, filename, transparent=True):
        """Save image to disk

        Parameters
        ----------
        filename: str
            Destination for the image file (format is inferred from extension).
        transparent : bool
            Whether to make background transparent (default True).
        """
        if transparent:
            mode = 'rgba'
        else:
            mode = 'rgb'
        im = self.screenshot(mode)
        imsave(filename, im)
