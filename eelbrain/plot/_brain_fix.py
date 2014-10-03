# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Fix up surfer.Brain"""
from matplotlib.image import imsave
import surfer


class Brain(surfer.Brain):

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
