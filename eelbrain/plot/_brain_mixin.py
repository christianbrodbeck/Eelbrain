# -*- coding: utf-8 -*-
# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from matplotlib.image import imsave
from matplotlib.colors import ListedColormap

from ..fmtxt import Image
from ._colors import ColorBar


class BrainMixin(object):
    "Additions to "
    # Mixin that adds Eelbrain functionality to the pysurfer Brain class,
    # defined in a separate module so that documentation can be built without
    # importing Mayavi
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

    def plot_colorbar(self, *args, **kwargs):
        """Plot a colorbar corresponding to the displayed data

        Parameters
        ----------
        label : None | str
            Label for the x-axis (default is the name of the colormap).
        label_position : 'left' | 'right' | 'top' | 'bottom'
            Position of the axis label. Valid values depend on orientation.
        clipmin : scalar
            Clip the color-bar below this value.
        clipmax : scalar
            Clip the color-bar above this value.
        orientation : 'horizontal' | 'vertical'
            Orientation of the bar (default is horizontal).
        unit : str
            Unit for the axis to determine tick labels (for example, ``u'µV'`` to
            label 0.000001 as '1').

        Returns
        -------
        colorbar : plot.ColorBar
            ColorBar plot object.
        """
        if self._hemi in ('both', 'split', 'lh'):
            data = self.data_dict['lh']
        elif self._hemi == 'rh':
            data = self.data_dict[self._hemi]
        else:
            raise RuntimeError("Brain._hemi=%s" % repr(self._hemi))
        cmap = ListedColormap(data['orig_ctable'] / 255., "??")
        return ColorBar(cmap, data['fmin'], data['fmax'], *args, **kwargs)

    def save_image(self, filename, transparent=True):
        """Save current image to disk

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

