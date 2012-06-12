""" 
Current
=======

eelbrain plottig routines



Types of plotting functions:

   ...     creates new figure
   _ax_...      plot to ax kwarg if provided, otherwise to P.gca()
   _row_...(row=?, nrows=?)     fill a whole row
 ? col_...


PLANS
=====

Create general superclass for figures which has
- wx interface with toolbar
- allows for management of spacing

"""

import matplotlib.pyplot as _plt

from eelbrain import vessels as _vsl
from eelbrain.vessels import data as _dt



defaults={
          'figsize':(6,-1), # was 7
        }
title_kwargs = {'size': 18,
                'family': 'serif'}
figs = [] # store callback figures (they need to be preserved)


def unpack_epochs_arg(ndvars, ndim, dataset=None, levels=1):
    """
    Returns a nested list of epochs (through the summary method)
    """
    ndvars = getattr(ndvars, '_default_plot_obj', ndvars)
    if not isinstance(ndvars, (tuple, list)):
        ndvars = [ndvars]

    if levels > 0:
        return [unpack_epochs_arg(v, ndim, dataset, levels-1) for v in ndvars]
    else:
        out = []
        for ndvar in ndvars:
            if isinstance(ndvar, basestring):
                ndvar = dataset[ndvar]
            
            if ndvar.ndim == ndim + 1:
                if ndvar.dims[0] is 'case':
                    ndvar = ndvar.summary()
            
            if ndvar.ndim == ndim:
                pass
            else:
                err = ("Plot requires ndim=%i; %r ndim==%i" % 
                       (ndim, ndvar, ndvar.ndim))
                raise _dt.DimensionMismatchError(err)
            out.append(ndvar)
        return out



def read_cs_arg(epoch, colorspace=None):
    if (colorspace is None):# and ('colorspace' in epoch.properties):
#        colorspace = epoch.properties['colorspace']
        colorspace = epoch.properties.get('colorspace', 
                                          _vsl.colorspaces.get_default())
    return colorspace
        



class CallbackFigure:
    "cf. wxutils.mpl_canvas"
    def create_figure(self, *args, **kwargs):
        "creates self.figure and self.canvas attributes and returns the figure"
        self.fig = _plt.figure(*args, **kwargs)
        self.canvas = self.fig.canvas
        figs.append(self)
        return self.fig
    
    def redraw_ax(self, *axes):
        "redraw an ax"
        canvas = self.canvas
        canvas.restore_region(self._background)
        for ax in axes:
            ax.draw_artist(ax)
            extent = ax.get_window_extent()
            canvas.blit(extent)
    
    def store_canvas(self):
        self._background = self.canvas.copy_from_bbox(self.fig.bbox)


# MARK: figure composition

def _loc(name, size=(0,0), title_space=0, frame=.01):
    """
    takes a loc argument and returns x,y of bottom left edge
    
    """
    if isinstance(name, basestring):
        y,x = name.split()
    # interpret x
    elif len(name) == 2:
        x,y = name
    else:
        raise NotImplementedError("loc needs to be string or len=2 tuple/list")
    if isinstance(x, basestring):
        if x=='left':
            x = frame
        elif x in ['middle', 'center', 'centre']:
            x = .5 - size[0] / 2
        elif x == 'right':
            x = 1 - frame - size[0]
        else:
            raise ValueError(x)
    # interpret y
    if isinstance(y, basestring):
        if y in ['top', 'upper']:
            y = 1 - frame - title_space - size[1]
        elif y in ['middle', 'center', 'centre']:
            y = .5 - title_space/2. - size[1]/2.
        elif y in ['lower', 'bottom']:
            y = frame
        else: 
            raise ValueError(y)
    return x, y



