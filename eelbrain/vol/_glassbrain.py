# Author: Proloy Das <proloy@umd.edu>
import glob
from os.path import join

import numpy as np
from nilearn.plotting import plot_glass_brain
import matplotlib
import mne

from .._data_obj import NDVar, SourceSpace, UTS
from .. import plot
from ..plot._base import TimeSlicer
from ._nifti_utils import _save_stc_as_volume


class GlassBrain(TimeSlicer):

    def __init__(self, ndvar, src, dest='mri', mri_resolution=False, black_bg=False, display_mode='lyrz',
                 threshold='auto', colorbar=False, cmap=None,alpha=0.7, vmin=None, vmax=None, plot_abs=True):

        if not matplotlib.is_interactive():
            print('Turning interactive backend on.')
            matplotlib.interactive(True)


        # src
        # check if file name
        if isinstance (src, str):
            print(('Reading src file %s...' % src))
            self.src = mne.read_source_spaces(src)
        else:
            self.src = src

        src_type = src[0]['type']
        if src_type != 'vol':
            raise ValueError('You need a volume source space. Got type: %s.'
                              % src_type)

        if ndvar.has_dim ('space'):
            ndvar = ndvar.norm ('space')
            # set vmax and vmin
            if vmax is None:
                vmax = ndvar.max ()
            if vmin is None:
                vmin = ndvar.min ()
        else:
            # set vmax and vmin
            if vmax is None:
                vmax = np.maximum (ndvar.max (), -ndvar.min ())
            if vmin is None:
                vmin = np.minimum (-ndvar.max (), ndvar.min ())

        self._ndvar = ndvar

        if ndvar.has_dim('time'):
            t_in = 0
            self.time = ndvar.get_dim('time')
            ndvar0 = ndvar.sub(time=self.time[t_in])
            title = 'time = %s ms' % round(t_in*1e3)
        else:
            self.time = None
            title = 'time = None'
            ndvar0 = ndvar

        self.kwargs0 = dict(dest=dest,
                            mri_resolution=mri_resolution)

        self.kwargs1 = dict(black_bg=black_bg,
                            display_mode=display_mode,
                            threshold=threshold,
                            cmap=cmap,
                            colorbar=colorbar,
                            alpha=alpha,
                            vmin=vmin,
                            vmax=vmax,
                            plot_abs=plot_abs,
                            )
        self.glassbrain = plot_glass_brain(_save_stc_as_volume(None, ndvar0, self.src, **self.kwargs0),
                                           title=title,
                                           **self.kwargs1
                                           )
        TimeSlicer.__init__(self, (ndvar,))

    def _update_time(self, t, fixate):
        ndvart = self._ndvar.sub(time=t)
        title = 'time = %s ms' % round (t * 1e3)

        # remove existing image
        for display_ax in list(self.glassbrain.axes.values()):
            if len(display_ax.ax.images) > 1:
                display_ax.ax.images[-1].remove()

        # No need to take care of the colorbar anymore
        # Still thete is some bug!
        if self.kwargs1['colorbar']:
            self.glassbrain._colorbar_ax.axes.remove()
            self.glassbrain._colorbar = False

        self.glassbrain.add_overlay(_save_stc_as_volume(None, ndvart, self.src, **self.kwargs0),
                                    threshold=self.kwargs1['threshold'],
                                    colorbar=self.kwargs1['colorbar'],
                                    **dict(cmap=self.kwargs1['cmap'],
                                           # norm=self.kwargs1['norm'],
                                           vmax=self.kwargs1['vmax'],
                                           vmin=self.kwargs1['vmin'],
                                           alpha=self.kwargs1['alpha'],))
        self.glassbrain.title(title)
        # update colorbar
        # if self.kwargs1['colorbar']:
        #     self._update_colorbar(None, None)

    def animate(self):
        for t in self.time:
            self.set_time(t)

    # this is only needed for Eelbrain < 0.28
    def set_time(self, time):
        """Set the time point to display

        Parameters
        ----------
        time : scalar
            Time to display.
        """
        self._set_time(time, True)


def butterfly(ndvar, src, dest='mri', mri_resolution=False, black_bg=False, display_mode='lyrz',
                 threshold='auto', cmap=None, colorbar=False, alpha=0.7, vmin=None, vmax=None, plot_abs=True):

    if ndvar.has_dim('space'):
        p = plot.Butterfly(ndvar.norm('space'), vmin=vmin, vmax=vmax)
    else:
        p = plot.Butterfly(ndvar, vmin=vmin, vmax=vmax)

    gb = GlassBrain(ndvar, src, dest=dest, mri_resolution=mri_resolution, black_bg=black_bg, display_mode=display_mode,
                    threshold=threshold, cmap=cmap, colorbar=colorbar, alpha=alpha, vmin=vmin, vmax=vmax, plot_abs=True)

    p.link_time_axis(gb)

    return p, gb


if __name__ == '__main__':
    import pickle

    ROOTDIR = 'G:/My Drive/Proloy/'

    fname = ROOTDIR + '/mri/fsaverage/bem/fsaverage-vol-10-src.fif'
    src = mne.read_source_spaces(fname)

    fname = ROOTDIR + 'Group analysis/Dataset wf-onset-u.pickled'
    ds = pickle.load(open(fname, 'rb'))
    h = ds['trf'].mean('case')
    gb = butterfly(h, src, dest='surf', threshold=10)
    p = plot.Butterfly(h.norm('space'))
    gb = GlassBrain(h, src, dest='surf', threshold=5e-13)
    p.link_time_axis(gb)


