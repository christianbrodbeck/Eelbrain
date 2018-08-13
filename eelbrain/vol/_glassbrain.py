# Author: Proloy Das <proloy@umd.edu>
import glob
from os.path import join

from .. import plot
from ..plot._base import TimeSlicer
from ._nifti_utils import _save_stc_as_volume


class GlassBrain(TimeSlicer):
    def __init__(self, ndvar, dest='mri', mri_resolution=False, black_bg=False, display_mode='lyrz',
                 threshold='auto', colorbar=False, cmap=None,alpha=0.7, vmin=None, vmax=None, plot_abs=True):
        import matplotlib
        if not matplotlib.is_interactive():
            print ('Turning interactive backend on.')
            matplotlib.interactive (True)

        from nilearn.plotting import plot_glass_brain, show
        self.show = show

        # mne src
        from mne import read_source_spaces
        self.src = read_source_spaces( join(ndvar.source.subjects_dir, ndvar.source.subject, 'bem',
                                            '%s-%s-src.fif'%(ndvar.source.subject, ndvar.source.src)))

        src_type = self.src[0]['type']
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
                vmax = max([ndvar.max (), -ndvar.min ()])
            if vmin is None:
                vmin = min([-ndvar.max (), ndvar.min ()])

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

        show()

    # Used by _update_axes to show the time
    def _update_title(self, t):
        first_axis = self.glassbrain._cut_displayed[0]
        ax = self.glassbrain.axes[first_axis].ax
        ax.texts[-1].set_text('time = %s ms' % round(t * 1e3))

    # used by _update_time to redraw image
    def _update_axes(self, t, fixate):
        ndvart = self._ndvar.sub(time=t)

        # remove existing image
        for display_ax in list(self.glassbrain.axes.values()):
            if len(display_ax.ax.images) > 1:
                display_ax.ax.images[-1].remove()

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
        # Update title
        self._update_title(t)

        for axis in self.glassbrain._cut_displayed:
            self.glassbrain.axes[axis].ax.redraw_in_frame()

    def _update_time(self, t, fixate):
        self._update_axes(t, fixate)
        self.show()

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


def butterfly(ndvar, dest='mri', mri_resolution=False, black_bg=False, display_mode='lyrz',
                 threshold='auto', cmap=None, colorbar=False, alpha=0.7, vmin=None, vmax=None, plot_abs=True):

    if ndvar.has_dim('space'):
        p = plot.Butterfly(ndvar.norm('space'), vmin=vmin, vmax=vmax)
    else:
        p = plot.Butterfly(ndvar, vmin=vmin, vmax=vmax)

    gb = GlassBrain(ndvar, dest=dest, mri_resolution=mri_resolution, black_bg=black_bg, display_mode=display_mode,
                    threshold=threshold, cmap=cmap, colorbar=colorbar, alpha=alpha, vmin=vmin, vmax=vmax, plot_abs=True)

    p.link_time_axis(gb)

    return p, gb


# if __name__ == '__main__':
#     from eelbrain import *
#     import pickle
#     from mne import read_source_spaces
#
#     ROOTDIR = '/mnt/c/Users/proloy/csslz/'
#
#     fname = ROOTDIR + 'Group analysis/Dataset audspec-1.pickled'
#     with open (fname, 'rb') as f:
#         ds = pickle.load (f)
#     h = ds['audspec'].mean('case')
#     gb = vol.glassbrain.butterfly (h, black_bg=True, dest='surf', threshold=1e-13, cmap='hot', colorbar=True)


