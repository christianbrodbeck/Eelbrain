## Author: Proloy Das <proloy@umd.edu>
import glob
from os.path import join

import mne
import numpy as np

from .._data_obj import NDVar, SourceSpace, UTS
from .. import plot
from ..plot._base import TimeSlicer


class VoxelLayer(object):

    def __init__(self, ndvar, res=10):
        from mayavi import mlab

        sss = ndvar.source.get_source_space()
        ss = sss[0]
        shape = [ss['shape'][i] for i in [2, 1, 0]]
        x = np.zeros(shape)
        xr = x.ravel()
        idx = ss['vertno']
        xr[idx] = ndvar.get_data('source')

        from scipy.ndimage import zoom
        x = zoom(x, (res, res, res))

        x = x.swapaxes(0, 2)
        vmax = max(x.max(), -x.min())
        self.source = mlab.pipeline.scalar_field(x)
        self.volume = mlab.pipeline.volume(self.source, vmin=0, vmax=vmax)

        dist = ndvar.source.grade * 0.001 / res
        offset = [-p / 2 * dist for p in x.shape]
        offset[1] -= .02
        self.volume.volume.position = offset
        self.volume.volume.scale = [dist] * 3

    def update(self, ndvar):
        pass


class VectorLayer(object):

    def __init__(self, ndvar):
        from mayavi import mlab

        x, y, z = ndvar.source.coordinates.T
        u, v, w = ndvar.get_data(('space', 'source'))
        self.q = mlab.quiver3d(
            x, y, z, u, v, w,
            scale_factor=0.4,  # length of the arrows
            mode='arrow'
        )

    def update(self, ndvar):
        q_array = self.q.mlab_source.dataset.point_data.get_array(0)
        q_array.from_array(ndvar.get_data(('source', 'space')))
        self.q.mlab_source.update()


class VolBrain(TimeSlicer):

    def __init__(self, ndvar, src=False, bem=True, surf='pial'):
        from mayavi import mlab

        self.mlab = mlab
        self.figure = mlab.figure(bgcolor=(1, 1, 1))
        # BEM
        if bem:
            bem_temp = join(ndvar.source.subjects_dir, ndvar.source.subject, 'bem',
                            '%s-*-bem.fif' % ndvar.source.subject)
            bem_file = glob.glob(bem_temp)[0]
            bem = mne.read_bem_surfaces(bem_file)[0]
            x, y, z = bem['rr'].T
            tris = bem['tris'][:, ::-1]
            self.bem = mlab.triangular_mesh(x, y, z, tris, #opacity=0.25,
                                            color=(1, 1, 1))
            self.bem.actor.property.backface_culling = True
        else:
            self.bem = None
        # surf
        self.surfs = []
        if surf:
            surf_temp = join(ndvar.source.subjects_dir, ndvar.source.subject,
                             'surf', '%%s.%s' % surf)
            for hemi in ('lh', 'rh'):
                rr, tris = mne.read_surface(surf_temp % hemi)
                rr /= 1000
                x, y, z = rr.T
                obj = mlab.triangular_mesh(x, y, z, tris, color=(1, 1, 1), opacity=1,
                                           representation='wireframe')
                obj.actor.property.backface_culling = True
                self.surfs.append(obj)
        if src:
            ss = SourceSpace.from_file(ndvar.source.subjects_dir, ndvar.source.subject, src, None)
            x, y, z = ss.coordinates.T
            src = mlab.pipeline.scalar_scatter(x, y, z, figure=self.figure)
            src.mlab_source.dataset.lines = ss.connectivity()
            lines = mlab.pipeline.stripper(src)
            mlab.pipeline.surface(lines,
                                  color=(0, 0, 0),
                                  # colormap='Accent',
                                  line_width=0.1,
                                  opacity=0.25,
                                  figure=self.figure)

        if ndvar.has_dim('time'):
            self.time = ndvar.get_dim('time')
            ndvar0 = ndvar.sub(time=self.time[0])
        else:
            self.time = None
            ndvar0 = ndvar

        if ndvar.has_dim('space'):
            self.p = VectorLayer(ndvar0)
        else:
            self.p = VoxelLayer(ndvar0)

        mlab.view(distance=.3)
        self._ndvar = ndvar

        self.figure.scene.camera.parallel_scale = .08
        self.figure.scene.camera.parallel_projection = True
        self.figure.render()
        TimeSlicer.__init__(self, (ndvar,))

    def _update_time(self, t, fixate):
        self.p.update(self._ndvar.sub(time=t))

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


def butterfly(ndvar, src=False, bem=True, surf='pial', vmin=None, vmax=None):
    if ndvar.has_dim ('space'):
        p = plot.Butterfly (ndvar.norm ('space'), vmin=vmin, vmax=vmax)
    else:
        raise TypeError("ndvar need to have space dimension.")

    vb = VolBrain(ndvar, src=src, bem=bem, surf=surf)
    p.link_time_axis (vb)

    return p, vb


# if __name__ == '__main__':
#     e = cssl.experiments.get('hayo')
#
#     # # single subject
#     # res = e.load_trf('audspec-1', **OPTIONS, make=True)
#     # h = res.h.smooth('time', 0.05)
#     #
#     # # surface comparison
#     # sres = e.load_trf('audspec-1', **SURF_OPTIONS, make=True)
#     # sh = sres.h.smooth('time', 0.05)
#     # plot.brain.butterfly(sh)
#
#     # group
#     ds = e.load_trfs('young', 'audspec-1', **OPTIONS, make=True, smooth=.05)
#     dss = e.load_trfs('young', 'audspec-1$fliphalves', **OPTIONS, make=True,
#                       smooth=.05)
#
#     # r
#     x = (ds['r'] - dss['r']).norm('space')
#     res = testnd.ttest_1samp(x, samples=10000, tfce=True, tail=1)
#     mean = ds['r'].mean('case').norm('space')
#     y = mean * (res.p <= .05)
#     # y = y.smooth('source', .05, 'gaussian')
#     b = VolBrain(y)
#
#     # h
#     # h = ds['audspec_1'].mean('case')
#     # p = plot.Butterfly(h.norm('space'))
#     # b = VolBrain(h)
#     # p.link_time_axis(b)
