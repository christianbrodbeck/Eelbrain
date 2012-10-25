'''
Created on Sep 30, 2012

@author: christian
'''
import numpy as np
from numpy import sin, cos
from scipy.optimize import leastsq

import traits.api as traits
from traitsui.api import View, Item, HGroup
from tvtk.pyface.scene_editor import SceneEditor
from mayavi.tools.mlab_scene_model import MlabSceneModel
from mayavi.core.ui.mayavi_scene import MayaviScene

import mne

from eelbrain import load



class coreg(traits.HasTraits):
    """

    http://docs.enthought.com/mayavi/mayavi/building_applications.html#making-the-visualization-live
    """
    # views
    frontal = traits.Button()
    left = traits.Button()
    top = traits.Button()

    # visibility
    fiducials = traits.Bool(True)
    dig_points = traits.Bool(True)
    head_shape = traits.Bool(True)
    scalp = traits.Bool(True)
    scalp_alpha = traits.Range(0., 1., 1.)
    sensors = traits.Bool(True)

    scene = traits.Instance(MlabSceneModel, ())

    def __init__(self, raw, fwd, bem=None):
        """
        Parameters
        ----------

        raw : mne.fiff.Raw | str(path)
            MNE Raw object, or path to a raw file.
        fwd : dict | str(path)
            MNE forward solution (returned by mne.read_forward_solution),
            or path to an MNE forward solution.
        bem : None | list | str(path)
            Bem file for the scalp surface: list as returned by
            mne.read_bem_surfaces, or path to a bem file.

        """
        traits.HasTraits.__init__(self)

        if isinstance(raw, basestring):
            raw = load.fiff.Raw(raw)
        if isinstance(fwd, basestring):
            fwd = mne.read_forward_solution(fwd)
        if isinstance(bem, basestring):
            bem = mne.read_bem_surfaces(bem)

        points3d = self.scene.mlab.points3d

        dev_2_head = fwd['info']['dev_head_t']['trans']
        head_2_dev = np.linalg.inv(dev_2_head)
        mri_2_head = fwd['mri_head_t']['trans']

        # sensors
        s = load.fiff.sensor_net(raw)
        x, y, z = s.locs.T
        self._sensors = points3d(x, y, z, scale_factor=0.005, color=(0, .2, 1))

        # head shape
        pts = np.array([d['r'] for d in raw.info['dig']])
        pts = np.hstack((pts, np.ones((len(pts), 1))))
        x, y, z, _ = np.dot(head_2_dev, pts.T)
        pts = points3d(x, y, z, opacity=0)  # color=(1,0,0), scale_factor=0.005)
        d = self.scene.mlab.pipeline.delaunay3d(pts)
        self._head_shape = self.scene.mlab.pipeline.surface(d)

        # scalp (mri-headshape)
        if bem:
            surf = bem[0]
            pts = surf['rr']
            pts = np.hstack((pts, np.ones((len(pts), 1))))
            pts = np.dot(mri_2_head, pts.T)
            pts = np.dot(head_2_dev, pts)
            x, y, z, _ = pts
            faces = surf['tris']
            self._bem = self.scene.mlab.triangular_mesh(x, y, z, faces, color=(.8, .8, .8), opacity=1)

        # fiducials
        pts = filter(lambda d: d['kind'] == 1, raw.info['dig'])
        pts = np.vstack([d['r'] for d in pts])
        pts = np.hstack((pts, np.ones((len(pts), 1))))
        pts = np.dot(head_2_dev, pts.T)
        x, y, z, _ = pts
        self._fiducials = points3d(x, y, z, color=(0, 1, 1), opacity=0.5)

        # dig points
        pts = filter(lambda d: d['kind'] == 2, raw.info['dig'])
        pts = np.vstack([d['r'] for d in pts])
        pts = np.hstack((pts, np.ones((len(pts), 1))))
        pts = np.dot(head_2_dev, pts.T)
        x, y, z, _ = pts
        self._dig_pts = points3d(x, y, z, color=(1, 0, 0), opacity=0.5)

        self.configure_traits()
        self.frontal = True
        if bem:
            self.head_shape = False

    @traits.on_trait_change('dig_points')
    def _show_dig_pts(self):
        self._dig_pts.actor.visible = self.dig_points

    @traits.on_trait_change('fiducials')
    def _show_fiducials(self):
        self._fiducials.actor.visible = self.fiducials

    @traits.on_trait_change('head_shape')
    def _show_hs(self):
        self._head_shape.actor.visible = self.head_shape

    @traits.on_trait_change('scalp')
    def _show_bem(self):
        if not hasattr(self, '_bem'):
            return
        self._bem.actor.visible = self.scalp

    @traits.on_trait_change('sensors')
    def _show_sensors(self):
        self._sensors.visible = self.sensors

    @traits.on_trait_change('scalp_alpha')
    def _set_scalp_alpha(self):
        self._bem.actor.property.opacity = self.scalp_alpha

    @traits.on_trait_change('frontal')
    def _view_frontal(self):
        self.set_view('frontal')

    @traits.on_trait_change('left')
    def _view_left(self):
        self.set_view('left')

    @traits.on_trait_change('top')
    def _view_top(self):
        self.set_view('top')

    def set_view(self, view='frontal'):
        self.scene.parallel_projection = True
        self.scene.camera.parallel_scale = .15
        kwargs = dict(azimuth=90, elevation=90, distance=None, roll=180, reset_roll=True)
        if view == 'left':
            kwargs.update(azimuth=180, roll=90)
        elif view == 'top':
            kwargs.update(elevation=0)
        self.scene.mlab.view(**kwargs)

    # the layout of the dialog created
    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=450, width=450, show_label=False),
                HGroup('top', 'frontal', 'left',),
                HGroup('sensors', 'head_shape'),
                HGroup('scalp', 'scalp_alpha'),
                HGroup('fiducials', 'dig_points'),
                )



def trans(pts, params):
    a, b, c, p, q, r = params
    assert len(pts) == 3
    # trans
    pts = pts + [[a], [b], [c]]
    # rot
    pts = np.dot([[1, 0, 0], [0, cos(p), -sin(p)], [0, sin(p), cos(p)]], pts)
    pts = np.dot([[cos(q), 0, sin(q)], [0, 1, 0], [-sin(q), 0, cos(q)]], pts)
    pts = np.dot([[cos(r), -sin(r), 0], [sin(r), cos(r), 0], [0, 0, 1]], pts)
    return pts


def fit(src, tgt):
    def err(params):
        est = trans(src, params)
        return (tgt - est).ravel()

    # initial guess
    params = (0, 0, 0, 0, 0, 0)
    est_params, _ = leastsq(err, params)
    return est_params


class mrk_fix(traits.HasTraits):
    # views
    center = traits.Button()

    # markers
    _0 = traits.Bool(True)
    _1 = traits.Bool(True)
    _2 = traits.Bool(True)
    _3 = traits.Bool(True)
    _4 = traits.Bool(True)

    scene = traits.Instance(MlabSceneModel, ())

    def __init__(self, mrk, raw):
        """
        Parameters
        ----------

        mrk : load.kit.marker_avg_file | str(path)
            marker_avg_file object, or path to a marker file.
        raw : mne.fiff.Raw | str(path)
            MNE Raw object, or path to a raw file.

        """
        traits.HasTraits.__init__(self)

        if isinstance(mrk, basestring):
            mrk = load.kit.marker_avg_file(mrk)
        self.mrk_pts = mrk_pts = mrk.points.T

        if isinstance(raw, basestring):
            raw = load.fiff.Raw(raw)
        dig_pts = filter(lambda d: d['kind'] == 2, raw.info['dig'])
        dig_pts = np.vstack([d['r'] for d in dig_pts]).T * 1000
        self.dig_pts = dig_pts

        self.configure_traits()
        self.plot()
        self.center = True

    @traits.on_trait_change('_0,_1,_2,_3,_4')
    def plot(self):
        mlab = self.scene.mlab
        mlab.clf()

        dig_pts = self.dig_pts
        mrk_pts = self.mrk_pts

        x, y, z = mrk_pts
        mlab.points3d(x, y, z, color=(1, 0, 0), opacity=.3)

        idx = np.array([self._0, self._1, self._2, self._3, self._4], dtype=bool)
        if np.sum(idx) < 3:
            mlab.text(0.1, 0.1, 'Need at least 3 points!')
            return

        for i in xrange(mrk_pts.shape[1]):
            x, y, z = mrk_pts[:, i]
            mlab.text3d(x, y, z, str(i), scale=10)

        est_params = fit(dig_pts[:, idx], mrk_pts[:, idx])
        est = trans(dig_pts, est_params)

        # plot dig fiducials
        x, y, z = est
        mlab.points3d(x, y, z, color=(0, .5, 1), opacity=.3)
        return

    @traits.on_trait_change('center')
    def _view_top(self):
        self.set_view('center')

    def set_view(self, view='frontal'):
#        self.scene.parallel_projection = True
#        self.scene.camera.parallel_scale = .15
#        kwargs = dict(azimuth=90, elevation=90, distance=None, roll=180, reset_roll=True)
#        if view == 'left':
#            kwargs.update(azimuth=180, roll=90)
#        elif view == 'top':
#            kwargs.update(elevation=0)
#        self.scene.mlab.view(**kwargs)
        self.scene.mlab.view(azimuth=90, elevation=0, roll=90)

    # the layout of the dialog created
    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=450, width=450, show_label=False),
                HGroup('center',),
                HGroup('_0', '_1', '_2', '_3', '_4'),
                )
