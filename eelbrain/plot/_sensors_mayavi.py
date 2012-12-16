'''
Created on Sep 30, 2012

@author: christian
'''
from copy import deepcopy
import logging
import os
import shutil
import subprocess
import time

import numpy as np
from numpy import sin, cos
import scipy
from scipy.optimize import leastsq

from mayavi import mlab
from mayavi.tools import pipeline

import traits.api as traits
from traitsui.api import View, Item, HGroup
from tvtk.pyface.scene_editor import SceneEditor
from mayavi.tools.mlab_scene_model import MlabSceneModel
from mayavi.core.ui.mayavi_scene import MayaviScene

import mne
from mne import fiff
from mne import write_trans
from mne.fiff.constants import FIFF

from eelbrain import load
from eelbrain import ui



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
        pts = filter(lambda d: d['kind'] == 4, raw.info['dig'])
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



def fit(src, tgt):
    def err(params):
        est = trans(src, params)
        return (tgt - est).ravel()

    # initial guess
    params = (0, 0, 0, 0, 0, 0)
    est_params, _ = leastsq(err, params)
    return est_params



class head:
    """
    represents a head model with fiducials

    Complete transformations:

        pts = T * origin * pts
        T = rot * mult

    split into

     - scale -> apply now
     - movement and rotation -> write to trans file

    get_T_scale:
        inv(origin) * mult * origin
    get_T_trans:
        inv(dig_origin) * rot * origin

    """
    def __init__(self, pts, fid, tri=None):
        """
        opacity
            opacity of points
        rep
            representation of the surface
        """
        self.origin = trans(0, 0, 0)
        self.rot = trans(0, 0, 0)
        self.mult = trans(0, 0, 0)

        if tri is None:
            d = scipy.spatial.Delaunay(pts)
            tri = d.convex_hull

        self.pts = pts = np.vstack((pts.T, np.ones(len(pts))))
        self.tri = tri

        # fiducials
        self.fid = np.vstack((fid.T, np.ones(len(fid))))
        self.aur_r = fid[0]
        self.nas = fid[1]
        self.aur_l = fid[2]

        self._plots = []

    def get_pts(self, T=None):
        "returns `T * self.origin * self.pts`  (i.e., ignores self.T)"
        Tc = self.origin
        if T is not None:
            Tc = T * Tc

        pts = Tc * self.pts
        return np.array(pts[:3].T)

    def get_T(self):
        "complete transformation (T * origin)"
        return self.rot * self.mult * self.origin

    def plot(self, fig, pt_scale=10, opacity=1., rep='surface'):
        h = {}
        self._plots.append(h)

        pts = self.pts
        tri = self.tri
        x, y, z, _ = pts
        mesh = pipeline.triangular_mesh_source(x, y, z, tri, figure=fig)
        h['surf'] = pipeline.surface(mesh, figure=fig, color=(1, 1, 1),
                                     representation=rep)

        kwargs = dict(color=(1, 0, 0), figure=fig, scale_factor=pt_scale,
                      opacity=opacity)
        h['mesh'] = mesh

        x, y, z = self.nas
        h['nas_pt'] = src = pipeline.scalar_scatter(x, y, z)
        pipeline.glyph(src, **kwargs)
        kwargs['color'] = (1, 1, 0)
        x, y, z = self.aur_l
        h['aur_l_pt'] = src = pipeline.scalar_scatter(x, y, z)
        pipeline.glyph(src, **kwargs)
        x, y, z = self.aur_r
        h['aur_r_pt'] = src = pipeline.scalar_scatter(x, y, z)
        pipeline.glyph(src, **kwargs)

    def set_origin(self, x, y, z):
        self.origin = trans(-x, -y, -z)
        self.update_plot()

    def set_T(self, rot, mult):
        self.mult = mult
        self.rot = rot
        self.update_plot()

    def update_plot(self):
        T = self.get_T()
        pts = np.array(T * self.pts)
        fid = np.array(T * self.fid)
        for h in self._plots:
            # mesh
            h['mesh'].data.points = pts[:3].T

            # points
            h['aur_r_pt'].data.points = fid[:3, :1].T
            h['nas_pt'].data.points = fid[:3, 1:2].T
            h['aur_l_pt'].data.points = fid[:3, 2:].T



class fit_coreg:
    def __init__(self, s_from, raw, s_to=None, subjects_dir=None):
        """
        s_from : str
            name of the source subject (e.g., 'fsaverage')
        raw : str(path)
            path to a raw file containing the digitizer data.
        dest : str
            name of the destination for the new files (i.e. the subject for
            which the files are adapted)

        """
        # interpret paths
        if subjects_dir is None:
            if 'SUBJECTS_DIR' in os.environ:
                subjects_dir = os.environ['SUBJECTS_DIR']
            else:
                err = ("If SUBJECTS_DIR is not set as environment variable, "
                       "it must be provided as subjects_dir parameter")
                raise ValueError(err)

        # MRI head shape
        fname = os.path.join(subjects_dir, s_from, 'bem', 'outer_skin.surf')
        pts, tri = mne.read_surface(fname)
        fname = os.path.join(subjects_dir, s_from, 'bem', s_from + '-fiducials.fif')
        fid, _ = read_fiducials(fname)
        fid = np.array([d['r'] for d in fid]) * 1000
        self.MRI = head(pts, fid, tri=tri)

        # digitizer data from raw
        self._raw = raw
        raw = mne.fiff.Raw(raw)
        pts = filter(lambda d: d['kind'] == 4, raw.info['dig'])
        pts = np.array([d['r'] for d in pts]) * 1000
#        fid = []
#        for d in raw.info['dig']:
# #            d['r'] *= 1000
#            if d['kind'] == 1:
#                dc = d.copy()
#                dc['r'] *= 1000
#                raw.info['dig']
#                fid.append(dc)

        fid = filter(lambda d: d['kind'] == 1, raw.info['dig'])
        self._fid_pts = fid
        fid = np.array([d['r'] for d in fid]) * 1000
        self.DIG = head(pts, fid,)

        # move to the origin
        self.MRI.set_origin(*self.MRI.nas[:3])
        self.DIG.set_origin(*self.DIG.nas[:3])
        # store the origin distance
        self._trans0 = self.DIG.nas[:3] - self.DIG.nas[:3]

        self.subjects_dir = subjects_dir
        self.s_from = s_from
        self.s_to = s_to

    def plot(self, size=(512, 512)):
        self.fig = fig = mlab.figure(size=size)
        self.MRI.plot(fig)
        self.DIG.plot(fig, pt_scale=40, opacity=.25, rep='wireframe')
        self.update_plot()

    def update_plot(self):
        self.MRI.update_plot()
        self.DIG.update_plot()

# --- fitting --- ---
    def _error(self, T):
        "For each point in pts, the distance to the closest point in pts0"
#        err = np.empty(pts.shape)
#        for i, pos in enumerate(pts):
#            dist3d = pts0 - pos[None, :]
#            dist = np.sqrt(np.sum(dist3d ** 2, 1))
#            idx = dist.argmin()
#            err[i] = dist3d[idx]
        pts = self.DIG.get_pts()
        pts0 = self.MRI.get_pts(T)
        Y = scipy.spatial.distance.cdist(pts, pts0, 'euclidean')
        dist = Y.min(axis=1)
        return dist

    def _dist_fixnas(self, param):
        rx, ry, rz, mx, my, mz = param
        T = rot(rx, ry, rz) * mult(mx, my, mz)
        err = self._error(T)
        logging.debug("Params = %s -> Error = %s" % (param, np.sum(err ** 2)))
        return err

    def _dist_fixnas_1mult(self, param):
        rx, ry, rz, m = param
        T = rot(rx, ry, rz) * mult(m, m, m)
        err = self._error(T)
        logging.debug("Params = %s -> Error = %s" % (param, np.sum(err ** 2)))
        return err

    def _estimate_fixnas(self, params=(0, 0, 0, 1, 1, 1), **kwargs):
        params = np.asarray(params, dtype=float)
        est_params, self.info = leastsq(self._dist_fixnas, params, **kwargs)
        return est_params

    def _estimate_fixnas_1mult(self, params=(0, 0, 0, 1), **kwargs):
        params = np.asarray(params, dtype=float)
        est_params, self.info = leastsq(self._dist_fixnas_1mult, params, **kwargs)
        return est_params

    def fit(self, epsfcn=0.01, method='3mult', **kwargs):
        """
        http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.leastsq.html
        """
        t0 = time.time()
        if method == '1mult':
            est = self._estimate_fixnas_1mult(epsfcn=epsfcn, **kwargs)
            est = np.hstack((est, np.ones(2) * est[-1:]))
        elif method == '3mult':
            est = self._estimate_fixnas(epsfcn=epsfcn, **kwargs)
        else:
            raise ValueError("method")
        self.set(*est)
        dt = time.time() - t0
        print "%r took %.2f minutes" % (method, dt / 60)
        return est

    def get_T_scale(self):
        return self.MRI.origin.I * self.DIG.mult * self.MRI.origin

    def get_T_trans(self, unit='mm'):
        "T for the trans file;, rot + translation "
        if unit == 'mm':
            T = self.DIG.origin.I * self.MRI.rot * self.MRI.origin
        elif unit == 'm':
            trans0 = self.MRI.origin.copy()
            trans0[:3, 3] /= 1000
            trans2 = self.DIG.origin.I
            trans2[:3, 3] /= 1000
            T = trans2 * self.MRI.rot * trans0
        else:
            raise ValueError('Unknown unit %r' % unit)
        return T.I

    def save(self, s_to=None):
        """
        s_to : None | str
            Override s_to set on initialization.
        """
        s_from = self.s_from
        if s_to is None:
            if self.s_to is None:
                raise IOError("No destination specified")
            else:
                s_to = self.s_to

        # make sure we have an empty target directory
        err = []
        rm_sdir = False
        rm_trans = False
        sdir = os.path.join(self.subjects_dir, '{sub}')
        sdir_dest = sdir.format(sub=s_to)
        if os.path.exists(sdir_dest):
            err.append("Subject directory exists: %r." % sdir_dest)
            rm_sdir = True
        rawdir = os.path.dirname(self._raw)
        trans_fname = os.path.join(rawdir, s_to + '-trans.fif')
        if os.path.exists(trans_fname):
            err.append("Trans file exists: %r" % trans_fname)
            rm_trans = True
        if err:
            msg = '\n'.join(err)
            if ui.ask("Overwrite?", msg):
                if rm_sdir:
                    shutil.rmtree(sdir_dest)
                if rm_trans:
                    os.remove(trans_fname)
            else:
                raise IOError(msg)


        bemdir = os.path.join(sdir, 'bem')
        os.makedirs(bemdir.format(sub=s_to))
        bempath = os.path.join(bemdir, '{name}.{ext}')
        surfdir = os.path.join(sdir, 'surf')
        os.mkdir(surfdir.format(sub=s_to))
        surfpath = os.path.join(surfdir, '{name}')

        # write T
        fname = os.path.join(sdir, 'T.txt').format(sub=s_to)
        with open(fname, 'w') as fid:
            fid.write(', '.join(map(str, self._params)))

        # write trans file
        T_trans = self.get_T_trans('m')
        dig = deepcopy(self._fid_pts)  # these are in m
        for d in dig:
            coord = apply_T_1pt(d['r'], T_trans, scale=1. / 1000)
            d['r'] = coord[:3, 0]
        info = {'to':FIFF.FIFFV_COORD_MRI, 'from': FIFF.FIFFV_COORD_HEAD,
                'trans': np.array(T_trans), 'dig': dig}
        write_trans(trans_fname, info)


        T = self.get_T_scale()


        # assemble list of surface files to duplicate
        # surf/ files
        surf_names = ('orig', 'orig_avg',
                      'inflated', 'inflated_avg', 'inflated_pre',
                      'pial', 'pial_avg',
                      'smoothwm',
                      'white', 'white_avg',
                      'sphere', 'sphere.reg', 'sphere.reg.avg')
        paths = {}
        for name in surf_names:
            for hemi in ('lh.', 'rh.'):
                k = surfpath.format(sub=self.s_from, name=hemi + name)
                v = surfpath.format(sub=s_to, name=hemi + name)
                paths[k] = v

        # watershed files
        for surf in ['inner_skull', 'outer_skull', 'outer_skin']:
            k = bempath.format(sub=s_from, name=surf, ext='surf')
            k = os.path.realpath(k)
            v = bempath.format(sub=s_to, name=surf, ext='surf')
            paths[k] = v

        # make surf files (in mm)
        for src, dest in paths.iteritems():
            pts, tri = mne.read_surface(src)
            pts = apply_T(pts, T)
            mne.write_surface(dest, pts, tri)


        # write bem
        path = os.path.join(self.subjects_dir, '{sub}', 'bem', '{sub}-{name}.fif')
        for name in ['head']:  # '5120-bem-sol',
            src = path.format(sub=s_from, name=name)
            dest = path.format(sub=s_to, name=name)
            surf = mne.read_bem_surfaces(src)[0]
            surf['rr'] = apply_T(surf['rr'], T)
            mne.write_bem_surface(dest, surf)

        # fiducials [in m]
        fname = path.format(sub=s_from, name='fiducials')
        pts, cframe = read_fiducials(fname)
        for pt in pts:
            pt['r'] = apply_T_1pt(pt['r'], T, scale=1. / 1000)
        fname = path.format(sub=s_to, name='fiducials')
        write_fiducials(fname, pts, cframe)

        # write src
        path = os.path.join(self.subjects_dir, '{sub}', 'bem', '{sub}-ico-4-src.fif')
        src = path.format(sub=s_from)
        sss = mne.read_source_spaces(src)
        for ss in sss:
            ss['rr'] = apply_T(ss['rr'], T)
            ss['nn'] = apply_T(ss['nn'], T.I.T)
        dest = path.format(sub=s_to)
        mne.write_source_spaces(dest, sss)

        # duplicate files
        path = os.path.join(self.subjects_dir, '{sub}', 'surf', '{name}')
        for name in ['lh.curv', 'rh.curv']:
            src = path.format(sub=s_from, name=name)
            dest = path.format(sub=s_to, name=name)
            shutil.copyfile(src, dest)

#        subprocess.call(["mne_setup_forward_model", "--subject", "R0040",
#                         "--ico", "4", "--surf"])

    def set(self, rx, ry, rz, mx, my, mz, error=False):
        self._params = (rx, ry, rz, mx, my, mz)
        r = rot(rx, ry, rz)
        m = mult(mx, my, mz)
        self.MRI.set_T(r, m)
        if error:
            print "error SS = %g" % np.sum(self._error(r * m) ** 2)



def apply_T_1pt(X, T, scale=1):
    X = np.vstack((X[:, None], [1]))
    if scale != 1:
        X[:3] *= scale

    X = T * X
    X = X[:3, 0]

    X = np.array(X)
    if scale != 1:
        X /= scale
    return X


def apply_T(X, T, scale=1):
    X = np.vstack((X.T, np.ones(len(X))))
    if scale != 1:
        X[:3] *= scale

    X = T * X
    X = X[:3].T

    X = np.array(X)
    if scale != 1:
        X /= scale
    return X


def trans(x=0, y=0, z=0):
    "MNE manual p. 95"
    m = np.matrix([[1, 0, 0, x],
                   [0, 1, 0, y],
                   [0, 0, 1, z],
                   [0, 0, 0, 1]])
    return m

def rot(x=0, y=0, z=0):
    r = np.matrix([[cos(y) * cos(z), -cos(x) * sin(z) + sin(x) * sin(y) * cos(z), sin(x) * sin(z) + cos(x) * sin(y) * cos(z), 0],
                  [cos(y) * sin(z), cos(x) * cos(z) + sin(x) * sin(y) * sin(z), -sin(x) * cos(z) + cos(x) * sin(y) * sin(z), 0],
                  [-sin(y), sin(x) * cos(y), cos(x) * cos(y), 0],
                  [0, 0, 0, 1]])
    return r

def mult(x=1, y=1, z=1):
    s = np.matrix([[x, 0, 0, 0],
                   [0, y, 0, 0],
                   [0, 0, z, 0],
                   [0, 0, 0, 1]])
    return s



from mne.fiff import write
from struct import pack
"""
$ mne_show_fiff --in fsaverage-fiducials.fif
100 = file ID
101 = dir pointer
106 = free list
104 = {    107 = isotrak
  3506 = MNE coordf
   213 = dig. point [3]
105 = }    107 = isotrak
108 = NOP

"""
def read_fiducials(fname):
    """
    Returns a list of  arrays

    Courtesy of Alex Gramfort


    Coordinate frames:
    1 FIFFV_COORD_DEVICE
    2 FIFFV_COORD_ISOTRAK
    3 FIFFV_COORD_HPI
    4 FIFFV_COORD_HEAD
    5 FIFFV_COORD_MRI
    6 FIFFV_COORD_MRI_SLICE
    7 FIFFV_COORD_MRI_DISPLAY
    8 FIFFV_COORD_DICOM_DEVICE
    9 FIFFV_COORD_IMAGING_DEVICE
    0 FIFFV_COORD_UNKNOWN

    """
    fid, tree, _ = fiff.open.fiff_open(fname)
    isotrak = fiff.tree.dir_tree_find(tree, FIFF.FIFFB_ISOTRAK)
    isotrak = isotrak[0]
    pts = []
    coord_frame = 0
    for k in range(isotrak['nent']):
        kind = isotrak['directory'][k].kind
        pos = isotrak['directory'][k].pos
        if kind == FIFF.FIFF_DIG_POINT:
            tag = fiff.tag.read_tag(fid, pos)
            pts.append(tag.data)
        elif kind == FIFF.FIFF_MNE_COORD_FRAME:
            tag = fiff.tag.read_tag(fid, pos)
            coord_frame = tag.data[0]

    fid.close()
    return pts, coord_frame

def write_fiducials(fname, dig, coord_frame=0):
    """
    Write
    """
    fid = write.start_file(fname)
    write.start_block(fid, FIFF.FIFFB_ISOTRAK)
#    write.write_id(fid, FIFF.FIFF_MNE_COORD_FRAME, dig['coord_frame'])
    write.write_int(fid, FIFF.FIFF_MNE_COORD_FRAME, coord_frame)
    for pt in dig:
        write.write_dig_point(fid, pt)

    write.end_block(fid, FIFF.FIFFB_ISOTRAK)
    write.end_file(fid)



def rotation(theta):
    """
    http://mail.scipy.org/pipermail/numpy-discussion/2009-March/040807.html
    """
    tx, ty, tz = theta

    Rx = np.array([[1, 0, 0], [0, cos(tx), -sin(tx)], [0, sin(tx), cos(tx)]])
    Ry = np.array([[cos(ty), 0, -sin(ty)], [0, 1, 0], [sin(ty), 0, cos(ty)]])
    Rz = np.array([[cos(tz), -sin(tz), 0], [sin(tz), cos(tz), 0], [0, 0, 1]])

    return np.dot(Rx, np.dot(Ry, Rz))


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


class geom:
    """
    represents a head model with fiducials

    Complete transformations:

        pts = T * origin * pts
        T = rot * mult

    split into

     - scale -> apply now
     - movement and rotation -> write to trans file

    get_T_scale:
        inv(origin) * mult * origin
    get_T_trans:
        inv(dig_origin) * rot * origin

    """
    def __init__(self, pts, tri=None):
        """
        tri : None | array


        opacity
            opacity of points
        rep
            representation of the surface
        """
        self.trans = []

        self.pts = pts = np.vstack((pts.T, np.ones(len(pts))))
        self.tri = tri

        self._plots_surf = []
        self._plots_pt = []

    def get_pts(self, T=None):
        """
        returns `T * self.pts`  (i.e., ignores self.T)

        T : None
            None: don't transform
            True: apply T that is stored in the object
            matrix: apply the matrix

        """
        if T is True:
            T = self.get_T()

        if T is None:
            pts = self.pts
        else:
            pts = T * self.pts

        return np.array(pts[:3].T)

    def get_T(self):
        "complete transformation"
        T = np.matrix(np.eye(4))
        for Ti in self.trans:
            T = Ti * T
        return T

    def plot_solid(self, fig, opacity=1., rep='surface', color=(1, 1, 1)):
        if self.tri is None:
            d = scipy.spatial.Delaunay(self.pts[:3].T)
            self.tri = d.convex_hull

        x, y, z, _ = self.pts

        mesh = pipeline.triangular_mesh_source(x, y, z, self.tri, figure=fig)
        surf = pipeline.surface(mesh, figure=fig, color=color, opacity=opacity,
                                representation=rep)

        self._plots_surf.append((mesh, surf))
        if self.trans:
            self.update_plot()

    def plot_points(self, fig, scale=1e-2, opacity=1., color=(1, 0, 0)):
        x, y, z, _ = self.pts

        src = pipeline.scalar_scatter(x, y, z)
        glyph = pipeline.glyph(src, color=color, figure=fig, scale_factor=scale,
                               opacity=opacity)

        self._plots_pt.append((src, glyph))
        if self.trans:
            self.update_plot()

    def add_T(self, T):
        self.trans.append(T)
        self.update_plot()

    def reset_T(self):
        self.trans = []
        self.update_plot()

    def set_opacity(self, v=1):
        if v == 1:
            v = True
        elif v == 0:
            v = False
        else:
            raise NotImplementedError

        for _, plt in self._plots_pt + self._plots_surf:
            if isinstance(v, bool):
                plt.visible = v

    def set_T(self, T):
        """
        T : list | transformation matrix, shape = (4,4)
            The transformation to be applied, or a list of transformations.

        """
        if not isinstance(T, (list)):
            T = [T]

        for Ti in T:
            assert Ti.shape == (4, 4)

        self.trans = T
        self.update_plot()

    def update_plot(self):
        pts = self.get_pts(T=True)
        for mesh, _ in self._plots_surf:
            mesh.data.points = pts
        for src, _ in self._plots_pt:
            src.data.points = pts


class fit_dev2head:
    def __init__(self, raw, mrk):
        """
        Parameters
        ----------

        raw : mne.fiff.Raw | str(path)
            MNE Raw object, or path to a raw file.
        mrk : load.kit.marker_avg_file | str(path)
            marker_avg_file object, or path to a marker file.

        """
        # interpret mrk
        if isinstance(mrk, basestring):
            mrk = load.kit.marker_avg_file(mrk)

        # interpret raw
        if isinstance(raw, basestring):
            raw = load.fiff.Raw(raw)
        self.raw = raw

        # sensors
        pts = filter(lambda d: d['kind'] == FIFF.FIFFV_MEG_CH, raw.info['chs'])
        pts = np.array([d['loc'][:3] for d in pts])
        self.sensors = geom(pts)

        # marker points
        pts = mrk.points / 1000
        pts = pts[:, [1, 0, 2]]
        pts[:, 0] *= -1
        self.mrk = geom(pts)

        # head shape
        pts = filter(lambda d: d['kind'] == FIFF.FIFFV_POINT_EXTRA, raw.info['dig'])
        pts = np.array([d['r'] for d in pts])
        self.headshape = geom(pts)

        # HPI points
        pts = filter(lambda d: d['kind'] == FIFF.FIFFV_POINT_HPI, raw.info['dig'])
        assert [d['ident'] for d in pts] == range(1, 6)
        pts = np.array([d['r'] for d in pts])
        self.HPI = geom(pts)

        # T head-to-device
        trans = raw.info['dev_head_t']['trans']
        self.T_head2dev = np.matrix(trans).I
        self.reset_T()
        self._HPI_flipped = False

    def fit(self, include=range(5)):
        """
        Fit the marker points to the digitizer points.

        include : list
            which points to include.
        """
        def err(params):
            T = trans(*params[:3]) * rot(*params[3:])
            est = self.HPI.get_pts(T)[include]
            tgt = self.mrk.get_pts()[include]
            return (tgt - est).ravel()

        # initial guess
        params = (0, 0, 0, 0, 0, 0)
        params, _ = leastsq(err, params)
        self.est_params = params

        T = trans(*params[:3]) * rot(*params[3:])
        self.est_T = T

        self.headshape.set_T(T)
        self.HPI.set_T(T)

    def plot(self, size=(800, 800), fig=None, HPI_ns=False):
        """
        Plot sensor helmet and head. ``fig`` is used if provided, otherwise
        a new mayavi figure is created with ``size``.

        HPI_ns : bool
            Add number labels to the HPI points.

        """
        if fig == None:
            fig = mlab.figure(size=size)

        self.mrk.plot_points(fig, scale=1.1e-2, opacity=.5, color=(1, 0, 0))
        self.sensors.plot_points(fig, scale=1e-2, color=(0, 0, 1))

        self.HPI.plot_points(fig, scale=1e-2, color=(1, .8, 0))
        self.headshape.plot_solid(fig, opacity=1., color=(1, 1, 1))

        # label marker points
        for i, pt in enumerate(self.mrk.pts[:3].T):
            x, y, z = pt
            self.txt = mlab.text3d(x, y, z, str(i), scale=.01)

        if HPI_ns:  # label HPI points
            for i, pt in enumerate(self.HPI.pts[:3].T):
                x, y, z = pt
                mlab.text3d(x, y, z, str(i), scale=.01, color=(1, .8, 0))

    def reset_T(self):
        """
        Reset the current device-to-head transformation to the one contained
        in the raw file

        """
        T = self.T_head2dev
        self.headshape.set_T(T)
        self.HPI.set_T(T)

    def save(self, fname=None):
        """
        Save a copy of the raw file with the current device-to-head
        transformation

        """
        if fname is None:
            msg = "Destination for the modified raw file"
            ext = [('fif', 'MNE Fiff File')]
            fname = ui.ask_saveas("Save Raw File", msg, ext)

        info = self.raw.info
        info['dev_head_t']['trans'] = np.array(self.est_T.I)

        self.raw.save(fname)

    def set_hs_opacity(self, v=1):
        self.headshape.set_opacity(v)


