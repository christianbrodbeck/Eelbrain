'''
Coregistration for :mod:`mne`.
'''
# author: Christian Brodbeck

import os

import numpy as np
from numpy import sin, cos
import scipy
from scipy.optimize import leastsq

import wx
from mayavi import mlab
from mayavi.tools import pipeline

import traits.api as traits
from traitsui.api import View, Item, HGroup
from tvtk.pyface.scene_editor import SceneEditor
from mayavi.tools.mlab_scene_model import MlabSceneModel
from mayavi.core.ui.mayavi_scene import MayaviScene

import mne
from mne import fiff
from mne.fiff import write
from mne.fiff.constants import FIFF

from ... import ui
from .. import load
import _base

__all__ = ['dev_head_viewer', 'dev_head_fitter', 'dev_mri', 'mri_head_viewer',
           'mri_head_fitter', 'set_nasion']


def get_subjects_dir(subjects_dir):
    if subjects_dir is None:
        if 'SUBJECTS_DIR' in os.environ:
            subjects_dir = os.environ['SUBJECTS_DIR']
        else:
            msg = "Set SUBJECTS_DIR"
            subjects_dir = ui.ask_dir(msg, msg, must_exist=True)
        if not subjects_dir:
            err = ("No SUBJECTS_DIR specified")
            raise ValueError(err)
    return subjects_dir


class dev_head_viewer(traits.HasTraits):
    """
    Mayavi viewer for modifying the device-to-head coordinate coregistration.

    """
    # views
    front = traits.Button()
    left = traits.Button()
    top = traits.Button()

    # visibility
    head_shape = traits.Bool(True)
    mri = traits.Bool(True)

    # fitting
    _refit = traits.Bool(False)
    _0 = traits.Bool(True)
    _1 = traits.Bool(True)
    _2 = traits.Bool(True)
    _3 = traits.Bool(True)
    _4 = traits.Bool(True)

    _save = traits.Button()
    scene = traits.Instance(MlabSceneModel, ())

    def __init__(self, raw, mrk, bem='head', trans=None, subject=None,
                 subjects_dir=None):
        """
        Parameters
        ----------

        raw : mne.fiff.Raw | str(path)
            MNE Raw object, or path to a raw file.
        mrk : load.kit.MarkerFile | str(path) | array, shape = (5, 3)
            MarkerFile object, or path to a marker file, or marker points.
        bem : None | str(path)
            Name of the bem model to load (optional, only for visualization
            purposes).
        trans : None | dict | str(path)
            MRI-Head transform (optional).
            Can be None if the file is located in the raw directory and named
            "{subject}-trans.fif"
        subject : None | str
            Name of the mri subject.
            Can be None if the raw file-name starts with "{subject}_".

        """
        traits.HasTraits.__init__(self)
        self.configure_traits()

        fig = self.scene.mayavi_scene
        self.fitter = dev_head_fitter(raw, mrk, bem, trans, subject, subjects_dir)
        self.scene.disable_render = True
        self.fitter.plot(fig=fig)
        self._current_fit = None

        subject = self.fitter.subject
        self.scene.mlab.text(0.01, 0.01, subject, figure=fig, width=.2)

        self.front = True
        self.scene.disable_render = False

    @traits.on_trait_change('head_shape')
    def _show_hs(self):
        self.fitter.headshape.set_opacity(int(self.head_shape))

    @traits.on_trait_change('mri')
    def _show_mri(self):
        if self.fitter.MRI:
            self.fitter.MRI.set_opacity(int(self.mri))
        else:
            ui.message("No MRI Loaded", "Load an MRI when initializing the "
                       "viewer", '!')

    @traits.on_trait_change('front')
    def _view_front(self):
        self.set_view('front')

    @traits.on_trait_change('left')
    def _view_left(self):
        self.set_view('left')

    @traits.on_trait_change('top')
    def _view_top(self):
        self.set_view('top')

    @traits.on_trait_change('_refit,_0,_1,_2,_3,_4')
    def _fit(self):
        if not self._refit:
            if self._current_fit is not None:
                self.fitter.reset()
                self._current_fit = None
            return

        idx = np.array([self._0, self._1, self._2, self._3, self._4], dtype=bool)
        if np.sum(idx) < 3:
            ui.message("Not Enough Points Selected", "Need at least 3 points.",
                       '!')
            return

        self.fitter.fit(idx)
        self._current_fit = idx

    @traits.on_trait_change('_save')
    def save(self):
        bi = wx.BusyInfo("Saving Raw...")
        self.fitter.save()
        bi.Destroy()

    def set_view(self, view='front'):
        self.scene.parallel_projection = True
        self.scene.camera.parallel_scale = .15
        kwargs = dict(azimuth=90, elevation=90, distance=None, roll=180,
                      reset_roll=True, figure=self.scene.mayavi_scene)
        if view == 'left':
            kwargs.update(azimuth=180, roll=90)
        elif view == 'top':
            kwargs.update(elevation=0)
        self.scene.mlab.view(**kwargs)

    # the layout of the dialog created
    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=500, width=500, show_label=False),
                HGroup('top', 'front', 'left',),
                HGroup('head_shape', 'mri'),
                HGroup('_refit', '_0', '_1', '_2', '_3', '_4'),
                HGroup('_save'),
                )



class dev_head_fitter:
    def __init__(self, raw, mrk, bem='head', trans=None, subject=None,
                 subjects_dir=None):
        """
        Parameters
        ----------

        raw : mne.fiff.Raw | str(path)
            MNE Raw object, or path to a raw file.
        mrk : load.kit.MarkerFile | str(path) | array, shape = (5, 3)
            MarkerFile object, or path to a marker file, or marker points.
        bem : None | str(path)
            Name of the bem model to load (optional, only for visualization
            purposes).
        trans : None | dict | str(path)
            MRI-Head transform (optional).
            Can be None if the file is located in the raw directory and named
            "{subject}-trans.fif"
        subject : None | str
            Name of the mri subject.
            Can be None if the raw file-name starts with "{subject}_".

        """
        subjects_dir = get_subjects_dir(subjects_dir)

        # interpret mrk
        if isinstance(mrk, basestring):
            mrk = load.kit.MarkerFile(mrk)
        if isinstance(mrk, load.kit.MarkerFile):
            mrk = mrk.points

        # interpret raw
        if isinstance(raw, basestring):
            raw_fname = raw
            raw = load.fiff.mne_raw(raw)
        else:
            raw_fname = raw.info['filename']
        self._raw_fname = raw_fname
        self.raw = raw

        # subject
        if subject is None:
            _, tail = os.path.split(raw_fname)
            subject = tail.split('_')[0]
        self.subject = subject

        # bem (mri-head-trans)
        if bem is None:
            self.MRI = None
        else:
            # trans
            if trans is None:
                head, _ = os.path.split(raw_fname)
                trans = os.path.join(head, subject + '-trans.fif')
            if isinstance(trans, basestring):
                head_mri_t = mne.read_trans(trans)

            # mri_dev_t
            self.mri_head_t = np.matrix(head_mri_t['trans']).I

            fname = os.path.join(subjects_dir, subject, 'bem',
                                 '%s-%s.fif' % (subject, bem))
            self.MRI = geom_bem(fname, unit='m')
            self.MRI.set_T(self.mri_head_t)

        # sensors
        pts = filter(lambda d: d['kind'] == FIFF.FIFFV_MEG_CH, raw.info['chs'])
        pts = np.array([d['loc'][:3] for d in pts])
        self.sensors = geom(pts)

        # marker points
        pts = mrk / 1000
        pts = pts[:, [1, 0, 2]]
        pts[:, 0] *= -1
        self.mrk = geom(pts)

        # head shape
        pts = filter(lambda d: d['kind'] == FIFF.FIFFV_POINT_EXTRA, raw.info['dig'])
        pts = np.array([d['r'] for d in pts])
        self.headshape = geom(pts)

        # HPI points
        pts = filter(lambda d: d['kind'] == FIFF.FIFFV_POINT_HPI, raw.info['dig'])
#         assert [d['ident'] for d in pts] == range(1, 6)
        pts = np.array([d['r'] for d in pts])
        self.HPI = geom(pts)

        # T head-to-device
        trans = raw.info['dev_head_t']['trans']
        self.T_head2dev = np.matrix(trans).I
        self.reset()
        self._HPI_flipped = False

    def fit(self, include=range(5)):
        """
        Fit the marker points to the digitizer points.

        include : index (numpy compatible)
            Which points to include in the fit. Index should select among
            points [0, 1, 2, 3, 4].
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
        if self.MRI:
            self.MRI.set_T(T * self.mri_head_t)

    def plot(self, size=(800, 800), fig=None, HPI_ns=False):
        """
        Plot sensor helmet and head. ``fig`` is used if provided, otherwise
        a new mayavi figure is created with ``size``.

        HPI_ns : bool
            Add number labels to the HPI points.

        """
        if fig is None:
            fig = mlab.figure(size=size)

        self.mrk.plot_points(fig, scale=1.1e-2, opacity=.5, color=(1, 0, 0))
        self.sensors.plot_points(fig, scale=1e-2, color=(0, 0, 1))

        self.HPI.plot_points(fig, scale=1e-2, color=(1, .8, 0))
        self.headshape.plot_solid(fig, opacity=1., color=(1, 1, 1))

        if self.MRI is not None:
            self.MRI.plot_solid(fig, opacity=1., color=(.6, .6, .5))

        # label marker points
        for i, pt in enumerate(self.mrk.pts[:3].T):
            x, y, z = pt
            self.txt = mlab.text3d(x, y, z, str(i), scale=.01)

        if HPI_ns:  # label HPI points
            for i, pt in enumerate(self.HPI.pts[:3].T):
                x, y, z = pt
                mlab.text3d(x, y, z, str(i), scale=.01, color=(1, .8, 0))

        return fig

    def reset(self):
        """
        Reset the current device-to-head transformation to the one contained
        in the raw file

        """
        T = self.T_head2dev
        self.headshape.set_T(T)
        self.HPI.set_T(T)
        if self.MRI:
            self.MRI.set_T(T * self.mri_head_t)

    def save(self, fname=None):
        """
        Save a copy of the raw file with the current device-to-head
        transformation

        """
        if fname is None:
            msg = "Destination for the modified raw file"
            filetypes = [('MNE Fiff File', '*.fif')]
            dirname, fname = os.path.split(self._raw_fname)
            fname = ui.ask_saveas("Save Raw File", msg, filetypes, dirname,
                                  fname)
        if not fname:
            return

        self.raw.info['dev_head_t']['trans'] = np.array(self.est_T.I)
        self.raw.save(fname)

    def set_hs_opacity(self, v=1):
        self.headshape.set_opacity(v)



class dev_mri(object):
    """
    Plot the sensor and the mri head in a mayavi figure with proper
    coregistration.

    """
    def __init__(self, raw, subject=None, head_mri_t=None, mri='head',
                 hs='wireframe', subjects_dir=None, fig=None):
        """
        Parameters
        ----------

        raw : str(path) | Raw
            Path to raw fiff file, or the mne.fiff.Raw instance.
        subject : None | str
            Name of the mri subject. Can be None if the raw file-name starts
            with "{subject}_".
        head_mri_t : None | str(path)
            Path to the trans file for head-mri coregistration. Can be None if
            the file is located in the raw directory and named
            "{subject}-trans.fif"
        mri : str
            Name of the mri model to load (default is 'head')
        hs : None | 'wireframe' | 'surface' | 'points' | 'balls'
            How to display the digitizer head-shape stored in the raw file.

        """
        subjects_dir = get_subjects_dir(subjects_dir)

        if fig is None:
            fig = mlab.figure()
        self.fig = fig

        # raw
        if isinstance(raw, basestring):
            raw_fname = raw
            raw = load.fiff.mne_raw(raw_fname)
        else:
            raw_fname = raw.info['filename']

        # subject
        if subject is None:
            _, tail = os.path.split(raw_fname)
            subject = tail.split('_')[0]

        # mri_head_t
        if head_mri_t is None:
            head, _ = os.path.split(raw_fname)
            head_mri_t = os.path.join(head, subject + '-trans.fif')
        if isinstance(head_mri_t, basestring):
            head_mri_t = mne.read_trans(head_mri_t)

        # mri_dev_t
        mri_head_t = np.matrix(head_mri_t['trans']).I
        head_dev_t = np.matrix(raw.info['dev_head_t']['trans']).I
        mri_dev_t = head_dev_t * mri_head_t

        # sensors
        pts = filter(lambda d: d['kind'] == FIFF.FIFFV_MEG_CH, raw.info['chs'])
        pts = np.array([d['loc'][:3] for d in pts])
        self.sensors = geom(pts)
        self.sensors.plot_points(fig, scale=0.005, color=(0, 0, 1))

        # mri
        bemdir = os.path.join(subjects_dir, subject, 'bem')
        bem = os.path.join(bemdir, '%s-%s.fif' % (subject, mri))
        self.mri = geom_bem(bem, unit='m')
        self.mri.set_T(mri_dev_t)
        self.mri.plot_solid(fig, color=(.8, .6, .5))

        # head-shape
        if hs:
            self.hs = geom_dig_hs(raw.info['dig'], unit='m')
            self.hs.set_T(head_dev_t)
            if hs in ['surface', 'wireframe', 'points']:
                self.hs.plot_solid(fig, opacity=1, rep=hs, color=(1, .5, 0))
            elif hs == 'balls':
                self.hs.plot_points(fig, 0.01, opacity=0.5, color=(1, .5, 0))
            else:
                raise ValueError('hs kwarg can not be %r' % hs)

            self.hs.plot_solid(fig, opacity=1, rep='points', color=(1, .5, 0))

            # Fiducials
            fname = os.path.join(bemdir, subject + '-fiducials.fif')
            if os.path.exists(fname):
                dig, _ = read_fiducials(fname)
                self.mri_fid = geom_fid(dig, unit='m')
                self.mri_fid.set_T(mri_dev_t)
                self.mri_fid.plot_points(fig, scale=0.005)

            self.dig_fid = geom_fid(raw.info['dig'], unit='m')
            self.dig_fid.set_T(head_dev_t)
            self.dig_fid.plot_points(fig, scale=0.04, opacity=.25,
                                     color=(.5, .5, 1))

        self.view()

    def save_views(self, fname, views=['top', 'front', 'left'], overwrite=False):
        if not overwrite and os.path.exists(fname):
            raise IOError("File already exists: %r" % fname)

        tiler = _base.ImageTiler(ncol=len(views))
        for i, view in enumerate(views):
            tile_fname = tiler.get_tile_fname(col=i)
            self.view(view)
            self.fig.scene.save(tile_fname)
        tiler.save_frame(fname, overwrite=overwrite)

    def view(self, view='front'):
        self.fig.scene.parallel_projection = True
        self.fig.scene.camera.parallel_scale = .15
        kwargs = dict(azimuth=90, elevation=90, distance=None, roll=180,
                      reset_roll=True, figure=self.fig)
        if view == 'left':
            kwargs.update(azimuth=180, roll=90)
        elif view == 'right':
            kwargs.update(azimuth=0, roll=270)
        elif view == 'top':
            kwargs.update(elevation=0)
        mlab.view(**kwargs)





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
                   [0, 0, 0, 1]], dtype=float)
    return m

def rot(x=0, y=0, z=0):
    r = np.matrix([[cos(y) * cos(z), -cos(x) * sin(z) + sin(x) * sin(y) * cos(z), sin(x) * sin(z) + cos(x) * sin(y) * cos(z), 0],
                  [cos(y) * sin(z), cos(x) * cos(z) + sin(x) * sin(y) * sin(z), -sin(x) * cos(z) + cos(x) * sin(y) * sin(z), 0],
                  [-sin(y), sin(x) * cos(y), cos(x) * cos(y), 0],
                  [0, 0, 0, 1]], dtype=float)
    return r

def scale(x=1, y=1, z=1):
    s = np.matrix([[x, 0, 0, 0],
                   [0, y, 0, 0],
                   [0, 0, z, 0],
                   [0, 0, 0, 1]], dtype=float)
    return s



def read_fiducials(fname):
    """
    Read fiducials from a fiff file


    Returns
    -------
    pts : list of dicts
        List of digitizer points (each point in a dict).
    coord_frame : int
        The coordinate frame of the points (see below).


    MNE Coordinate Frames
    ---------------------

    1  FIFFV_COORD_DEVICE
    2  FIFFV_COORD_ISOTRAK
    3  FIFFV_COORD_HPI
    4  FIFFV_COORD_HEAD
    5  FIFFV_COORD_MRI
    6  FIFFV_COORD_MRI_SLICE
    7  FIFFV_COORD_MRI_DISPLAY
    8  FIFFV_COORD_DICOM_DEVICE
    9  FIFFV_COORD_IMAGING_DEVICE
    0  FIFFV_COORD_UNKNOWN

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
    write.write_int(fid, FIFF.FIFF_MNE_COORD_FRAME, coord_frame)
    for pt in dig:
        write.write_dig_point(fid, pt)

    write.end_block(fid, FIFF.FIFFB_ISOTRAK)
    write.end_file(fid)



class geom(object):
    """
    Represents a set of points and a list of transformations, and can plot the
    points as points or as surface to a mayavi figure.

    """
    def __init__(self, pts, tri=None):
        """
        pts : array, shape = (n_pts, 3)
            A list of points
        tri : None | array, shape = (n_tri, 3)
            Triangularization (optional). A list of triangles, each triangle
            composed of the indices of three points forming a triangle
            together.

        """
        self.trans = []

        self.pts = pts = np.vstack((pts.T, np.ones(len(pts))))
        self.tri = tri

        self._plots_surf = []
        self._plots_pt = []

    def get_pts(self, T=None):
        """
        returns the points contained in the object


        Parameters
        ----------

        T : None | true | Matrix (4x4)
            None: don't transform the points
            True: apply the transformation matrix that is stored in the object
            matrix: apply the given transformation matrix


        Returns
        -------

        pts : array, shape = (n_pts, 3)
            The points.

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
        "Returns: mesh, surf"
        if self.tri is None:
            d = scipy.spatial.Delaunay(self.pts[:3].T)
            self.tri = d.convex_hull

        x, y, z, _ = self.pts

        if rep == 'wireframe':
            kwa = dict(line_width=1)
        else:
            kwa = {}

        mesh = pipeline.triangular_mesh_source(x, y, z, self.tri, figure=fig)
        surf = pipeline.surface(mesh, figure=fig, color=color, opacity=opacity,
                                representation=rep, **kwa)

        self._plots_surf.append((mesh, surf))
        if self.trans:
            self.update_plot()

        return mesh, surf

    def plot_points(self, fig, scale=1e-2, opacity=1., color=(1, 0, 0)):
        "Returns: src, glyph"
        x, y, z, _ = self.pts

        src = pipeline.scalar_scatter(x, y, z)
        glyph = pipeline.glyph(src, color=color, figure=fig, scale_factor=scale,
                               opacity=opacity)

        self._plots_pt.append((src, glyph))
        if self.trans:
            self.update_plot()

        return src, glyph

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



class geom_fid(geom):
    def __init__(self, dig, unit='mm'):
        if unit == 'mm':
            x = 1000
        elif unit == 'm':
            x = 1
        else:
            raise ValueError('Unit: %r' % unit)

        dig = filter(lambda d: d['kind'] == 1, dig)
        pts = np.array([d['r'] for d in dig]) * x

        super(geom_fid, self).__init__(pts)
        self.unit = unit

        self.source_dig = dig
        digs = {d['ident']: d for d in dig}
        self.rap = digs[1]['r'] * x
        self.nas = digs[2]['r'] * x
        self.lap = digs[3]['r'] * x



class geom_dig_hs(geom):
    def __init__(self, dig, unit='mm'):
        if unit == 'mm':
            x = 1000
        elif unit == 'm':
            x = 1
        else:
            raise ValueError('Unit: %r' % unit)

        pts = filter(lambda d: d['kind'] == 4, dig)
        pts = np.array([d['r'] for d in pts]) * x

        super(geom_dig_hs, self).__init__(pts)



class geom_bem(geom):
    def __init__(self, bem, unit='m'):
        if isinstance(bem, basestring):
            bem = mne.read_bem_surfaces(bem)[0]

        pts = bem['rr']
        tri = bem['tris']

        if unit == 'mm':
            pts *= 1000
        elif unit == 'm':
            pass
        else:
            raise ValueError('Unit: %r' % unit)

        super(geom_bem, self).__init__(pts, tri)

