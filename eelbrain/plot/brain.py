'''
Created on Oct 25, 2012

@author: christian
'''
import os
import shutil
import subprocess
import tempfile

import numpy as np
from mayavi import mlab
import surfer

from eelbrain.utils.subp import cmd_exists
from eelbrain.vessels.dimensions import find_time_point


__all__ = ['stc', 'stat']


def stat(p_map, param_map=None, p0=0.05, p1=0.01, solid=False, hemi='both'):
    """
    solid : bool
        Use solid color patches between p0 and p1 (else: blend transparency
        between p0 and p1)

    """
    pmap, lut, vmax = colorize_p(p_map, param_map, p0=p0, p1=p1, solid=solid)
    return stc(pmap, colormap=lut, min= -vmax, max=vmax, colorbar=False, hemi=hemi)


class stc:
    def __init__(self, v, colormap='hot', min=0, max=30, surf='smoothwm',
                 figsize=(500, 500), colorbar=True, hemi='both'):
        """
        Parameters
        ----------

        v : ndvar [source [ x time]]
            Ndvar to plot. Must contain a source dimension, and can optionally
            contain a time dimension.

        surf : 'smoothwm' | ...
            Freesurfer surface.

        hemi : 'both' | 'l[eft]' | 'r[ight]'
            Only plot one hemisphere.

        """
        hemi = hemi.lower()
        if hemi.startswith('l'):
            v = v.subdata(source='lh')
        elif hemi.startswith('r'):
            v = v.subdata(source='rh')

        self.fig = fig = mlab.figure(size=figsize)
        self.lh = self.rh = None
        b_kwargs = dict(surf=surf, figure=fig, curv=True)
        d_kwargs = dict(colormap=colormap, alpha=1, smoothing_steps=20,
                        time_label='%.3g s', min=min, max=max,
                        colorbar=colorbar)
        if v.has_dim('time'):
            d_kwargs['time'] = v.time.x

        if v.source.lh_n:
            self.lh = self._hemi(v, 'lh', b_kwargs, d_kwargs)
            d_kwargs['time_label'] = ''
        if v.source.rh_n:
            self.rh = self._hemi(v, 'rh', b_kwargs, d_kwargs)

        # time
        if v.has_dim('time'):
            self._time = v.get_dim('time')
        else:
            self._time = False

    def _hemi(self, v, hemi, b_kwargs, d_kwargs):
        brain = surfer.Brain(v.source.subject, hemi, **b_kwargs)
        data = v.subdata(source=hemi).x
        vert = v.source.vertno[hemi == 'rh']
        brain.add_data(data, vertices=vert, **d_kwargs)
        return brain

    def animate(self, tstart=None, tstop=None, tstep=None,
                save_frames=False, save_mov=False, framerate=10,
                codec='mpeg4'):
        """
        cycle through time points and optionally save each image. Saving the
        animation (``save_mov``) requires `ffmpeg <http://ffmpeg.org>`_


        Parameters
        ----------

        tstart, tstop, tstep | scalar
            Start, end and step time for the animation.

        save_frames : str(path)
            Path to save frames to. Should contain '%s' for frame index.
            Extension determines format (mayavi supported formats).

        save_mov : str(path)
            save the movie

        """
        # make sure movie file will be writable
        if save_mov:
            save_mov = os.path.expanduser(save_mov)
            save_mov = os.path.abspath(save_mov)
            root, ext = os.path.splitext(save_mov)
            dirname = os.path.dirname(save_mov)
            if ext not in ['.mov', '.avi']:
                ext = '.mov'
                save_mov = root + ext

            if not cmd_exists('ffmpeg'):
                err = ("Need ffmpeg for saving movies. Download from "
                       "http://ffmpeg.org/download.html")
                raise RuntimeError(err)
            elif os.path.exists(save_mov):
                os.remove(save_mov)
            elif not os.path.exists(dirname):
                os.mkdir(dirname)

        # find time points
        if tstep is None:
            times = self._time.x
            if tstart is not None:
                times = times[times >= tstart]
            if tstop is not None:
                times = times[times <= tstop]
        else:
            if tstart is None:
                tstart = self._time.x.min()
            if tstop is None:
                tstop = self._time.x.max()
            times = np.arange(tstart, tstop + tstep / 2, tstep)

        # find number of digits necessary to name frames
        n_digits = 1 + int(np.log10(len(times)))
        fmt = '%%0%id' % n_digits

        # find output paths
        if save_frames:
            tempdir = False
            save_frames = os.path.expanduser(save_frames)
            save_frames = os.path.abspath(save_frames)
            try:
                save_frames = save_frames % fmt
            except TypeError:
                try:
                    save_frames % 0
                except TypeError:
                    err = ("save needs to specify a path that can be formatted "
                           "with exactly one integer")
                    raise ValueError(err)
            dirname = os.path.split(save_frames)[0]
            if not os.path.exists(dirname):
                os.makedirs(dirname)
        else:
            tempdir = tempfile.mkdtemp()
            save_frames = os.path.join(tempdir, 'frame%s.png' % fmt)

        # render and save the frames
        for i, t in enumerate(times):
            self.set_time(t)
            if save_frames:
                fname = save_frames % i
                self.fig.scene.save(fname)

        # make the movie
        if save_mov:
            frame_dir, frame_name = os.path.split(save_frames)
            cmd = ['ffmpeg',  # ?!? order of options matters
                   '-f', 'image2',  # force format
                   '-r', str(framerate),  # framerate
                   '-i', frame_name,
                   '-c', codec,
                   '-sameq', save_mov,
                   '-pass', '2'  #
                   ]
            sp = subprocess.Popen(cmd, cwd=frame_dir,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)
            stdout, stderr = sp.communicate()
            if not os.path.exists(save_mov):
                raise RuntimeError("ffmpeg failed:\n" + stderr)

            if tempdir:
                shutil.rmtree(tempdir)

    def close(self):
        if self.lh is None:
            self.rh.close
        else:
            self.lh.close()

    def set_time(self, t):
        "set the time frame displayed (in seconds)"
        if self._time is False:
            return

        time_idx , t = find_time_point(self._time, t)

        if self.lh is not None:
            self.lh.set_data_time_index(time_idx)
        if self.rh is not None:
            self.rh.set_data_time_index(time_idx)


def colorize_p(pmap, tmap, p0=0.05, p1=0.01, solid=False):
    """

    assuming

    loop up table
    -------------

    index -> p-value
    0 -> 0
    .
    126
    .    -> p0
    127
    .    -> vmax
    128
    .    -> p0 (neg)
    .
    255

    """
    # modify pmap so that
    pstep = 2 * p0 / 125.5  # d p / index
    vmax = p0 + pstep
    pmap = vmax - pmap
    pmap.x.clip(0, vmax, pmap.x)

    # add sign to p-values
    if tmap is not None:
        pmap.x *= np.sign(tmap.x)

    # http://docs.enthought.com/mayavi/mayavi/auto/example_custom_colormap.html
    lut = np.zeros((256, 4), dtype=np.uint8)
    i0 = 1
    i1 = int(p1 / pstep)

    # negative
    lut[:128, 2] = 255
    lut[:i1, 0] = 255
    # positive
    lut[128:, 0] = 255
    lut[-i1:, 1] = 255
    # alpha
    if solid:
        lut[:126, 3] = 255
        lut[126, 3] = 127
        lut[129, 3] = 127
        lut[130:, 3] = 255
    else:
        n = 127 - i1
        lut[:i1, 3] = 255
        lut[126:i1 - 1:-1, 3] = np.linspace(0, 255, n)
        lut[129:-i1, 3] = np.linspace(0, 255, n)
        lut[-i1:, 3] = 255

    return pmap, lut, vmax

