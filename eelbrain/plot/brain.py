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

from eelbrain.vessels.dimensions import find_time_point


__all__ = ['stc', 'stat']


def stat(p_map, param_map=None, p0=0.05, p1=0.01):
    pmap, lut, vmax = colorize_p(p_map, param_map, p0=p0, p1=p1)
    return stc(pmap, colormap=lut, min= -vmax, max=vmax, colorbar=False)


class stc:
    def __init__(self, v, colormap='hot', min=0, max=30, surf='smoothwm',
                 figsize=(500, 500), colorbar=True):
        """
        Parameters
        ----------

        v : ndvar [source_space ( x time)]
            ndvar to plot
        surf : 'smoothwm' |
            Freesurfer surface

        """
        self.fig = fig = mlab.figure(size=figsize)
        self.lh = self.rh = None
        b_kwargs = dict(surf=surf, figure=fig, curv=True)
        d_kwargs = dict(colormap=colormap, alpha=1, smoothing_steps=20,
                        time_label='%.3g s', min=min, max=max)
        if v.has_dim('time'):
            d_kwargs['time'] = v.time.x

        if v.source_space.lh_n:
            self.lh = self._hemi(v, 'lh', b_kwargs, d_kwargs)
            d_kwargs['time_label'] = ''
        if v.source_space.rh_n:
            self.rh = self._hemi(v, 'rh', b_kwargs, d_kwargs)

        # time
        if v.has_dim('time'):
            self._time = v.get_dim('time')
        else:
            self._time = False

    def _hemi(self, v, hemi, b_kwargs, d_kwargs):
        brain = surfer.Brain(v.source_space.subject, hemi, **b_kwargs)
        data = v.subdata(source_space=hemi).x
        vert = v.source_space.vertno[hemi == 'rh']
        brain.add_data(data, vertices=vert, **d_kwargs)
        return brain

    def animate(self, tstart=None, tstop=None, tstep=None,
                save_frames=False, save_mov=False, framerate=10,
                codec='mpeg4'):
        """
        cycle through time points and optionally save each image. Saving the
        animation (``save_mov``) requires `ffmpeg <http://ffmpeg.org>`_

        tstart, tstop, tstep | scalar
            Start, end and step time for the animation.

        save_frames : str(path)
            Path to save frames to. Should contain '%03d' for frame index.
            Extension determines format (mayavi supported formats).

        save_mov : str(path)
            save the movie

        """
        if save_frames:
            tempdir = False
            save_frames = os.path.expanduser(save_frames)
            save_frames = os.path.abspath(save_frames)
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
            save_frames = os.path.join(tempdir, 'frame%03d.png')

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

        for i, t in enumerate(times):
            self.set_time(t)
            if save_frames:
                fname = save_frames % i
                self.fig.scene.save(fname)
        
        if save_mov:
            save_mov = os.path.expanduser(save_mov)
            save_mov = os.path.abspath(save_mov)
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

    def set_time(self, t):
        "set the time frame displayed (in seconds)"
        if self._time is False:
            return

        time_idx , t = find_time_point(self._time, t)

        if self.lh is not None:
            self.lh.set_data_time_index(time_idx)
        if self.rh is not None:
            self.rh.set_data_time_index(time_idx)


def colorize_p(pmap, tmap, p0=0.05, p1=0.01):
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
    lut[:128, 0] = 255
    lut[:i1, 1] = 255
    # positive
    lut[128:, 2] = 255
    lut[-i1:, 0] = 255
    # alpha
    lut[:126, 3] = 255
    lut[126, 3] = 127
    lut[129, 3] = 127
    lut[130:, 3] = 255

    return pmap, lut, vmax

