'''
Created on Oct 9, 2012

@author: christian
'''

import os
import re
import tempfile

import numpy as np

__all__ = ['MarkerFile']


def fix_ptb_eeg_events(raw):
    """Fix events from a vmrk file recorded with psychtoolbox/stim tracker

    Parameters
    ----------
    raw : RawBrainVision
        MNE-Python object with psychtoolbox events.
    """
    events = raw.get_brainvision_events()

    if not np.all(events[:, 1] == 1):
        err = ("Not KIT psychtoolbox input data (not all durations are 1)")
        raise ValueError(err)

    # invert trigger codes
    events[:, 2] = np.invert(events[:, 2].astype(np.uint8))

    # extend duration until next tigger start
    events[:-1, 1] = np.diff(events[:, 0])

    # remove 0 events
    idx = np.nonzero(events[:, 2])[0]
    events = events[idx]

    raw.set_brainvision_events(events)


class MarkerFile:
    """
    Attributes
    ----------

    points : np.array
        array with shape point by coordinate (x, y, z)

    path : str
        path to the temporary file containing the simplified marker file for
        input to mne_kit2fiff

    """
    def __init__(self, path):
        """
        path : str
            Path to marker avg file (saved as text form MEG160).

        """
        self.src_path = path

        # pattern by Tal:
        p = re.compile(r'Marker \d:   MEG:x= *([\.\-0-9]+), y= *([\.\-0-9]+), z= *([\.\-0-9]+)')
        str_points = p.findall(open(path).read())
        txt = '\n'.join(map('\t'.join, str_points))
        self.points = np.array(str_points, dtype=float)

        fd, self.path = tempfile.mkstemp(suffix='hpi', text=True)
        f = os.fdopen(fd, 'w')
        f.write(txt)
        f.close()

    def __del__(self):
        os.remove(self.path)

    def __repr__(self):
        return 'MarkerFile(%r)' % self.src_path

    def plt(self, marker='+k'):
        self.plot_mpl(marker=marker)

    def plot_mpl(self, marker='+k', ax=None, title=True):
        "returns: axes object with 3d plot"
        import matplotlib.pyplot as plt

        if ax is None:
            fig = plt.figure()
            ax = fig.gca(projection='3d')

        ax.plot(self.points[:, 0], self.points[:, 1], self.points[:, 2], marker)
        for i, (x, y, z) in enumerate(self.points):
            ax.text(x, y, z, str(i))

        xmin, ymin, zmin = self.points.min(0) - 1
        xmax, ymax, zmax = self.points.max(0) + 1
        ax.set_xlim3d(xmin, xmax)
        ax.set_ylim3d(ymin, ymax)
        ax.set_zlim3d(zmin, zmax)

        if title:
            if title is True:
                title = os.path.basename(self.src_path)
            ax.set_title(str(title))

        return ax
