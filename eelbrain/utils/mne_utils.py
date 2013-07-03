'''
Created on Feb 12, 2013

@author: Christian M Brodbeck
'''

import os

import numpy as np
from matplotlib.mlab import PCA

import mne
from mne import Label

from . import intervals


def is_fake_mri(mri_dir):
    """Check whether a directory is a fake MRI subject directory

    Parameters
    ----------
    mri_dir : str(path)
        Path to a directory.

    Returns
    -------
    True is `mri_dir` is a fake MRI directory.

    """
    items = os.listdir(mri_dir)
    # need to contain:
    nc = [c for c in ['bem', 'label', 'surf'] if c not in items]
    # does not contain:
    c = [c for c in ['mri', 'src', 'stats'] if c in items]
    if c or nc:
        return False
    else:
        return True


def split_label(label, source_space=None, axis='pca', pieces=3):
    """
    Split an mne Label object into several parts

    Project all points included in a label onto a specified axis, and then
    evenly divide the points along this axis into several Label objects.

    Parameters
    ----------
    label : mne Label
        Source label, which is to be split.
    source_space : None | dict | str(path)
        Mne source space or a file containing a source space (*-src.fif,
        *-fwd.fif). The source space is needed to constrain the label to
        those points that are relevant for the source space.
    axis : 'pca' | 0 | 1 | 2
        The axis along which to split the label. For axis='pca', use the first
        component in a principal component analysis of all coordinates. The
        integers 0, 1 and 2 specify axes in the right/anterior/superior
        coordinate system. In each case, the actual axis will be determined as
        the direction between the label's most extreme points along the given
        dimension. For example, for axis=0, the projection axis will be the
        vector from the left-most to the right-most point in the label.
    pieces : int >= 2
        Number of labels to create.

    Returns
    -------
    labels : list of Label (len = pieces)
        The labels, starting from the lowest to the highest end of the
        projection axis.
    """
    if isinstance(source_space, basestring):
        source_space = mne.read_source_spaces(source_space)
    if isinstance(label, basestring):
        label = mne.read_label(label)

    # centered label coordinates
    cpos_all = label.pos - np.mean(label.pos, 0)

    # find label coordinates that are in the source space (if given)
    if source_space is None:
        cpos_in_src = cpos_all
    else:
        hemi = (label.hemi == 'rh')
        ss_vert = source_space[hemi]['vertno']
        idx_in_src = np.array([v in ss_vert for v in label.vertices])
        cpos_in_src = cpos_all[idx_in_src]


    if axis == 'pca':
        # project all label coords onto pca-0 of the label's source space coords
        pca = PCA(cpos_in_src)
        proj_in_src = pca.Y[:, 0]
        proj_all = pca.project(cpos_all)[:, 0]
    elif axis in (0, 1, 2):
        idx_min = np.argmin(cpos_in_src[:, axis])
        idx_max = np.argmax(cpos_in_src[:, axis])
        ax_vect = cpos_in_src[idx_max] - cpos_in_src[idx_min]
        ax_vect /= np.linalg.norm(ax_vect)
        proj_in_src = np.sum(ax_vect * cpos_in_src, 1)
        proj_all = np.sum(ax_vect * cpos_all, 1)
    else:
        err = "The axis parameter must be 'pca' or 0, 1 or 2, not %s." % axis
        raise ValueError(err)

    # find locations to cut the label
    limits = np.linspace(proj_in_src.min(), proj_in_src.max(), pieces + 1)
    limits[0] = proj_all.min() - 1
    limits[-1] = proj_all.max() + 1

    labels = []
    for i, j in intervals(limits):
        idx = np.logical_and(proj_all >= i, proj_all < j)
        vert = label.vertices[idx]
        pos = label.pos[idx]
        values = label.values[idx]
        hemi = label.hemi
        comment = label.comment
        lbl = Label(vert, pos, values, hemi, comment=comment)
        labels.append(lbl)

    return labels
