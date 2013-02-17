'''
Created on Oct 21, 2012

@author: christian
'''
import numpy as np

try:
    import mdp
except:
    pass

import mne


__all__ = ['split_label']



def split_label(label, fwd_fname, name1='{name}_post', name2='{name}_ant',
                divide=np.median):
    """
    Splits an mne Label object into two labels along its principal axis. The
    principle axis is determined using principal component analysis (PCA).
    Returns 2 mne Label instances ``(label1, label2)``.


    Parameters
    ----------

    label : mne Label
        Source label, which is to be split.
    fwd_fname : str(path)
        Filename to a fwd file (used to load the source space which is needed
        to constrain the label points to those that are relevant for the
        source space).
    name1, name2 : str
        Name for the new labels. '{name}' will be formatted with the input
        label's name.
    divide : func
        Function that takes the one-dimensional array of point location among
        the principal axis and returns the point at which the points should be
        split (default is the median).

    """
    source_space = mne.read_source_spaces(fwd_fname)
    if isinstance(label, basestring):
        label = mne.read_label(label)

    hemi = (label.hemi == 'rh')
    ss_vert = source_space[hemi]['vertno']
    idx = np.array(map(ss_vert.__contains__, label.vertices))

    # centered label coordinates
    cpos = label.pos - np.mean(label.pos, 0)

    # project all label coords onto pca-0 of the label's source space coords
    cpos_i = cpos[idx]
    node = mdp.nodes.PCANode(output_dim=1)
    node.train(cpos_i)
    proj = node.execute(cpos)[:, 0]
    if node.v[1, 0] < 0:
        proj *= -1

    div = divide(proj[idx])

    ip = proj < div
    ia = proj >= div

    out = []
    for idx, name in [(ip, name1), (ia, name2)]:
        label_name = name.format(name=label.name)
        lblout = mne.label.Label(label.vertices[idx], label.pos[idx],
                                 label.values[idx], label.hemi,
                                 comment=label.comment, name=label_name)
        out.append(lblout)

    return tuple(out)
