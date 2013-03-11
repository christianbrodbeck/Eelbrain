'''
Created on Oct 21, 2012

@author: christian
'''
import numpy as np

try:
    import mdp
    _has_mdp = True
except:
    from matplotlib.mlab import PCA
    _has_mdp = False

import mne


__all__ = ['split_label']



def split_label(label, source_space, name1='{name}_post', name2='{name}_ant',
                divide=np.median):
    """
    Splits an mne Label object into two labels along its principal axis. The
    principle axis is determined using principal component analysis (PCA).
    Returns 2 mne Label instances ``(label1, label2)``.


    Parameters
    ----------

    label : mne Label
        Source label, which is to be split.
    source_space : dict | str(path)
        Mne source space or a file containing a source space (*-src.fif,
        *-fwd.fif). The source space is needed to constrain the label to
        those points that are relevant for the source space.
    name1, name2 : str
        Name for the new labels. '{name}' will be formatted with the input
        label's name.
    divide : func
        Function that takes the one-dimensional array of point location among
        the principal axis and returns the point at which the points should be
        split (default is the median).

    """
    if isinstance(source_space, basestring):
        source_space = mne.read_source_spaces(source_space)
    if isinstance(label, basestring):
        label = mne.read_label(label)

    hemi = (label.hemi == 'rh')
    ss_vert = source_space[hemi]['vertno']
    idx = np.array(map(ss_vert.__contains__, label.vertices))

    # centered label coordinates
    cpos = label.pos - np.mean(label.pos, 0)

    # project all label coords onto pca-0 of the label's source space coords
    cpos_i = cpos[idx]
    if _has_mdp:
        node = mdp.nodes.PCANode(output_dim=1)
        node.train(cpos_i)
        proj = node.execute(cpos)[:, 0]
        if node.v[1, 0] < 0:
            proj *= -1
    else:
        pca = PCA(cpos_i)
        proj = pca.project(cpos)[:, 0]

    div = divide(proj)

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
