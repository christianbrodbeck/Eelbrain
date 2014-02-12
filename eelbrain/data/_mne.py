from itertools import izip

import numpy as np

from mne.label import _n_colors, Label


def labels_from_clusters(clusters, names=None):
    """Create Labels from source space clusters

    Parameters
    ----------
    clusters : NDVar
        NDVar which is non-zero on the cluster. Can have a case dimension.
    names : None | list of str | str
        Label names corresponding to clusters (default is "clusterX").

    Returns
    -------
    labels : list of mne.Label
        One label for each cluster.
    """
    source = clusters.source
    source_space = clusters.source.get_source_space()
    subject = source.subject

    if clusters.has_case:
        n_clusters = len(clusters)
        x = clusters.x
    else:
        n_clusters = 1
        x = clusters.x[None, :]

    if isinstance(names, basestring) and n_clusters == 1:
        names = [names]
    elif names is None:
        names = ("cluster%i" % i for i in xrange(n_clusters))
    elif len(names) != n_clusters:
        err = "Number of names difference from number of clusters."
        raise ValueError(err)

    colors = _n_colors(n_clusters)
    labels = []
    for i, color, name in izip(xrange(n_clusters), colors, names):
        idx = (x[i] != 0)
        where = np.nonzero(idx)[0]
        src_in_lh = (where < source.lh_n)
        if np.all(src_in_lh):
            hemi = 'lh'
            hemi_vertices = source.lh_vertno
        elif np.any(src_in_lh):
            raise ValueError("Can't have clusters spanning both hemispheres")
        else:
            hemi = 'rh'
            hemi_vertices = source.rh_vertno
            where -= source.lh_n
        vertices = hemi_vertices[where]

        label = Label(vertices, hemi=hemi, name=name, subject=subject,
                      color=color, src=source_space)
        labels.append(label)

    return labels
