# excerpt from development version of nibabel (>1.3.0) (MIT license)
from __future__ import division

import numpy as np


def read_annot(filepath, orig_ids=False):
    """Read in a Freesurfer annotation from a .annot file.

    Parameters
    ----------
    filepath : str
        Path to annotation file.
    orig_ids : bool
        Whether to return the vertex ids as stored in the annotation
        file or the positional colortable ids. With orig_ids=False
        vertices with no id have an id set to -1.

    Returns
    -------
    labels : ndarray, shape (n_vertices,)
        Annotation id at each vertex. If a vertex does not belong
        to any label and orig_ids=False, its id will be set to -1.
    ctab : ndarray, shape (n_labels, 5)
        RGBA + label id colortable array.
    names : list of str
        The names of the labels. The length of the list is n_labels.
    """
    with open(filepath, "rb") as fobj:
        dt = ">i4"
        vnum = np.fromfile(fobj, dt, 1)[0]
        data = np.fromfile(fobj, dt, vnum * 2).reshape(vnum, 2)
        labels = data[:, 1]

        ctab_exists = np.fromfile(fobj, dt, 1)[0]
        if not ctab_exists:
            raise Exception('Color table not found in annotation file')
        n_entries = np.fromfile(fobj, dt, 1)[0]
        if n_entries > 0:
            length = np.fromfile(fobj, dt, 1)[0]
            orig_tab = np.fromfile(fobj, '>c', length)
            orig_tab = orig_tab[:-1]

            names = list()
            ctab = np.zeros((n_entries, 5), np.int)
            for i in xrange(n_entries):
                name_length = np.fromfile(fobj, dt, 1)[0]
                name = np.fromfile(fobj, "|S%d" % name_length, 1)[0]
                names.append(name)
                ctab[i, :4] = np.fromfile(fobj, dt, 4)
                ctab[i, 4] = (ctab[i, 0] + ctab[i, 1] * (2 ** 8) +
                              ctab[i, 2] * (2 ** 16) +
                              ctab[i, 3] * (2 ** 24))
        else:
            ctab_version = -n_entries
            if ctab_version != 2:
                raise Exception('Color table version not supported')
            n_entries = np.fromfile(fobj, dt, 1)[0]
            ctab = np.zeros((n_entries, 5), np.int)
            length = np.fromfile(fobj, dt, 1)[0]
            _ = np.fromfile(fobj, "|S%d" % length, 1)[0]  # Orig table path
            entries_to_read = np.fromfile(fobj, dt, 1)[0]
            names = list()
            for i in xrange(entries_to_read):
                _ = np.fromfile(fobj, dt, 1)[0]  # Structure
                name_length = np.fromfile(fobj, dt, 1)[0]
                name = np.fromfile(fobj, "|S%d" % name_length, 1)[0]
                names.append(name)
                ctab[i, :4] = np.fromfile(fobj, dt, 4)
                ctab[i, 4] = (ctab[i, 0] + ctab[i, 1] * (2 ** 8) +
                              ctab[i, 2] * (2 ** 16))
        ctab[:, 3] = 255
    if not orig_ids:
        ord = np.argsort(ctab[:, -1])
        mask = labels != 0
        labels[~mask] = -1
        labels[mask] = ord[np.searchsorted(ctab[ord, -1], labels[mask])]
    return labels, ctab, names


def write_annot(filepath, labels, ctab, names):
    """Write out a Freesurfer annotation file.

    See:
    http://ftp.nmr.mgh.harvard.edu/fswiki/LabelsClutsAnnotationFiles#Annotation

    Parameters
    ----------
    filepath : str
        Path to annotation file to be written
    labels : ndarray, shape (n_vertices,)
        Annotation id at each vertex.
    ctab : ndarray, shape (n_labels, 5)
        RGBA + label id colortable array.
    names : list of str
        The names of the labels. The length of the list is n_labels.
    """
    with open(filepath, "wb") as fobj:
        dt = ">i4"
        vnum = len(labels)

        def write(num, dtype=dt):
            np.array([num]).astype(dtype).tofile(fobj)

        def write_string(s):
            write(len(s))
            write(s, dtype='|S%d' % len(s))

        # vtxct
        write(vnum)

        # convert labels into coded CLUT values
        clut_labels = ctab[:, -1][labels]
        clut_labels[np.where(labels == -1)] = 0

        # vno, label
        data = np.vstack((np.array(range(vnum)).astype(dt),
                          clut_labels.astype(dt))).T
        data.byteswap().tofile(fobj)

        # tag
        write(1)

        # ctabversion
        write(-2)

        # maxstruc
        write(np.max(labels) + 1)

        # File of LUT is unknown.
        write_string('NOFILE')

        # num_entries
        write(ctab.shape[0])

        for ind, (clu, name) in enumerate(zip(ctab, names)):
            write(ind)
            write_string(name)
            for val in clu[:-1]:
                write(val)
