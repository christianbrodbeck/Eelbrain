# Author: Proloy Das <proloy@umd.edu>
import numpy as np
import warnings
from .._data_obj import NDVar


def _to_MNI152(trans):
    """transfrom from MNI305 space (fsaverage space) to MNI152

    parameters
    ----------
    trans: ndarray
        the affine transform

    Notes
    -----
    uses approximate transformation mentioned `Link here <https://surfer.nmr.mgh.harvard.edu/fswiki/CoordinateSystems>`_
    """
    T = np.array([[ 0.9975, -0.0073,  0.0176, -0.0429],
                     [ 0.0146,  1.0009, -0.0024,  1.5496],
                     [-0.0130, -0.0093,  0.9971,  1.1840],
                     [      0,       0,       0,       1]])
    trans = np.dot(T, trans)

    return trans


def _save_stc_as_volume(fname, ndvar, src, dest='mri', mri_resolution=False, mni_correction=False):
    """Save a volume source estimate in a NIfTI file.

    Parameters
    ----------
    fname : string | None
        The name of the generated nifti file. If None, the image is only
        returned and not saved.
    ndvar : NDVar
        The source estimate
    src : list | string
        The list of source spaces (should actually be of length 1). If
        string, it is the filepath.

    dest : 'mri' | 'surf'
        If 'mri' the volume is defined in the coordinate system of
        the original T1 image. If 'surf' the coordinate system
        of the FreeSurfer surface is used (Surface RAS).
    mri_resolution: bool
        It True the image is saved in MRI resolution.
        WARNING: if you have many time points the file produced can be
        huge.
    mni_correction: bool
        Set to True to convert RAS coordinates of a voxel in MNI305 space (fsaverage space)
        to MNI152 space via updating the affine transformation matrix.

    Returns
    -------
    img : instance Nifti1Image
        The image object.
    """
    # check if file name
    if isinstance(src, str):
        print(('Reading src file %s...' %src))
        from mne import read_source_spaces
        src = read_source_spaces(src)
    else:
        src = src

    src_type = src[0]['type']
    if src_type != 'vol':
        raise ValueError('You need a volume source space. Got type: %s.'
                         % src_type)

    if ndvar.has_dim('space'):
        ndvar = ndvar.norm('space')

    if ndvar.has_dim('time'):
        data = ndvar.get_data(('source', 'time'))
    else:
        data = ndvar.get_data(('source', np.newaxis))

    n_times = data.shape[1]
    shape = src[0]['shape']
    shape3d = (shape[2], shape[1], shape[0])
    shape = (n_times, shape[2], shape[1], shape[0])
    vol = np.zeros(shape)

    if mri_resolution:
        mri_shape3d = (src[0]['mri_height'], src[0]['mri_depth'],
                       src[0]['mri_width'])
        mri_shape = (n_times, src[0]['mri_height'], src[0]['mri_depth'],
                     src[0]['mri_width'])
        mri_vol = np.zeros(mri_shape)
        interpolator = src[0]['interpolator']

    n_vertices_seen = 0
    for this_src in src:  # loop over source instants, which is basically one element only!
        assert tuple(this_src['shape']) == tuple(src[0]['shape'])
        mask3d = this_src['inuse'].reshape(shape3d).astype(np.bool)
        n_vertices = np.sum(mask3d)

        for k, v in enumerate(vol):  # loop over time instants
            stc_slice = slice(n_vertices_seen, n_vertices_seen + n_vertices)
            v[mask3d] = data[stc_slice, k]

        n_vertices_seen += n_vertices

    if mri_resolution:
        for k, v in enumerate(vol):
            mri_vol[k] = (interpolator * v.ravel()).reshape(mri_shape3d)
        vol = mri_vol

    vol = vol.T

    if mri_resolution:
        affine = src[0]['vox_mri_t']['trans'].copy()
    else:
        affine = src[0]['src_mri_t']['trans'].copy()
    if dest == 'mri':
        affine = np.dot(src[0]['mri_ras_t']['trans'], affine)

    affine[:3] *= 1e3
    if mni_correction:
        affine = _to_MNI152(affine)

    # write the image in nifty format
    import nibabel as nib
    header = nib.nifti1.Nifti1Header()
    header.set_xyzt_units('mm', 'msec')
    if ndvar.has_dim('time'):
        header['pixdim'][4] = 1e3 * ndvar.time.tstep
    else:
        header['pixdim'][4] = None
    with warnings.catch_warnings(record=True):  # nibabel<->numpy warning
        img = nib.Nifti1Image(vol, affine, header=header)
        if fname is not None:
            nib.save(img, fname)
    return img


def _safe_get_data(img):
    """Get the data in the Nifti1Image object avoiding non-finite values

    Parameters
    ----------
    img: Nifti image/object
        Image to get data.

    Returns
    -------
    data: numpy array
        get_data() return from Nifti image.
        # inspired from nilearn._utils.niimg._safe_get_data
    """
    data = img.get_data()
    non_finite_mask = np.logical_not(np.isfinite(data))
    if non_finite_mask.sum() > 0: # any non_finite_mask values?
        data[non_finite_mask] = 0

    return data


def _fast_abs_percentile(ndvar, percentile=80):
    """A fast version of the percentile of the absolute value.

    Parameters
    ----------
    data: ndvar
        The input data
    percentile: number between 0 and 100
        The percentile that we are asking for

    Returns
    -------
    value: number
        The score at percentile

    Notes
    -----
    This is a faster, and less accurate version of
    scipy.stats.scoreatpercentile(np.abs(data), percentile)
    # inspired from nilearn._utils.extmath.fast_abs_percentile
    """
    data = abs(ndvar.x)
    data = data.ravel()
    index = int(data.size * .01 * percentile)
    try:
        # Partial sort: faster than sort
        # partition is available only in numpy >= 1.8.0
        from numpy import partition
        data = partition(data, index)
    except ImportError:
        data.sort()

    return data[index]
