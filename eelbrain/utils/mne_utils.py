'''
Created on Feb 12, 2013

@author: Christian M Brodbeck
'''

import os



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
    nc = [c for c in ['bem', 'label', 'surf'] if c not in items]
    c = [c for c in ['mri', 'src', 'stats'] if c in items]
    if c or nc:
        return False
    else:
        return True
