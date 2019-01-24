# skip test: data unavailable
from pathlib import Path

import mne.coreg
# from mne.coreg import read_mri_cfg

root = Path('~/Data/Mouse').expanduser()
mri_dir = root / 'mri'
# mne.coreg.create_default_subject('/Applications', subjects_dir=mri_dir)
for subject_path in mri_dir.glob('S*'):
    cfg = mne.coreg.read_mri_cfg(subject_path.name, mri_dir)
    scale = cfg['scale']
    mne.scale_mri('fsaverage', subject_path.name, scale, overwrite=True, subjects_dir=mri_dir, labels=False)
