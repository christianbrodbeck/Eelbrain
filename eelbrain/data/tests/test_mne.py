"""Test mne interaction"""

import os

import mne

from eelbrain.lab import load, combine


data_path = mne.datasets.sample.data_path()
mri_sdir = os.path.join(data_path, 'subjects')
meg_dir = os.path.join(data_path, 'MEG', 'sample')
stc_path = os.path.join(meg_dir, 'fsaverage_audvis-meg-eeg-lh.stc')

def test_source_estimate():
    "Test SourceSpace dimension"
    stc = mne.read_source_estimate(stc_path, 'fsaverage')
    ndvar = load.fiff.stc_ndvar(stc, 'fsaverage', 'ico-5', mri_sdir)
    ndvar.source.connectivity()

    ndvar2 = ndvar.copy()
    big_ndvar = combine((ndvar, ndvar2))
