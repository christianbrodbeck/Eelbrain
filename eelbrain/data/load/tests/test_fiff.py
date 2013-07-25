# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>

import mne
import os

from eelbrain.eellab import load

data_path = mne.datasets.sample.data_path()
raw_path = os.path.join(data_path, 'MEG', 'sample',
                        'sample_audvis_filt-0-40_raw.fif')


def test_load_fiff_from_raw():
    "Test loading data from a fiff raw file"
    ds = load.fiff.events(raw_path)
    ds = ds.subset('eventID == 32')
    ds = load.fiff.add_epochs(ds, tstart=-0.1, tstop=0.6, baseline=None,
                              decim=10, data='mag', reject={'mag': 2e-12},
                              target='meg')
    meg = ds['meg']
    assert meg.ndim == 3, "Loading evoked data from fiff raw"
    data = meg.get_data(('case', 'sensor', 'time'))
    assert data.shape == (16, 102, 11), "Loading evoked data from fiff raw"
