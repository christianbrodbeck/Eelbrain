# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>

import os

from nose.tools import eq_
from numpy.testing import assert_array_equal, assert_array_almost_equal

import mne
from mne import pick_types

from eelbrain import load

from ...tests.test_data import assert_dataobj_equal


data_path = mne.datasets.sample.data_path()
meg_path = os.path.join(data_path, 'MEG', 'sample')
raw_path = os.path.join(meg_path, 'sample_audvis_filt-0-40_raw.fif')
evt_path = os.path.join(meg_path, 'sample_audvis_filt-0-40_raw-eve.fif')


def test_load_fiff_from_raw():
    "Test loading data from a fiff raw file"
    ds = load.fiff.events(raw_path)

    # test separate events
    ds_evt = load.fiff.events(events=evt_path)
    ds_evt.name = ds.name
    assert_dataobj_equal(ds_evt, ds)

    # add epochs as ndvar
    ds = ds.sub('trigger == 32')
    ds_ndvar = load.fiff.add_epochs(ds, -0.1, 0.3, decim=10, data='mag',
                                    proj=False, reject=2e-12)
    meg = ds_ndvar['meg']
    eq_(meg.ndim, 3)
    data = meg.get_data(('case', 'sensor', 'time'))
    eq_(data.shape, (14, 102, 6))

    # compare with mne epochs
    ds_mne = load.fiff.add_mne_epochs(ds, -0.1, 0.3, decim=10, proj=False,
                                      reject={'mag': 2e-12})
    epochs = ds_mne['epochs']
    picks = pick_types(epochs.info, meg='mag')
    mne_data = epochs.get_data()[:, picks]
    eq_(meg.sensor.names, [epochs.info['ch_names'][i] for i in picks])
    assert_array_equal(data, mne_data)

    # with proj
    meg = load.fiff.epochs(ds, -0.1, 0.3, decim=10, data='mag', proj=True,
                           reject=2e-12)
    epochs = load.fiff.mne_epochs(ds, -0.1, 0.3, decim=10, proj=True,
                                  reject={'mag': 2e-12})
    picks = pick_types(epochs.info, meg='mag')
    mne_data = epochs.get_data()[:, picks]
    assert_array_almost_equal(meg.x, mne_data, 10)
