'''
Created on Jul 11, 2013

@author: christian
'''
import os

import mne
import numpy as np
from mne import Evoked
from mne.io import Raw

from eelbrain import datasets, plot, testnd


def test_plot_butterfly():
    "Test plot.Butterfly"
    ds = datasets.get_uts(utsnd=True)
    p = plot.Butterfly('utsnd', ds=ds, show=False)
    p.close()
    p = plot.Butterfly('utsnd', 'A%B', ds=ds, show=False)
    p.close()

def test_plot_array():
    "Test plot.Array"
    ds = datasets.get_uts(utsnd=True)
    p = plot.Array('utsnd', 'A%B', ds=ds, show=False)
    p.close()
    p = plot.Array('utsnd', ds=ds, show=False)
    p.close()

def test_plot_mne_evoked():
    "Test plotting evoked from the mne sample dataset"
    data_path = mne.datasets.sample.data_path()
    evoked_path = os.path.join(data_path, 'MEG', 'sample',
                               'sample_audvis-ave.fif')
    evoked = Evoked(evoked_path, "Left Auditory")
    p = plot.Array(evoked, show=False)
    p.close()

def test_plot_mne_epochs():
    "Test plotting epochs from the mne sample dataset"
    # find paths
    data_path = mne.datasets.sample.data_path()
    raw_path = os.path.join(data_path, 'MEG', 'sample',
                            'sample_audvis_filt-0-40_raw.fif')
    events_path = os.path.join(data_path, 'MEG', 'sample',
                               'sample_audvis_filt-0-40_raw-eve.fif')

    # read epochs
    raw = Raw(raw_path)
    events = mne.read_events(events_path)
    idx = np.logical_or(events[:, 2] == 5, events[:, 2] == 32)
    events = events[idx]
    epochs = mne.Epochs(raw, events, None, -0.1, 0.3)

    # grand average
    p = plot.Array(epochs, show=False)
    p.close()

    # with model
    p = plot.Array(epochs, events[:, 2], show=False)
    p.close()


def test_plot_results():
    "Test plotting test results"
    ds = datasets.get_uts(True)

    # ANOVA
    res = testnd.anova('utsnd', 'A*B*rm', ds=ds, samples=0, pmin=0.05)
    p = plot.Array(res, show=False)
    p.close()
    res = testnd.anova('utsnd', 'A*B*rm', ds=ds, samples=2, pmin=0.05)
    p = plot.Array(res, show=False)
    p.close()

    # Correlation
    res = testnd.corr('utsnd', 'Y', 'rm', ds=ds)
    p = plot.Array(res, show=False)
    p.close()
    res = testnd.corr('utsnd', 'Y', 'rm', ds=ds, samples=10, pmin=0.05)
    p = plot.Array(res, show=False)
    p.close()
