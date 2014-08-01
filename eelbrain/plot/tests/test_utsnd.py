'''
Created on Jul 11, 2013

@author: christian
'''
import os

import mne
import numpy as np
from mne import Evoked
from mne.io import Raw

from eelbrain import datasets, plot


def test_plot_butterfly():
    "Test plot.Butterfly"
    plot.configure_backend(False, False)
    ds = datasets.get_rand(utsnd=True)
    p = plot.Butterfly('utsnd', ds=ds)
    p.close()
    p = plot.Butterfly('utsnd', 'A%B', ds=ds)
    p.close()

def test_plot_array():
    "Test plot.Array"
    plot.configure_backend(False, False)
    ds = datasets.get_rand(utsnd=True)
    p = plot.Array('utsnd', 'A%B', ds=ds)
    p.close()
    p = plot.Array('utsnd', ds=ds)
    p.close()

def test_plot_mne_evoked():
    "Test plotting evoked from the mne sample dataset"
    data_path = mne.datasets.sample.data_path()
    evoked_path = os.path.join(data_path, 'MEG', 'sample',
                               'sample_audvis-ave.fif')
    evoked = Evoked(evoked_path, setno="Left Auditory")
    p = plot.Array(evoked)
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
    p = plot.Array(epochs)
    p.close()

    # with model
    p = plot.Array(epochs, events[:, 2])
    p.close()
