# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from os.path import join

import mne
from eelbrain import gui, load
from eelbrain._utils.testing import gui_test, TempDir
from eelbrain._utils.system import IS_OSX


@gui_test
def test_select_components():
    "Test Select-Epochs GUI Document"
    tempdir = TempDir()
    PATH = join(tempdir, 'test-ica.fif')

    data_path = mne.datasets.testing.data_path()
    raw_path = join(data_path, 'MEG', 'sample', 'sample_audvis_trunc_raw.fif')
    raw = mne.io.Raw(raw_path, preload=True).pick_types('mag', stim=True)
    ds = load.fiff.events(raw)
    ds['epochs'] = load.fiff.mne_epochs(ds, tmax=0.1)
    ica = mne.preprocessing.ICA(0.95, fit_params={'tol': 0.001})
    ica.fit(raw)
    ica.save(PATH)

    frame = gui.select_components(PATH, ds)
    if not IS_OSX:  # FIXME:  Linux and Windows
        return
    frame.model.toggle(1)
    frame.OnSave(None)
    ica = mne.preprocessing.read_ica(PATH)
    assert ica.exclude == [1]

    frame.OnUndo(None)
    frame.OnSave(None)
    ica = mne.preprocessing.read_ica(PATH)
    assert ica.exclude == []

    frame.Close()
