# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import os

from eelbrain import *
import mne


PATH = 'test-ica.fif'

ds = datasets.get_mne_sample()
epochs = ds['epochs'].pick_types('mag')
if os.path.exists(PATH):
    ica = mne.preprocessing.read_ica(PATH)
else:
    ica = mne.preprocessing.ICA(0.95)
    ica.fit(epochs)
    ica.save(PATH)
g = gui.select_components(PATH, epochs, ds['index'])
