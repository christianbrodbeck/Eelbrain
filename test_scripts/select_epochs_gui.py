from eelbrain import *

set_log_level('warning', 'mne')

ds = datasets.get_mne_sample(sns=True)
g = gui.select_epochs(ds, 'sns')
