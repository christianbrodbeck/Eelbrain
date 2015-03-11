from eelbrain import *

set_log_level('debug')
set_log_level('warning', 'mne')

mark = []
mark = ['MEG 2313']

ds = datasets.get_mne_sample(sns=True)
g = gui.select_epochs(ds, 'sns', mark=mark)
