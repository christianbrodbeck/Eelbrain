from eelbrain import *

mark = []
mark = ['MEG 2313']

ds = datasets.get_mne_sample(sns=True)
g = gui.select_epochs(ds, 'meg', mark=mark)
