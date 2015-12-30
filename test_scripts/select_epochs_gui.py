from eelbrain import *

mark = []
mark = ['MEG 2313']
mark = ['MEG 1421']
mark = ['1421']

ds = datasets.get_mne_sample(sns=True)
g = gui.select_epochs(ds, 'meg', mark=mark, vlim=2e-12)
