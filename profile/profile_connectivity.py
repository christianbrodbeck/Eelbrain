import cProfile
import os
import pstats

import mne
import numpy as np
from eelbrain.lab import *

fname = 'profile_of_connectivity.profile'

mne.set_log_level('warning')
stats.testnd.multiprocessing = False

ds = datasets.get_mne_sample(-0.1, 0.2, src='ico', sub="modality == 'A'")

code = '''
np.random.seed(0)
res = testnd.ttest_ind('src', 'side', 'L', 'R', ds=ds, samples=5, pmin=0.5,
                       tstart=0.05, mintime=0.02, minsource=10)
'''

cProfile.run(code, fname)

p = pstats.Stats(fname)
p.sort_stats('cumulative')
p.print_stats(20)
