import os
import timeit

import mne
from eelbrain.lab import datasets, save

mne.set_log_level('warning')

fname = 'temp.pickled'
if not os.path.exists(fname):
    ds = datasets.get_mne_sample(-0.1, 0.2, src='ico', sub="modality == 'A'")
    source = ds['src'].source
    y = ds['src'][0].x
    save.pickle((y, source), fname)


setup = '''
from itertools import izip
import numpy as np
import scipy as sp
from eelbrain.lab import stats, load

y, source = load.unpickle(%r)

out = np.empty(y.shape, np.uint32)
bin_buff = np.empty(y.shape, np.bool_)
bin_buff2 = np.empty(y.shape, np.bool_)
int_buff = np.empty(y.shape, np.uint32)
threshold = 1
tail = 0
struct = sp.ndimage.generate_binary_structure(y.ndim, 1)
struct[::2] = False
all_adjacent = False
flat_shape = (y.shape[0], np.prod(y.shape[1:]))
connectivity_src, connectivity_dst = source.connectivity().T
criteria=None
''' % fname

stmt = '''
stats.testnd._label_clusters(y, out, bin_buff, bin_buff2, int_buff, threshold,
                             tail, struct, all_adjacent, flat_shape,
                             connectivity_src, connectivity_dst, criteria)
'''

timer = timeit.Timer(stmt, setup)
times = timer.repeat(100, 1)
print times
print "min = %s" % min(times)
print "avg of 10 lowest = %s" % (sum(sorted(times)[:10]) / 10)
