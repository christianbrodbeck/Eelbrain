# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from eelbrain import *

# turn off multiprocessing
configure(0)
ds = datasets._get_continuous(10000)

timeit boosting(ds['y'], ds['x1'], 0, .5)
