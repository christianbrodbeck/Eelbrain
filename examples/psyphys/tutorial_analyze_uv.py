"""
Illustration for analyzing a univariate variable

"""
import eelbrain.psyphys as pp
import eelbrain.wxgui.psyphys as ppgui
from eelbrain.eellab import *


# load the experiment (saved in 'tutorial_import.py`)
e = pickle.load(open(u'/Users/christian/Data/tutorial_scr.eelbrain'))

# collect the statistics
attach(e.variables)
ds = pp.collect.timewindow(subject * condition, e.SCRs, e.event, tstart=.1, tend=.6)

print ds[:20]
print '...'

attach(ds)

# make a boxplot
fig = plot.uv.boxplot(Y, condition, match=subject)

# test difference
print test.pairwise(Y, condition, match=subject)

