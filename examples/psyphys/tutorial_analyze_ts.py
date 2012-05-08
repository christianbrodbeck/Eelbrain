"""
Illustration for collecting and plotting a time-series

"""
import eelbrain.psyphys as pp
import eelbrain.wxgui.psyphys as ppgui
from eelbrain.eellab import *


# load the experiment (saved in 'tutorial_import.py`)
e = pickle.load(open(u'/Users/christian/Data/tutorial_scr.eelbrain'))

# collect the statistics
attach(e.variables)
ds = pp.collect.timeseries(subject * magnitude, e.SCRs, e.event, sr=20, 
                           mode='mw', windur=.5, tstart=-.2, tend=1.5)

attach(ds)
p = plot.uts.stat(Y, magnitude)
p = plot.uts.stat(Y, magnitude, 'all')



