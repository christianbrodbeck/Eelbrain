# skip test
"""Compute DSS on Alex's Toolbox demo data"""
from eelbrain import *


# If the raw file is already filtered, the events can be loaded directly with
# the filename
ds = load.fiff.events('DATA/R2290_HAYO_P3H_1-8-raw.fif')
# select the desired events
ds = ds.sub("trigger == 167")
# add MEG trial data to the dataset
ds['meg'] = load.fiff.epochs(ds, 0, 60)

# DSS from
todss, fromdss = dss(ds['meg'])

# plot the DSS topography
plot.Topomap(fromdss[:, :6], '.dss', h=2, ncol=6)
# save the DSS for later use
save.pickle((todss, fromdss), 'DATA/R2290_HAYO_P3H_DSS.pickle')
