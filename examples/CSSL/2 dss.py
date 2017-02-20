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


# DSS for multiple conditions
# ---------------------------
# To compute the DSS for multiple conditions, concatenate the data for
# different conditions:
ds = load.fiff.events('DATA/R2290_HAYO_P3H_1-8-raw.fif')
# load data for all 6 trials (each trial 10 s long)
ds['meg'] = load.fiff.epochs(ds, 0, 10)
# extract NDVar for each condition; each condition has 3 trials:
c167 = ds[ds.eval("trigger == 167"), 'meg']
c163 = ds[ds.eval("trigger == 163"), 'meg']
# concatenate trials along the time axis:
data = concatenate((c163, c167))
# now data has 3 cases ("trials") of length 20 s.
# compute DSS using both conditions:
todss, fromdss = dss(data)
