# skip test
"""Compute DSS on Alex's Toolbox demo data"""
from eelbrain import *


# If the raw file is already filtered, the events can be loaded directly with
# the filename
ds = load.fiff.events('DATA/R2290_HAYO_P3H_1-8-raw.fif')
# select the desired events
ds = ds.sub("trigger == 167")
# add MEG trial data to the dataset
ds['meg'] = load.fiff.epochs(ds, 0, tstop=60)

# DSS from
todss, fromdss = dss(ds['meg'])

# plot the DSS topography
plot.Topomap(fromdss[:, :6], '.dss', h=2, ncol=6)
# save the DSS for later use
save.pickle((todss, fromdss), 'DATA/R2290_HAYO_P3H_DSS.pickle')


# Exclude artifacts
# -----------------
# High amplitude artifacts can exert a bias on DSS components. A simple way to
# exclude the largest artifacts is using a simple threshold. First, confirm the
# presence of an artifact in the first trial:
p = plot.TopoButterfly('meg', '.case', ds=ds, xlim=(0, 10))
p.set_topo_t(11.62)
# find all time points where data does not exced 1000 fT
good_times = ds['meg'].abs().max(('case', 'sensor')) < 1e-12
# find time intervals with good data
intervals = find_intervals(good_times)
# add the intervals to the plot to confirm that we found the right intervals
p.add_vspans(intervals, color='green', alpha=0.2)
# concatenate all the good segments
meg_good = concatenate([ds['meg'].sub(time=interval) for interval in intervals])
# conmpute DSS on good data
todss, fromdss = dss(meg_good)
plot.Topomap(fromdss[:, :6], '.dss', h=2, ncol=6, title='Clean data only')


# DSS for multiple conditions
# ---------------------------
# To compute the DSS for multiple conditions, concatenate the data for
# different conditions:
ds = load.fiff.events('DATA/R2290_HAYO_P3H_1-8-raw.fif')
# load data for all 6 trials (each trial 10 s long)
ds['meg'] = load.fiff.epochs(ds, 0, tstop=10)
# extract NDVar for each condition; each condition has 3 trials:
c167 = ds[ds.eval("trigger == 167"), 'meg']
c163 = ds[ds.eval("trigger == 163"), 'meg']
# concatenate trials along the time axis:
data = concatenate((c163, c167))
# now data has 3 cases ("trials") of length 20 s.
# compute DSS using both conditions:
todss, fromdss = dss(data)
