# skip test: data unavailable
"""Load data from *.sqd file, look for bad channels and filter data"""
import mne
from eelbrain import *

# load the raw data
raw = mne.io.read_raw_kit('Data/R2290_HAYO_P3H_08.26.2015.sqd',
                          stim_code='channel', stim=range(162, 176),
                          preload=True)

# filter the raw data
raw.filter(1, 8, n_jobs=1)

# load the events
ds = load.fiff.events(raw)
# look at events
print(ds)
# select the desired events
ds = ds.sub("trigger == 167")

# Add MEG data for trials
ds['meg'] = load.fiff.epochs(ds, 0, tstop=60)
# concatenate MEG data into one long "trial"
meg = concatenate(ds['meg'])
# plot average correlation with neighbors
plot.Topomap(neighbor_correlation(meg))

# add bad channels
raw.info['bads'] = ['MEG 056']

# check the result of removing the channel (need to add the data with the
# new bad channel setting)
ds['meg'] = load.fiff.epochs(ds, 0, tstop=60)
plot.Topomap(neighbor_correlation(concatenate(ds['meg'])))

# remove the mean from the data
ds['meg'] -= ds['meg'].mean('time')
# plot the first trial (ds[0]) as continuous data
plot.TopoButterfly(ds[0, 'meg'])

# save the filtered raw data
raw.save('Data/R2290_HAYO_P3H_1-8-raw.fif')
