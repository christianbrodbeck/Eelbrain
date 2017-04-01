# skip test
# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""
MTRF-Dataset
============
Analyze continuous speech data from the mTRF dataset (see [1] for a 
description of the data).


References
----------
.. [1] Crosse, M. J., Liberto, D., M, G., Bednar, A., & Lalor, E. C. (2016). 
       The Multivariate Temporal Response Function (mTRF) Toolbox: A MATLAB 
       Toolbox for Relating Neural Signals to Continuous Stimuli. Frontiers in 
       Human Neuroscience, 10. https://doi.org/10.3389/fnhum.2016.00604
"""
import os

from scipy.io import loadmat
import mne
from eelbrain import *

# load the mTRF speech dataset
root = mne.datasets.mtrf.data_path()
speech_path = os.path.join(root, 'speech_data.mat')
mdata = loadmat(speech_path)

# convert data to NDVars
# ----------------------
# create sampling time information
tstep = 1. / mdata['Fs'][0, 0]
n_times = mdata['envelope'].shape[0]
time = UTS(0, tstep, n_times)
# load the EEG sensor coordinates
sensor = Sensor.from_standard_layout('biosemi128')
# frequency dimension in the spectrogram
band = Scalar('band', range(16))
# create dataset
ds = Dataset()
ds['envelope'] = NDVar(mdata['envelope'][:, 0], (time,))
ds['eeg'] = NDVar(mdata['EEG'], (time, sensor))
ds['spectrogram'] = NDVar(mdata['spectrogram'], (time, band))

# Reverse correlation
# -------------------
# TRF using regression
res = regression(ds['eeg'], ds['envelope'], -0.150, 0.450, 10)
plot.TopoButterfly(res.h)
# STRF using regression
res = regression(ds['eeg'].sub(sensor='D12'), ds['spectrogram'], -0.150, 0.450, 1000)
plot.Array(res.h)
# using boosting
res = boosting(ds['eeg'], ds['envelope'], -0.150, 0.450)
plot.TopoButterfly(res.h)
