'''
Functions for loading datasets from mne's fiff files. 



Created on Feb 21, 2012

@author: christian
'''
from __future__ import division

import os

import numpy as np

__all__ = []
unavailable = []
try:
    import mne
    __all__.extend(('fiff_events', 'fiff_epochs'))
except ImportError:
    unavailable.append('mne import failed')

if unavailable:
    __all__.append('unavailable')

import data as _data
import colorspaces as _cs
import sensors
from eelbrain import ui





_default_fiff_properties = {'proj': 'ideal',
                            'ylim': 2e-12,
                            'summary_ylim': 3.5e-13,
                            'colorspace': _cs.get_MEG(2e-12),
                            'summary_colorspace': _cs.get_MEG(3.5e-13),
                            }


def fiff_events(source_path=None, name=None, merge=-1):
    """
    Returns a dataset containing events from a raw fiff file. Use
    :func:`fiff_epochs` to load MEG data corresponding to those events.
    
    source_path : str (path)
        the location of the raw file (if ``None``, a file dialog will be 
        displayed).
    
    merge : int
        use to merge events lying in neighboring samples. The integer value 
        indicates over how many samples events should be merged, and the sign
        indicates in which direction they should be merged (negative means 
        towards the earlier event, positive towards the later event) 
    
    name : str
        A name for the dataset.

    """
    if source_path is None:
        source_path = ui.ask_file("Pick a Fiff File", "Pick a Fiff File",
                                  ext=[('fif', 'Fiff')])
    
    if name is None:
        name = os.path.basename(source_path)
    
    raw = mne.fiff.Raw(source_path)
    events = mne.find_events(raw)
    if any(events[:,1] != 0):
        raise NotImplementedError("Events starting with ID other than 0")
        # this was the case in the raw-eve file, which contained all event 
        # offsets, but not in the raw file created by kit2fiff. For handling
        # see :func:`fiff_event_file`
    
    if merge:
        index = np.ones(len(events), dtype=bool)
        diff = np.diff(events[:,0])
        where = np.where(diff <= abs(merge))[0]
        
        if merge > 0:
            # drop the earlier event
            index[where] = False
        else:
            # drop the later event
            index[where + 1] = False
            # move the trigger value to the earlier event
            for w in reversed(where):
                i1 = w
                i2 = w + 1
                events[i1,2] = events[i2,2]
        
        events = events[index]
    
    istart = _data.var(events[:,0], name='i_start')
    event = _data.var(events[:,2], name='eventID')
    info = {'source': source_path,
            'samplingrate': raw.info['sfreq'][0]}
    return _data.dataset(event, istart, name=name, info=info)


def fiff_epochs(dataset, i_start='i_start', target="MEG", add=True,
                tstart=-.2, tstop=.6, baseline=(None,  0), downsample=1,
                properties=None, sensorsname='fiff-sensors'):
    """
    Uses the events in ``dataset[i_start]`` to extract epochs from the raw 
    file associated with ``dataset``; returns ndvar or nothing (see ``add`` 
    argument).
    
    dataset : dataset
        Dataset containing a variable (i_start) which defines epoch cues
    add : bool
        Add the variable to the dataset. If ``True`` (default), the data is 
        added to the dataset and the function returns nothing; if ``False``,
        the function returns the ndvar object.
    downsample : int
        Downsample the data by this factor when importing. ``1`` means no 
        downsampling. Note that this function does not low-pass filter 
        the data. The data is downsampled by picking out every
        n-th sample (see `Wikipedia <http://en.wikipedia.org/wiki/Downsampling>`_).
    i_start : str
        name of the variable containing the index of the events to be
        imported
    target : str
        name for the new ndvar containing the epoch data  
         
    """
    events = np.empty((dataset.N, 3), dtype=np.int32)
    events[:,0] = dataset[i_start].x
    events[:,1] = 0
    events[:,2] = 1
    
    source_path = dataset.info['source']
    raw = mne.fiff.Raw(source_path)
    
    # parse sensor net
    sensor_list = []
    for ch in raw.info['chs']:
        ch_name = ch['ch_name']
        if ch_name.startswith('MEG'):
            x, y, z = ch['loc'][:3]
            sensor_list.append([x, y, z, ch_name])
    sensor_net = sensors.sensor_net(sensor_list, name=sensorsname)
    
    # source
    picks = mne.fiff.pick_types(raw.info, meg=True, eeg=False, stim=False, 
                                eog=False, include=[], exclude=[])
    epochs = mne.Epochs(raw, events, 1, tstart, tstop, picks=picks, 
                        baseline=baseline)
    
    # transformation
    index = slice(None, None, downsample)
    
    # target container
    T = epochs.times[index]
    time = _data.var(T, 'time')
    dims = (sensor_net, time)
    epoch_shape = (len(picks), len(time))
    data_shape = (len(events), len(picks), len(time))
    data = np.empty(data_shape, dtype='float32') 
    
    # read the data
#    data = epochs.get_data() # this call iterates through epochs as well
    for i, epoch in enumerate(epochs):
        epoch_data = epoch[:,index]
        if epoch_data.shape == epoch_shape:
            data[i] = epoch_data
        else:
            msg = ("Epoch %i shape mismatch: does your epoch definition "
                   "result in an epoch that overlaps the end of your data "
                   "file?" % i)
            raise IOError(msg)
    
    # read data properties
    props = _default_fiff_properties.copy()
    props['samplingrate'] = epochs.info['sfreq'][0] / downsample
    if properties:
        props.update(properties)
    
    ndvar = _data.ndvar(dims, data, properties=props, name=target)
    if add:
        dataset.add(ndvar)
        dataset.default_DV = target
    else:
        return ndvar
