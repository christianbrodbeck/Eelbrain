'''
Functions for loading datasets from different types of files. The most recent 
workflow is based on loading events and data from a single file, and employs 
:func:`fiff_events` and :func:`fiff_epochs` as in the following example::

    >>> 


Created on Feb 21, 2012

@author: christian
'''

__all__ = ['unavailable']
unavailable = []

import os

import numpy as np

try:
    import mne
    __all__.extend(('fiff_events', 'fiff_epochs'))
except ImportError:
    unavailable.append('mne import failed')

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


def fiff_events(source_path=None, name=None, bin_invert=True, merge=-1):
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
    
    bin_invert : bool
        Use True if you want event trigger values to equal those sent through 
        psychtoolbox. This fixes the fact that mne's kit2fif and psychtoolbox
        interpret the binary coding in opposite ways.
          
    
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
    
    # kit2fif reads trigger channels in the reverse sequence from PTB, 
    # so we need to fix that in order to retrieve the original 
    # trigger values.
    if bin_invert:
        getbin8 = lambda x: bin(i)[2:].rjust(8, '0')
        revbin = lambda x: int(''.join(reversed(getbin8(x))), 2)
        events[:,2] = [revbin(i) for i in events[:,2]]
    
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


def fiff_epochs(dataset, i_start='i_start', 
                tstart=-.2, tstop=.6, baseline=(None,  0),
                properties=None, name="MEG", sensorsname='fiff-sensors'):
    """
    Uses the events in ``dataset[i_start]`` to extract epochs from the raw 
    file
    
    i_start : str
        name of the variable containing the index of the events to be
        imported
         
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
    
    picks = mne.fiff.pick_types(raw.info, meg=True, eeg=False, stim=False, 
                                eog=False, include=[], exclude=[])
    
    epochs = mne.Epochs(raw, events, 1, tstart, tstop, picks=picks, 
                        baseline=baseline)
    
    # determine data container properties
    epoch_shape = (len(picks), len(epochs.times))
    data_shape = (len(events), len(epochs.times), len(picks))
    # mne.fiff.raw.read_raw_segment uses float32
    data = np.empty(data_shape, dtype='float32') 

    # read the data
    for i, epoch in enumerate(epochs):
        if epoch.shape == epoch_shape:
            data[i] = epoch.T
        else:
            msg = ("Epoch %i shape mismatch: does your epoch definition "
                   "result in an epoch that overlaps the end of your data "
                   "file?" % i)
            raise IOError(msg)
    
    # read data properties
    props = {'samplingrate': epochs.info['sfreq'][0]}
    props.update(_default_fiff_properties)
    if properties:
        props.update(properties)
    
    T = epochs.times
    timevar = _data.var(T, 'time')
    dims = (timevar, sensor_net)
    
    dataset.add(_data.ndvar(dims, data, properties=props, name=name))
    dataset.default_DV = name

        



def fiff_event_file(path, labels={}):
    events = mne.read_events(path).reshape((-1,6))
    name = os.path.basename(path)
    assert all(events[:,1] == events[:,5])
    assert all(events[:,2] == events[:,4])
    istart = _data.var(events[:,0], name='i_start')
    istop = _data.var(events[:,3], name='i_stop')
    event = _data.var(events[:,2], name='eventID')
    dataset = _data.dataset(event, istart, istop, name=name)
    if labels:
        dataset.add(_data.factor(events[:,2], name='event', labels=labels))
    return dataset



def add_fiff_to_events(path, dataset, i_start='i_start', 
                       tstart=-.2, tstop=.6, properties=None, 
                       name="MEG", sensorsname='fiff-sensors'):
    """
    Adds MEG data form a new file to a dataset containing events. 
    
    """
    events = np.empty((dataset.N, 3), dtype=np.uint32)
    events[:,0] = dataset[i_start].x
    events[:,1] = 0
    events[:,2] = 1
    
    raw = mne.fiff.Raw(path)
    
    # parse sensor net
    sensor_list = []
    for ch in raw.info['chs']:
        ch_name = ch['ch_name']
        if ch_name.startswith('MEG'):
            x, y, z = ch['loc'][:3]
            sensor_list.append([x, y, z, ch_name])
    sensor_net = sensors.sensor_net(sensor_list, name=sensorsname)
    
    picks = mne.fiff.pick_types(raw.info, meg=True, eeg=False, stim=False, 
                                eog=False, include=[], exclude=[])
    
    # read the data
    epochs = mne.Epochs(raw, events, 1, tstart, tstop, picks=picks)
    data = np.array([e.T for e in epochs.get_data()])
    
    props = {'samplingrate': epochs.info['sfreq'][0]}
    props.update(_default_fiff_properties)
    if properties:
        props.update(properties)
    
    T = epochs.times
    timevar = _data.var(T, 'time')
    dims = (timevar, sensor_net)
    
    dataset.add(_data.ndvar(dims, data, properties=props, name=name))
    dataset.default_DV = name




def fiff(raw, events, conditions, varname='condition', dataname='MEG',
         tstart=-.2, tstop=.6, properties=None, name=None, c_colors={},
         sensorsname='fiff-sensors'):
    """
    Loads data directly when two files (raw and events) are provided 
    separately.
    
    conditions : dict
        ID->name dictionary of conditions that should be imported
    event : str
        path to the event file
    properties : dict
        set properties in addition to the defaults
    raw : str
        path to the raw file
    varname : str
        variable name that will contain the condition value 
    
    """
    if name is None:
        name = os.path.basename(raw)
    
    raw = mne.fiff.Raw(raw)
    
    # parse sensor net
    sensor_list = []
    for ch in raw.info['chs']:
        ch_name = ch['ch_name']
        if ch_name.startswith('MEG'):
            x, y, z = ch['loc'][:3]
            sensor_list.append([x, y, z, ch_name])
    sensor_net = sensors.sensor_net(sensor_list, name=sensorsname)
    
    events = mne.read_events(events)
    picks = mne.fiff.pick_types(raw.info, meg=True, eeg=False, stim=False, 
                                eog=False, include=[], exclude=[])
    
    data = []
    c_x = []
    
    # read the data
    for ID in conditions:
        epochs = mne.Epochs(raw, events, ID, tstart, tstop, picks=picks)
        samplingrate = epochs.info['sfreq'][0]
        
        # data
        c_data = epochs.get_data()        # n_ep, n_ch, n_t 
        
        for epoch in c_data:
            data.append(epoch.T)
#        data.append(c_data.T)

        T = epochs.times
        
        # conditions variable
        n_ep = len(c_data)
        c_x.extend([ID] * n_ep)
    
    # construct the dataset
    c_factor = _data.factor(c_x, name=varname, labels=conditions, 
                            colors=c_colors, retain_label_codes=True)
    
    props = {'samplingrate': samplingrate}
    props.update(_default_fiff_properties)
    if properties is not None:
        props.update(properties)
    
    data = np.array(data)
#    data = np.concatenate(data, axis=0)
    
    timevar = _data.var(T, 'time')
    dims = (timevar, sensor_net)
    
    Y = _data.ndvar(dims, data, properties=props, name=dataname)
    
    dataset = _data.dataset(Y, c_factor, name=name, default_DV=dataname)
    return dataset
