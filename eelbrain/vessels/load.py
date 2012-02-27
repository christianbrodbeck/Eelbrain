'''
Functions for loading datasets from different types of files.


Created on Feb 21, 2012

@author: christian
'''

import os

import numpy as np
import mne

import data as _data
import colorspaces as _cs
import sensors


__all__ = ['fiff', 'fiff_events', 'add_fiff_to_dataset']



_default_fiff_properties = {'proj': 'ideal',
                            'ylim': 2e-12,
                            'summary_ylim': 3.5e-13,
                            'colorspace': _cs.get_MEG(2e-12),
                            'summary_colorspace': _cs.get_MEG(3.5e-13),
                            }


def fiff_events(path, labels={}):
    events = mne.read_events(path).reshape((-1,6))
    name = os.path.basename(path)
    assert all(events[:,1] == events[:,5])
    assert all(events[:,2] == events[:,4])
    istart = _data.var(events[:,0], name='i_start')
    istop = _data.var(events[:,3], name='i_stop')
    event = _data.var(events[:,2], name='eventID')
    dataset = _data.dataset(name, event, istart, istop)
    if labels:
        dataset.add(_data.factor(events[:,2], name='event', labels=labels))
    return dataset



def add_fiff_to_events(path, dataset, i_start='i_start', 
                       tstart=-.2, tstop=.6, properties=None, 
                       name="MEG", sensorsname='fiff-sensors'):#, i_stop='i_stop'):
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
    
    dataset = _data.dataset(name, Y, c_factor, default_DV=dataname)
    return dataset
