'''
Functions for loading datasets from mne's fiff files. 



Created on Feb 21, 2012

@author: christian
'''
from __future__ import division

import os
import fnmatch

import numpy as np

import mne
import mne.minimum_norm as _mn

import eelbrain.vessels.data as _data
import eelbrain.vessels.colorspaces as _cs
import eelbrain.vessels.sensors as sensors
from eelbrain import ui

__all__ = ['events', 'add_epochs', # basic pipeline
           'ds_2_evoked', 'evoked_2_stc', # get lists of mne objects
           'mne_events', 'mne_write_cov', 'mne_Raw', 'mne_Epochs' # get mne objects
           ]



def events(source_path=None, name=None, merge=-1, baseline=0):
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
        towards the earlier event, positive towards the later event).
    
    name : str
        A name for the dataset.
    
    baseline : int
        After kit2fiff conversion of sqd files with unused trigger channels, 
        the resulting fiff file's event channel can contain a baseline other 
        than 0. This interferes with normal event extraction. If the baseline
        value is provided as parameter, the events can still be extracted.
     
    """
    if source_path is None:
        source_path = ui.ask_file("Pick a Fiff File", "Pick a Fiff File",
                                  ext=[('fif', 'Fiff')])
        if not source_path:
            return
    elif not os.path.isfile(source_path):
        raise ValueError("Invalid source_path: %r" % source_path)
    
    if name is None:
        name = os.path.basename(source_path)
    
    raw = mne.fiff.Raw(source_path)
    if baseline:
        pick = mne.event.pick_channels(raw.info['ch_names'], include='STI 014')
        data, times = raw[pick, :]
        idx = np.where(np.abs(np.diff(data[0])) > 0)[0]
        
        # find baseline NULL-events
        values = data[0, idx + 1]
        valid_events = np.where(values != baseline)[0]
        idx = idx[valid_events]
        values = values[valid_events]
        
        N = len(values)
        events = np.empty((N, 3), dtype=np.int32)
        events[:,0] = idx
        events[:,1] = np.zeros_like(idx)
        events[:,2] = values
    else:
        events = mne.find_events(raw)
    
    if len(events) == 0:
        raise ValueError("No events found!")
    
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
            'samplingrate': raw.info['sfreq'][0],
            'info': raw.info}
    return _data.dataset(event, istart, name=name, info=info)



def add_epochs(dataset, i_start='i_start', target="MEG", add=True,
              tstart=-.2, tstop=.6, baseline=None, 
              downsample=1, mult=1, unit='T',
              properties=None, sensorsname='fiff-sensors'):
    """
    Adds data from individual epochs as a ndvar to a dataset.
    Uses the events in ``dataset[i_start]`` to extract epochs from the raw 
    file associated with ``dataset``; returns ndvar or nothing (see ``add`` 
    argument).
    
    add : bool
        Add the variable to the dataset. If ``True`` (default), the data is 
        added to the dataset and the function returns nothing; if ``False``,
        the function returns the ndvar object.
    baseline : tuple(start, stop) or ``None``
        Time interval in seconds for baseline correction; ``None`` omits 
        baseline correction (default).
    dataset : dataset
        Dataset containing a variable (i_start) which defines epoch cues
    downsample : int
        Downsample the data by this factor when importing. ``1`` means no 
        downsampling. Note that this function does not low-pass filter 
        the data. The data is downsampled by picking out every
        n-th sample (see `Wikipedia <http://en.wikipedia.org/wiki/Downsampling>`_).
    i_start : str
        name of the variable containing the index of the events to be
        imported
    mult : scalar
        multiply all data by a constant. If used, the ``unit`` kwarg should
        specify the target unit, not the source unit.
    tstart : scalar
        start of the epoch relative to the cue
    tstop : scalar
        end of the epoch relative to the cue
    unit : str
        Unit of the data (default is 'T').
    target : str
        name for the new ndvar containing the epoch data  
         
    """
    events = mne_events(ds=dataset, i_start=i_start)
    
    source_path = dataset.info['source']
    raw = mne.fiff.Raw(source_path)
    
    # parse sensor net
    ch_locs = []
    ch_names = []
    for ch in raw.info['chs']:
        ch_name = ch['ch_name']
        if ch_name.startswith('MEG'):
            x, y, z = ch['loc'][:3]
            ch_locs.append((x, y, z))
            ch_names.append(ch_name)
    sensor_net = sensors.sensor_net(ch_locs, ch_names, name=sensorsname)
    
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
            if mult != 1:
                epoch_data = epoch_data * mult
            data[i] = epoch_data
        else:
            msg = ("Epoch %i shape mismatch: does your epoch definition "
                   "result in an epoch that overlaps the end of your data "
                   "file?" % i)
            raise IOError(msg)
    
    # read data properties
    props = {'proj': 'z root',
             'unit': unit,
             'ylim': 2e-12 * mult,
             'summary_ylim': 3.5e-13 * mult,
             'colorspace': _cs.get_MEG(2e-12 * mult),
             'summary_colorspace': _cs.get_MEG(2e-13 * mult), # was 2.5
             }

    props['samplingrate'] = epochs.info['sfreq'][0] / downsample
    if properties:
        props.update(properties)
    
    ndvar = _data.ndvar(dims, data, properties=props, name=target)
    if add:
        dataset.add(ndvar)
    else:
        return ndvar



def ds_2_evoked(ds, X, tstart=-0.1, tstop=0.6, baseline=(None, 0), 
                target='evoked', i_start='i_start', eventID='eventID', count='n',
                reject=None, 
                ):
    """
    Takes as input a single-trial dataset ``ds``, and returns a dataset 
    compressed to the model ``X``, adding a list variable named ``target`` (by 
    default ``"evoked"``) containing an ``mne.Evoked`` object for each cell.
    
    """
    evoked = []
    for cell in X.values():
        ds_cell = ds.subset(X == cell)
        epochs = mne_Epochs(ds_cell, tstart=tstart, tstop=tstop, 
                            baseline=baseline, reject=reject)
        evoked.append(epochs.average())
    
    
    dsc = ds.compress(X, count=count)
    if isinstance(count, str):
        count = dsc[count]
    
    dsc[target] = evoked
    
    # update n cases per average
    for i,ev in enumerate(evoked):
        count[i] = ev.nave
    
    return dsc



def evoked_2_stc(ds, fwd, cov, evoked='evoked', target='stc', 
                 loose=0.2, depth=0.8,
                 lambda2 = 1.0 / 3**2, dSPM=True, pick_normal=False):
    """
    Takes a dataset with an evoked list and adds a corresponding stc list
    
    
    *mne inverse operator:*
    
    loose: float in [0, 1]
        Value that weights the source variances of the dipole components
        defining the tangent space of the cortical surfaces.
    depth: None | float in [0, 1]
        Depth weighting coefficients. If None, no depth weighting is performed.
    
    **mne apply inverse:**
    
    lambda2, dSPM, pick_normal
    """
    stcs = []
    for case in ds.itercases():
        fwd_file = fwd.format(**case)
        cov_file = cov.format(**case)
        
        fwd_obj = mne.read_forward_solution(fwd_file, force_fixed=False, surf_ori=True)
        cov_obj = mne.Covariance(cov_file)
        
        evoked = case['evoked']
        inv = _mn.make_inverse_operator(evoked.info, fwd_obj, cov_obj, loose=loose, depth=depth)
        
        stc = _mn.apply_inverse(evoked, inv, lambda2=lambda2, dSPM=dSPM, pick_normal=pick_normal)
        stc.src = inv['src'] # add the source space so I don't have to retrieve it independently 
        stcs.append(stc)
    
    if target:
        ds[target] = stcs
    else:
        return stcs


def fiff_mne(ds, fwd='{fif}*fwd.fif', cov='{fif}*cov.fif', label=None, name=None,
             tstart=-0.1, tstop=0.6, baseline=(None, 0)):
    """
    adds data from one label as
    
    """
    if name is None:
        if label:
            _, lbl = os.path.split(label)
            lbl, _ = os.path.splitext(lbl)
            name = lbl.replace('-', '_')
        else:
            name = 'stc'
    
    info = ds.info['info']
    
    fif_name = ds.info['source']
    fif_name, _ = os.path.splitext(fif_name)
    if fif_name.endswith('raw'):
        fif_name = fif_name[:-3]
    
    fwd = fwd.format(fif=fif_name)
    if '*' in fwd:
        d, n = os.path.split(fwd)
        names = fnmatch.filter(os.listdir(d), n)
        if len(names) == 1:
            fwd = os.path.join(d, names[0])
        else:
            raise IOError("No unique fwd file matching %r" % fwd)
    
    cov = cov.format(fif=fif_name)
    if '*' in cov:
        d, n = os.path.split(cov)
        names = fnmatch.filter(os.listdir(d), n)
        if len(names) == 1:
            cov = os.path.join(d, names[0])
        else:
            raise IOError("No unique cov file matching %r" % cov)
    
    fwd = mne.read_forward_solution(fwd, force_fixed=False, surf_ori=True)
    cov = mne.Covariance(cov)
    inv = _mn.make_inverse_operator(info, fwd, cov, loose=0.2, depth=0.8)
    epochs = mne_Epochs(ds, tstart=tstart, tstop=tstop, baseline=baseline)
    
    # mne example:
    snr = 3.0
    lambda2 = 1.0 / snr ** 2
    
    if label is not None:
        label = mne.read_label(label)
    stcs = _mn.apply_inverse_epochs(epochs, inv, lambda2, dSPM=False, label=label)
    
    x = np.vstack(s.data.mean(0) for s in stcs)
    s = stcs[0]
    dims = (_data.var(s.times, 'time'),)
    ds[name] = _data.ndvar(dims, x, properties=None, info='')
    
    return stcs
    
    
    
#    data = sum(stc.data for stc in stcs) / len(stcs)
#    
#    # compute sign flip to avoid signal cancelation when averaging signed values
#    flip = mne.label_sign_flip(label, inverse_operator['src'])
#    
#    label_mean = np.mean(data, axis=0)
#    label_mean_flip = np.mean(flip[:, np.newaxis] * data, axis=0)



def mne_events(ds=None, i_start='i_start'):
    if isinstance(i_start, basestring):
        i_start = ds[i_start]
    
    events = np.empty((ds.N, 3), dtype=np.int32)
    events[:,0] = i_start.x
    events[:,1] = 0
    events[:,2] = 1
    return events


def mne_write_cov(ds, tstart=-.1, tstop=0, baseline=(None, 0), dest='{source}-cov.fif'):
    events = mne_events(ds=ds)
    source_path = ds.info['source']
    raw = mne.fiff.Raw(source_path)
    epochs = mne.Epochs(raw, events, 1, tstart, tstop, baseline=baseline)
    cov = mne.compute_covariance(epochs, keep_sample_mean=True)
    
    source, _ = os.path.splitext(source_path)
    dest = dest.format(source=source)
    cov.save(dest)


def mne_Raw(ds):
    source_path = ds.info['source']
    raw = mne.fiff.Raw(source_path)
    return raw


def mne_Epochs(ds, tstart=-0.1, tstop=0.6, baseline=(None, 0), reject=None):
    """
    reject : 
        e.g., {'mag': 2e-12}
    """
    source_path = ds.info['source']
    raw = mne.fiff.Raw(source_path)
    
    events = mne_events(ds=ds)
    
    epochs = mne.Epochs(raw, events, 1, tmin=tstart, tmax=tstop, 
                        baseline=baseline, reject=reject)
    return epochs
