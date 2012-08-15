'''
Functions for loading data from mne's fiff files. 



Created on Feb 21, 2012

@author: christian
'''
from __future__ import division

import os
import fnmatch

import numpy as np

import mne
import mne.minimum_norm as _mn

from eelbrain.vessels.data import var, ndvar, dataset
import eelbrain.vessels.colorspaces as _cs
import eelbrain.vessels.sensors as _sensors
from eelbrain import ui

__all__ = ['raw', 'events', 'add_epochs', # basic pipeline
           'ds_2_evoked', 'evoked_2_stc', # get lists of mne objects
           'mne2ndvar', 'mne_events', 'mne_Raw', 'mne_Epochs', # get mne objects
           'sensors',
           ]

# make this available here for consistency
from mne.fiff import Raw as raw



def _raw_name_root(raw_file):
    "returns the 'root' of a raw file path"
    raw_root, _ = os.path.splitext(raw_file)
    for prune in ['raw', '_']:
        if raw_root.endswith(prune):
            raw_root = raw_root[:-len(prune)]
    return raw_root



def events(raw=None, merge=-1, proj=False, name=None, baseline=0):
    """
    Returns a dataset containing events from a raw fiff file. Use
    :func:`fiff_epochs` to load MEG data corresponding to those events.
    
    raw : str(path) | None | mne.fiff.Raw
        The raw fiff file from which to extract events (if ``None``, a file 
        dialog will be displayed).
    
    merge : int
        use to merge events lying in neighboring samples. The integer value 
        indicates over how many samples events should be merged, and the sign
        indicates in which direction they should be merged (negative means 
        towards the earlier event, positive towards the later event).
    
    proj : bool | str
        Path to the projections file that will be loaded with the raw file.
        ``'{raw}'`` will be expanded to the raw file's path minus 
        ``'_raw.fif'``. With ``proj=True``, ``'{raw}_*proj.fif'`` will be used,
        looking for any proj file with the raw file's name. If multiple files 
        match the pattern, a ValueError will be raised.
    
    name : str | None
        A name for the dataset. If ``None``, the raw filename will be used.
    
    baseline : int
        After kit2fiff conversion of sqd files with unused trigger channels, 
        the resulting fiff file's event channel can contain a baseline other 
        than 0. This interferes with normal event extraction. If the baseline
        value is provided as parameter, the events can still be extracted.
     
    """
    if raw is None:
        raw = ui.ask_file("Pick a Fiff File", "Pick a Fiff File",
                          ext=[('fif', 'Fiff')])
        if not raw:
            return
    
    if isinstance(raw, basestring):
        if os.path.isfile(raw):
            raw = mne.fiff.Raw(raw)
        else:
            raise IOError("%r is not a file" % raw)
    
    raw_file = raw.info['filename']
    
    if proj:
        if proj == True:
            proj = '{raw}_*proj.fif'
        
        if '{raw}' in proj:
            proj = proj.format(raw=_raw_name_root(raw_file))
        
        if '*' in proj:
            head, tail = os.path.split(proj)
            names = fnmatch.filter(os.listdir(head), tail)
            if len(names) == 1:
                proj = os.path.join(head, names[0])
            else:
                if len(names) == 0:
                    err = "No file matching %r"
                else:
                    err = "Multiple files matching %r"
                raise ValueError(err % proj)
        
        # add the projections to the raw file
        proj = mne.read_proj(proj)
        raw.info['projs'] += proj[:]
    
    if name is None:
        name = os.path.basename(raw_file)
    
    if baseline:
        pick = mne.event.pick_channels(raw.info['ch_names'], include='STI 014')
        data, _ = raw[pick, :]
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
    
    istart = var(events[:,0], name='i_start')
    event = var(events[:,2], name='eventID')
    info = {'raw': raw,
            'samplingrate': raw.info['sfreq'][0],
            'info': raw.info}
    return dataset(event, istart, name=name, info=info)



def add_epochs(ds, tstart=-0.1, tstop=0.6, baseline=None,
               downsample=1, mult=1, unit='T', proj=True,
               data='mag', raw=None,
               add=True, target="MEG", i_start='i_start', 
               properties=None, sensorsname='fiff-sensors'):
    """
    Adds data from individual epochs as a ndvar to a dataset (``ds``).
    Uses the events in ``ds[i_start]`` to extract epochs from the raw 
    file associated with ``ds``; returns ndvar or nothing (see ``add`` 
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
    if data == 'eeg':
        meg = False 
        eeg = True
    elif data in ['grad', 'mag']:
        meg = data 
        eeg = False
    else:
        err = 'data=%r' % data
        raise NotImplementedError(err)
    
    if raw is None:
        raw = ds.info['raw']
    
    picks = mne.fiff.pick_types(raw.info, meg=meg, eeg=eeg, stim=False, 
                                eog=False, include=[], exclude=raw.info['bads'])
    
    epochs = mne_Epochs(ds, tstart=tstart, tstop=tstop, baseline=baseline,
                        proj=proj, i_start=i_start, raw=raw, picks=picks)
    
    # read the data
    x = epochs.get_data() # this call iterates through epochs as well
    T = epochs.times
    if downsample != 1:
        index = slice(None, None, downsample)
        x = x[:,:,index]
        T = T[index]
    if mult != 1:
        x = x * mult
        
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
    
    # target container
    picks = mne.fiff.pick_types(epochs.info, meg=meg, eeg=eeg, stim=False, 
                                eog=False, include=[])
    sensor = sensors(epochs, picks=picks, name=sensorsname)
    time = var(T, 'time')
    dims = ('case', sensor, time)
    
    epochs_var = ndvar(x, dims=dims, properties=props, name=target)
    if add:
        ds.add(epochs_var)
    else:
        return epochs_var



def ds_2_evoked(ds, X, tstart=-0.1, tstop=0.6, baseline=(None, 0), 
                reject=None, 
                target='evoked', i_start='i_start', eventID='eventID', count='n',
                ):
    """
    Takes as input a single-trial dataset ``ds``, and returns a dataset 
    compressed to the model ``X``, adding a list variable named ``target`` (by 
    default ``"evoked"``) containing an ``mne.Evoked`` object for each cell.
    
    """
    evoked = []
    for cell in X.cells:
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



def evoked_2_stc(ds, files={'fwd':None, 'cov':None}, loose=0.2, depth=0.8,
                 lambda2 = 1.0 / 9, dSPM=True, pick_normal=False,
                 evoked='evoked', target='stc'):
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
    fwd_file = files.get('fwd')
    cov_file = files.get('cov')
    fwd_obj = mne.read_forward_solution(fwd_file, force_fixed=False, surf_ori=True)
    cov_obj = mne.Covariance(cov_file)
    for case in ds.itercases():
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
    
    raw = ds.info['raw']
    fif_name = raw.info['filename']
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
    dims = ('case', var(s.times, 'time'),)
    ds[name] = ndvar(x, dims, properties=None, info='')
    
    return stcs
    
    
    
#    data = sum(stc.data for stc in stcs) / len(stcs)
#    
#    # compute sign flip to avoid signal cancelation when averaging signed values
#    flip = mne.label_sign_flip(label, inverse_operator['src'])
#    
#    label_mean = np.mean(data, axis=0)
#    label_mean_flip = np.mean(flip[:, np.newaxis] * data, axis=0)


def mne2ndvar(mne_object, data='mag', name=None, bads=False):
    """
    Converts an mne data object to an ndvar. 
    
    The main difference is that an ndvar
    can only contain one type of data ('mag', 'grad', 'eeg')
    
    """
    if data == 'eeg':
        meg = False 
        eeg = True
    elif data in ['grad', 'mag']:
        meg = data 
        eeg = False
    else:
        err = 'data=%r' % data
        raise NotImplementedError(err)
    
    if bads:
        exclude = mne_object.info['bads']
    else:
        exclude = []
    
    picks = mne.fiff.pick_types(mne_object.info, meg=meg, eeg=eeg, stim=False, 
                                eog=False, include=[], exclude=exclude)
    
    if isinstance(mne_object, mne.fiff.Evoked):
        x = mne_object.data[picks]
        time = var(mne_object.times, name='time')
        sensor = sensors(mne_object, picks=picks)
        dims = (sensor, time)
        return ndvar(x, dims=dims, name=name)
    else:
        err = "converting %s is not implemented" % type(mne_object)
        raise NotImplementedError(err)


def mne_events(ds=None, i_start='i_start', eventID='eventID'):
    if isinstance(i_start, basestring):
        i_start = ds[i_start]
    
    if isinstance(eventID, basestring):
        eventID = ds[eventID]
    elif eventID is None:
        eventID = np.ones(len(i_start))
    
    events = np.empty((ds.N, 3), dtype=np.int32)
    events[:,0] = i_start.x
    events[:,1] = 0
    events[:,2] = eventID
    return events


def mne_Raw(ds):
    return ds.info['raw']


def mne_Epochs(ds, tstart=-0.1, tstop=0.6, i_start='i_start', raw=None, **kwargs):
    """
    All ``**kwargs`` are forwarded to the mne.Epochs instance creation.
    
    reject : 
        e.g., {'mag': 2e-12}
    
    """
    if raw is None:
        raw = ds.info['raw']
    
    events = mne_events(ds=ds, i_start=i_start)
    
    epochs = mne.Epochs(raw, events, None, tmin=tstart, tmax=tstop, **kwargs)
    return epochs


def sensors(fiff, picks=None, name='fiff-sensors'):
    """
    returns a sensor_net object based on the info in a fiff file.
     
    """
    info = fiff.info
    if picks is None:
        if hasattr(fiff, 'picks'):
            picks = fiff.picks
        
        if picks is None:
            picks = mne.fiff.pick_types(info)

    chs = [info['chs'][i] for i in picks]
    ch_locs = []
    ch_names = []
    for ch in chs:
        x, y, z = ch['loc'][:3]
        ch_name = ch['ch_name']
        ch_locs.append((x, y, z))
        ch_names.append(ch_name)
    return _sensors.sensor_net(ch_locs, ch_names, name=name)
