'''
Functions for loading data from mne's fiff files.

To load events as a dataset::

    >>> ds = load.fiff.events(path)

Events can then be modified in he ds (adding variables, discarding events,
...). These events can then be used in the following ways (for more
options, see the documentation of the relevant functions):


1) load epochs as ndvar
^^^^^^^^^^^^^^^^^^^^^^^

The epochs can be added as an ndvar object with::

    >>> ds = load.fiff.add_epochs(ds)

The returned ds contains the new ndvar as 'MEG'
If no epochs are rejected during loading, the returned ds is identical with the
input ds.
If epochs are rejected during loading, the returned ds is a shortened version
of the input dataset that only contains the good epochs.


2) load epochs as mne.Epochs object
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The epochs can be loaded as mne.Epochs object using::

    >>> epochs = load.fiff.mne_Epochs(ds)

Note that the returned epochs event does not contain meaningful event ids,
and ``epochs.event_id`` is None.




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
from eelbrain.vessels.dimensions import source_space
from eelbrain import ui

__all__ = ['Raw', 'events', 'add_epochs', 'add_mne_epochs',  # basic pipeline
           'mne_events', 'mne_Raw', 'mne_Epochs',  # get mne objects
           'sensor_net',
           'epochs_ndvar', 'evoked', 'evoked_ndvar', 'stc', 'stc_ndvar',
           'brainvision_events_to_fiff',
           ]



def Raw(path=None, proj=False, **kwargs):
    """
    Returns a mne.fiff.Raw object with added projections if appropriate.

    Arguments
    ---------

    path : None | str(path)
        path to the raw fiff file. If ``None``, a file can be chosen form a
        file dialog.

    proj : bool | str(path)
        Add projections from a separate file to the Raw object.
        **``False``**: No proj file will be added.
        **``True``**: ``'{raw}*proj.fif'`` will be used.
        ``'{raw}'`` will be replaced with the raw file's path minus '_raw.fif',
        and '*' will be expanded using fnmatch. If multiple files match the
        pattern, a ValueError will be raised.
        **``str``**: A custom path template can be provided, ``'{raw}'`` and
        ``'*'`` will be treated as with ``True``.

    **kwargs**
        Additional keyword arguments are forwarded to mne.fiff.Raw
        initialization.

    """
    if path is None:
        path = ui.ask_file("Pick a Raw Fiff File", "Pick a Raw Fiff File",
                           ext=[('fif', 'Fiff')])
        if not path:
            return

    if not os.path.isfile(path):
        raise IOError("%r is not a file" % path)

    if 'verbose' not in kwargs:
        kwargs['verbose'] = False
    raw = mne.fiff.Raw(path, **kwargs)

    if proj:
        if proj == True:
            proj = '{raw}*proj.fif'

        if '{raw}' in proj:
            raw_file = raw.info['filename']
            raw_root, _ = os.path.splitext(raw_file)
            raw_root = raw_root.rstrip('raw')
            proj = proj.format(raw=raw_root)

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
        raw.add_proj(proj, remove_existing=True)

    return raw


def events(raw=None, merge= -1, proj=False, name=None,
           stim_channel='STI 014',
           stim_channel_bl=0, verbose=False):
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
        ``'{raw}'`` will be expanded to the raw file's path minus extension.
        With ``proj=True``, ``'{raw}_*proj.fif'`` will be used,
        looking for any projection file starting with the raw file's name.
        If multiple files match the pattern, a ValueError will be raised.

    name : str | None
        A name for the dataset. If ``None``, the raw filename will be used.

    stim_channel : str
        Name of the trigger channel.

    stim_channel_bl : int
        For corrupted event channels:
        After kit2fiff conversion of sqd files with unused trigger channels,
        the resulting fiff file's event channel can contain a baseline other
        than 0. This interferes with normal event extraction. If the baseline
        value is provided as parameter, the events can still be extracted.

    """
    if raw is None or isinstance(raw, basestring):
        raw = Raw(raw, proj=proj, verbose=verbose)

    if name is None:
        raw_file = raw.info['filename']
        name = os.path.basename(raw_file)

    if stim_channel_bl:
        pick = mne.event.pick_channels(raw.info['ch_names'], include=stim_channel)
        data, _ = raw[pick, :]
        idx = np.where(np.abs(np.diff(data[0])) > 0)[0]

        # find baseline NULL-events
        values = data[0, idx + 1]
        valid_events = np.where(values != stim_channel_bl)[0]
        idx = idx[valid_events]
        values = values[valid_events]

        N = len(values)
        events = np.empty((N, 3), dtype=np.int32)
        events[:, 0] = idx
        events[:, 1] = np.zeros_like(idx)
        events[:, 2] = values
    else:
        events = mne.find_events(raw, verbose=verbose, stim_channel=stim_channel)

    if len(events) == 0:
        raise ValueError("No events found!")

    if any(events[:, 1] != 0):
        raise NotImplementedError("Events starting with ID other than 0")
        # this was the case in the raw-eve file, which contained all event
        # offsets, but not in the raw file created by kit2fiff. For handling
        # see :func:`fiff_event_file`

    if merge:
        index = np.ones(len(events), dtype=bool)
        diff = np.diff(events[:, 0])
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
                events[i1, 2] = events[i2, 2]

        events = events[index]

    istart = var(events[:, 0], name='i_start')
    event = var(events[:, 2], name='eventID')
    info = {'raw': raw,
            'samplingrate': raw.info['sfreq'],
            'info': raw.info}
    return dataset(event, istart, name=name, info=info)



def add_epochs(ds, tstart= -0.1, tstop=0.6, baseline=None,
               downsample=1, mult=1, unit='T', proj=True,
               data='mag', reject=None,
               raw=None, add=True,
               target="MEG", i_start='i_start',
               properties=None, sensors=None):
    """
    Adds data from individual epochs as a ndvar to the dataset ``ds`` and
    returns the dataset. Unless the ``reject`` argument is specified, ``ds``
    is modified in place. With ``reject``, a subset of ``ds`` is returned
    containing only those events for which data was loaded.

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
    proj : bool
        mne.Epochs kwarg (subtract projections when loading data)
    tstart : scalar
        start of the epoch relative to the cue
    tstop : scalar
        end of the epoch relative to the cue
    unit : str
        Unit of the data (default is 'T').
    target : str
        name for the new ndvar containing the epoch data
    reject : None | scalar
        Threshold for rejecting epochs (peak to peak). Requires a for of
        mne-python which implements the Epochs.model['index'] variable.
    raw : None | mne.fiff.Raw
        Raw file providing the data; if ``None``, ``ds.info['raw']`` is used.
    sensors : None | eelbrain.vessels.sensors.sensor_net
        The default (``None``) reads the sensor locations from the fiff file.
        If the fiff file contains incorrect sensor locations, a different
        sensor_net can be supplied through this kwarg.

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
                                eog=False, include=[], exclude='bads')

    if reject:
        reject = {data: reject}
    else:
        reject = None

    epochs = mne_Epochs(ds, tstart=tstart, tstop=tstop, baseline=baseline,
                        proj=proj, i_start=i_start, raw=raw, picks=picks,
                        reject=reject, preload=True, decim=downsample)

    # read the data
    x = epochs.get_data()
    if len(x) == 0:
        raise RuntimeError("No events left in %r" % raw.info['filename'])
    T = epochs.times
    if mult != 1:
        x *= mult

    # read data properties
    props = {'proj': 'z root',
             'unit': unit,
             'ylim': 2e-12 * mult,
             'summary_ylim': 3.5e-13 * mult,
             'colorspace': _cs.get_MEG(2e-12 * mult),
             'summary_colorspace': _cs.get_MEG(2e-13 * mult),  # was 2.5
             }

    props['samplingrate'] = epochs.info['sfreq']
    if properties:
        props.update(properties)

    # target container
    picks = mne.fiff.pick_types(epochs.info, meg=meg, eeg=eeg, stim=False,
                                eog=False, include=[])
    if sensors is None:
        sensor = sensor_net(epochs, picks=picks)
    else:
        sensor = sensors
    time = var(T, 'time')
    dims = ('case', sensor, time)

    epochs_var = ndvar(x, dims=dims, properties=props, name=target)
    if add:
        ds = trim_ds(ds, epochs)
        ds.add(epochs_var)
        return ds
    else:
        return epochs_var



def add_mne_epochs(ds, target='epochs', **kwargs):
    """
    Add an mne.Epochs object to the dataset and return the dataset.

    If, after loading, the Epochs contain fewer cases than the dataset, a copy
    of the dataset is made containing only those events also contained in the
    Epochs. Note that the Epochs are always loaded with ``preload==True``.


    Parameters
    ----------

    ds : dataset
        Dataset with events from a raw fiff file (i.e., created by
        load.fiff.events).
    target : str
        Name for the Epochs object in the dataset.
    kwargs :
        Any additional keyword arguments are forwarded to the mne Epochs
        object initialization.

    """
    kwargs['preload'] = True
    epochs = mne_Epochs(ds, **kwargs)
    ds = trim_ds(ds, epochs)
    ds[target] = epochs
    return ds



def brainvision_events_to_fiff(ds, raw=None, i_start='i_start', proj=False):
    """
    ..Warning:
        modifies the dataset ``ds`` in place

    """
    ds[i_start] -= 1
    if raw is None or isinstance(raw, basestring):
        raw = Raw(raw, proj=proj)

    ds.info['raw'] = raw


def fiff_mne(ds, fwd='{fif}*fwd.fif', cov='{fif}*cov.fif', label=None, name=None,
             tstart= -0.1, tstop=0.6, baseline=(None, 0)):
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


def mne2ndvar(mne_object, data='mag', vmax=2e-12, unit='T', name=None):
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

    exclude = mne_object.info.get('bads', [])
    picks = mne.fiff.pick_types(mne_object.info, meg=meg, eeg=eeg, stim=False,
                                eog=False, include=[], exclude=exclude)
    properties = {'colorspace': _cs.get_MEG(vmax=vmax, unit=unit)}

    if isinstance(mne_object, mne.fiff.Evoked):
        x = mne_object.data[picks]
        time = var(mne_object.times, name='time')
        sensor = sensor_net(mne_object, picks=picks)
        dims = (sensor, time)
        return ndvar(x, dims=dims, name=name, properties=properties)
    else:
        err = "converting %s is not implemented" % type(mne_object)
        raise NotImplementedError(err)


def mne_events(ds=None, i_start='i_start', eventID='eventID'):
    if isinstance(i_start, basestring):
        i_start = ds[i_start]

    N = len(i_start)

    if isinstance(eventID, basestring):
        eventID = ds[eventID]
    elif eventID is None:
        eventID = np.ones(N)

    events = np.empty((N, 3), dtype=np.int32)
    events[:, 0] = i_start.x
    events[:, 1] = 0
    events[:, 2] = eventID
    return events


def mne_Raw(ds):
    return ds.info['raw']


def mne_Epochs(ds, i_start='i_start', raw=None,
               drop_bad_chs=True, name='{name}', **kwargs):
    """
    All ``**kwargs`` are forwarded to the mne.Epochs instance creation. If the
    mne-python fork in use supports the ``epochs.model`` attribute,
    ``epochs.model`` is updated with ``ds``

    drop_bad_chs : bool
        Drop all channels in raw.info['bads'] form the Epochs. This argument is
        ignored if the picks argument is specified.
    raw : None | mne.fiff.Raw
        If None, ds.info['raw'] is used.
    name : str
        Name for the Epochs object. ``'{name}'`` is formatted with the dataset
        name ``ds.name``.

    """
    if raw is None:
        raw = ds.info['raw']

    if drop_bad_chs and ('picks' not in kwargs) and raw.info['bads']:
        kwargs['picks'] = mne.fiff.pick_channels(raw.info['ch_names'], [],
                                                 raw.info['bads'])

    kwargs['name'] = name = name.format(name=ds.name)
    if 'tstart' in kwargs:
        kwargs['tmin'] = kwargs.pop('tstart')
    elif not 'tmin' in kwargs:
        kwargs['tmin'] = -0.1
    if 'tstop' in kwargs:
        kwargs['tmax'] = kwargs.pop('tstop')
    elif not 'tmax' in kwargs:
        kwargs['tmax'] = 0.5

    events = mne_events(ds=ds, i_start=i_start)
    # epochs with (event_id == None) does not use columns 1 and 2 of events
    events[:, 2] = np.arange(len(events))

    epochs = mne.Epochs(raw, events, event_id=None, **kwargs)

    return epochs


def sensor_net(fiff, picks=None, name='fiff-sensors'):
    """
    returns a sensor_net object based on the info in a fiff file.

    """
    info = fiff.info
    if picks is None:
        chs = info['chs']
    else:
        chs = [info['chs'][i] for i in picks]

    ch_locs = []
    ch_names = []
    for ch in chs:
        x, y, z = ch['loc'][:3]
        ch_name = ch['ch_name']
        ch_locs.append((x, y, z))
        ch_names.append(ch_name)
    return _sensors.sensor_net(ch_locs, ch_names, name=name)


def epochs_ndvar(epochs, name='MEG', meg=True, eeg=False):
    picks = mne.fiff.pick_types(epochs.info, meg=meg, eeg=eeg, stim=False,
                                eog=False, include=[], exclude='bads')
    x = epochs.get_data()[:, picks]
    sensor = sensor_net(epochs, picks)
    time = var(epochs.times, name='time')
    return ndvar(x, ('case', sensor, time), name=name)


def evoked(fname):
    "Load an mne evoked file as ndvar"
    evoked = mne.fiff.Evoked(fname)
    return evoked_ndvar(evoked)


def evoked_ndvar(evoked, name='MEG'):
    "Convert an mne Evoked object of a list thereof to an ndvar"
    if isinstance(evoked, mne.fiff.Evoked):
        x = evoked.data
        dims = ()
    else:
        x = np.array([e.data for e in evoked])
        dims = ('case',)
        evoked = evoked[0]
    sensor = sensor_net(evoked)
    time = var(evoked.times, name='time')
    return ndvar(x, dims + (sensor, time), name=name)


def stc(fname, subject='fsaverage'):
    "Load an stc as ndvar from a file"
    stc = mne.read_source_estimate(fname)
    return stc_ndvar(stc, subject=subject)


def stc_ndvar(stc, subject='fsaverage', name=None, check=True):
    """
    create an ndvar object from an mne SourceEstimate object

    stc : SourceEstimate | list of SourceEstimates
        The source estimate object(s).
    subject : str
        MRI subject (used for loading MRI in PySurfer plotting)
    name : str | None
        Ndvar name.
    check : bool
        If multiple stcs are provided, check if all stcs have the same times
        and vertices.

    """
    if isinstance(stc, mne.SourceEstimate):
        case = False
        x = stc.data
    else:
        case = True
        stcs = stc
        stc = stcs[0]
        if check:
            vert_lh, vert_rh = stc.vertno
            times = stc.times
            for stc_ in stcs[1:]:
                assert np.all(times == stc_.times)
                lh, rh = stc_.vertno
                assert np.all(vert_lh == lh)
                assert np.all(vert_rh == rh)
        x = np.array([s.data for s in stcs])

    time = var(stc.times, name='time')
    ss = source_space(stc.vertno, subject=subject)
    if case:
        dims = ('case', ss, time)
    else:
        dims = (ss, time)

    return ndvar(x, dims, name=name)



def trim_ds(ds, epochs):
    """
    Trim a dataset to account for rejected epochs. If no epochs were rejected,
    the original ds is rturned.

    Parameters
    ----------

    ds : dataset
        Dataset that was used to construct epochs.
    epochs : Epochs
        Epochs loaded with mne_Epochs()

    """
    if len(epochs.events) < ds.n_cases:
        index = epochs.events[:, 2]
        ds = ds.subset(index)

    return ds
