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

from ..vessels.data import var, ndvar, dataset, Sensor, SourceSpace, UTS
from ..vessels import colorspaces as _cs
from .. import ui

__all__ = ['Raw', 'events', 'add_epochs', 'add_mne_epochs',  # basic pipeline
           'mne_events', 'mne_Raw', 'mne_Epochs',  # get mne objects
           'sensor_dim',
           'epochs_ndvar', 'evoked', 'evoked_ndvar', 'stc', 'stc_ndvar',
           'brainvision_events_to_fiff',
           ]


def Raw(path=None, proj=False, **kwargs):
    """
    Returns a mne.fiff.Raw object with added projections if appropriate.

    Parameters
    ----------
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
    kwargs
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


def events(raw=None, merge=-1, proj=False, name=None,
           bads=None, stim_channel=None):
    """
    Read events from a raw fiff file.

    Use :func:`fiff_epochs` to load MEG data corresponding to those events.

    Parameters
    ----------
    raw : str(path) | None | mne.fiff.Raw
        The raw fiff file from which to extract events (if ``None``, a file
        dialog will be displayed).
    merge : int
        Merge steps occurring in neighboring samples. The integer value
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
    bads : None | list
        Specify additional bad channels in the raw data file (these are added
        to the ones that are already defined in the raw file).
    stim_channel : None | string | list of string
        Name of the stim channel or all the stim channels
        affected by the trigger. If None, the config variables
        'MNE_STIM_CHANNEL', 'MNE_STIM_CHANNEL_1', 'MNE_STIM_CHANNEL_2',
        etc. are read. If these are not found, it will default to
        'STI 014'.

    Returns
    -------
    events : dataset
        A dataset with the following variables:
         - *i_start*: the index of the event in the raw file.
         - *eventID*: the event value.
        The dataset's info dictionary contains the following values:
         - *raw*: the mne Raw object.

    """
    if raw is None or isinstance(raw, basestring):
        raw = Raw(raw, proj=proj)

    if bads is not None:
        raw.info['bads'].extend(bads)

    if name is None:
        raw_file = raw.info['filename']
        name = os.path.basename(raw_file)

    # stim_channel_bl: see commit 52796ad1267b5ad4fba10f6ca5f2b7cfba65ba9b or earlier
    evts = mne.find_stim_steps(raw, merge=merge, stim_channel=stim_channel)
    idx = np.nonzero(evts[:, 2])
    evts = evts[idx]

    if len(evts) == 0:
        raise ValueError("No events found!")

    i_start = var(evts[:, 0], name='i_start')
    eventID = var(evts[:, 2], name='eventID')
    info = {'raw': raw}
    return dataset(eventID, i_start, name=name, info=info)


def add_epochs(ds, tstart=-0.1, tstop=0.6, baseline=None,
               decim=1, mult=1, unit='T', proj=True,
               data='mag', reject=None,
               raw=None, add=True,
               target="MEG", i_start='i_start',
               info=None, sensors=None, exclude='bads'):
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
    decim : int
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
    sensors : None | Sensor
        The default (``None``) reads the sensor locations from the fiff file.
        If the fiff file contains incorrect sensor locations, a different
        Sensor instance can be supplied through this kwarg.
    exclude : list of string | str
        Channels to exclude (:func:`mne.fiff.pick_types` kwarg).
        If 'bads' (default), exclude channels in info['bads'].
        If empty do not exclude any.

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
                                eog=False, include=[], exclude=exclude)

    if reject:
        reject = {data: reject}
    else:
        reject = None

    epochs = mne_Epochs(ds, tstart=tstart, tstop=tstop, baseline=baseline,
                        proj=proj, i_start=i_start, raw=raw, picks=picks,
                        reject=reject, preload=True, decim=decim)

    epochs_var = epochs_ndvar(epochs, name=target, meg=meg, eeg=eeg,
                              exclude=exclude, mult=mult, unit=unit,
                              info=info, sensors=sensors)

    if len(epochs_var) == 0:
        raise RuntimeError("No events left in %r" % raw.info['filename'])

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
    events[:, 1] = np.arange(len(events))

    epochs = mne.Epochs(raw, events, event_id=None, **kwargs)

    return epochs


def sensor_dim(fiff, picks=None, sysname='fiff-sensors'):
    """
    returns a Sensor object based on the info in a fiff object.

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
    return Sensor(ch_locs, ch_names, sysname=sysname)


def epochs_ndvar(epochs, name='MEG', meg=True, eeg=False, exclude='bads',
                 mult=1, unit='T', info=None, sensors=None, vmax=None):
    """
    Convert an mne.Epochs object to an ndvar.

    Parameters
    ----------
    epoch : mne.Epochs
        The epochs object
    name : None | str
        Name for the ndvar.
    meg : bool or string
        MEG channels to include (:func:`mne.fiff.pick_types` kwarg).
        If True include all MEG channels. If False include None
        If string it can be 'mag' or 'grad' to select only gradiometers
        or magnetometers. It can also be 'ref_meg' to get CTF
        reference channels.
    eeg : bool
        If True include EEG channels (:func:`mne.fiff.pick_types` kwarg).
    exclude : list of string | str
        Channels to exclude (:func:`mne.fiff.pick_types` kwarg).
        If 'bads' (default), exclude channels in info['bads'].
        If empty do not exclude any.
    mult : scalar
        multiply all data by a constant. If used, the ``unit`` kwarg should
        specify the target unit, not the source unit.
    unit : str
        Unit of the data (default is 'T').
    target : str
        name for the new ndvar containing the epoch data
    reject : None | scalar
        Threshold for rejecting epochs (peak to peak). Requires a for of
        mne-python which implements the Epochs.model['index'] variable.
    raw : None | mne.fiff.Raw
        Raw file providing the data; if ``None``, ``ds.info['raw']`` is used.
    sensors : None | Sensor
        The default (``None``) reads the sensor locations from the fiff file.
        If the fiff file contains incorrect sensor locations, a different
        Sensor can be supplied through this kwarg.
    vmax : None | scalar
        The default range for plotting (the default is 2e-12 T).

    """
    vmax = vmax or 2e-12 * mult
    info_ = _cs.meg_info(vmax, unit)
    info_.update(proj='z root', samplingrate=epochs.info['sfreq'],
                 summary_info=_cs.meg_info(0.1 * vmax))

    if info:
        info_.update(info)

    picks = mne.fiff.pick_types(epochs.info, meg=meg, eeg=eeg, stim=False,
                                eog=False, include=[], exclude=exclude)
    x = epochs.get_data()[:, picks]
    if mult != 1:
        x *= mult

    sensor = sensors or sensor_dim(epochs, picks=picks)
    time = UTS(epochs.tmin, 1. / epochs.info['sfreq'], len(epochs.times))
    return ndvar(x, ('case', sensor, time), info=info_, name=name)


def evoked_ndvar(evoked, name='MEG', meg=True, eeg=False, exclude='bads'):
    """
    Convert an mne Evoked object or a list thereof to an ndvar.

    Parameters
    ----------
    evoked : str | Evoked | list of Evoked
        The Evoked to convert to ndvar. Can be a string designating a file
        path to a evoked fiff file containing only one evoked.
    name : str
        Name of the ndvar.
    meg : bool | 'mag' | 'grad'
        What MEG data to keep.
    eeg : bool
        Whether to keep EEG data.
    exclude : see mne documentation
        Channels to exclude.

    Notes
    -----
    If evoked objects have different channels, the intersection is used (i.e.,
    only the channels present in all objects are retained).
    """
    if isinstance(evoked, basestring):
        evoked = mne.fiff.Evoked(evoked)

    if isinstance(evoked, mne.fiff.Evoked):
        picks = mne.fiff.pick_types(evoked.info, meg=meg, eeg=eeg,
                                    exclude=exclude)
        x = evoked.data[picks]
        sensor = sensor_dim(evoked, picks=picks)
        time = UTS.from_int(evoked.first, evoked.last, evoked.info['sfreq'])
        dims = (sensor, time)
    else:
        e0 = evoked[0]

        # find common channels
        all_chs = set(e0.info['ch_names'])
        exclude = set(e0.info['bads'])
        times = e0.times
        for e in evoked[1:]:
            chs = set(e.info['ch_names'])
            all_chs.update(chs)
            exclude.update(e.info['bads'])
            missing = all_chs.difference(chs)
            exclude.update(missing)
            if not np.all(e.times == times):
                raise ValueError("Not all evoked have the same time points.")

        # get data
        x = []
        sensor = None
        for e in evoked:
            picks = mne.fiff.pick_types(e.info, meg=meg, eeg=eeg,
                                        exclude=list(exclude))
            x.append(e.data[picks])
            if sensor is None:
                sensor = sensor_dim(e, picks=picks)

        time = UTS.from_int(e0.first, e0.last, e0.info['sfreq'])
        dims = ('case', sensor, time)

    info = _cs.meg_info(2e-13)
    return ndvar(x, dims, info=info, name=name)


def stc_ndvar(stc, subject='fsaverage', name=None, check=True):
    """
    create an ndvar object from an mne SourceEstimate object

    stc : SourceEstimate | list of SourceEstimates | str
        The source estimate object(s) or a path to an stc file.
    subject : str
        MRI subject (used for loading MRI in PySurfer plotting)
    name : str | None
        Ndvar name.
    check : bool
        If multiple stcs are provided, check if all stcs have the same times
        and vertices.

    """
    if isinstance(stc, basestring):
        stc = mne.read_source_estimate(stc)

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

    time = UTS(stc.tmin, stc.tstep, stc.shape[1])
    ss = SourceSpace(stc.vertno, subject=subject)
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
        index = epochs.events[:, 1]
        ds = ds.subset(index)

    return ds
