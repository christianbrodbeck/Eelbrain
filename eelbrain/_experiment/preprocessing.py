# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""
Pre-processing operations based on NDVars
"""
from os import mkdir
from os.path import dirname, exists, getmtime

import mne

from .. import load
from .._data_obj import NDVar


class RawPipe(object):

    def __init__(self, name, path, log):
        self.name = name
        self.path = path
        self.log = log

    def load(self, subject, session, add_bads=True, preload=False):
        path = self.path.format(subject=subject, session=session)
        raw = load.fiff.mne_raw(path, preload=preload)
        if add_bads:
            raw.info['bads'] = self.load_bad_channels(subject, session)
        else:
            raw.info['bads'] = []
        return raw

    def load_bad_channels(self, subject, session):
        raise NotImplementedError

    def make_bad_channels(self, subject, session, bad_chs, redo):
        raise NotImplementedError

    def mtime(self, subject, session, bad_chs=True):
        "modification time of anything influencing the output of load"
        raise NotImplementedError


class RawSource(RawPipe):
    "raw data source"

    def __init__(self, name, path, bads_path, log):
        RawPipe.__init__(self, name, path, log)
        self.bads_path = bads_path

    def load_bad_channels(self, subject, session):
        path = self.bads_path.format(subject=subject, session=session)
        if exists(path):
            with open(path) as fid:
                return [l for l in fid.read().splitlines() if l]
        # need to create one to know mtime after user deletes the file
        self.log.info("No bad channel definition for: %s/%s, creating empty "
                      "bad_channels file", subject, session)
        self.make_bad_channels(subject, session, (), False)
        return []

    def make_bad_channels(self, subject, session, bad_chs, redo):
        path = self.bads_path.format(subject=subject, session=session)
        if exists(path):
            old_bads = self.load_bad_channels(subject, session)
        else:
            old_bads = None
        # find new bad channels
        if isinstance(bad_chs, basestring):
            bad_chs = (bad_chs,)
        raw = self.load(subject, session, add_bads=False)
        sensor = load.fiff.sensor_dim(raw)
        new_bads = sensor._normalize_sensor_names(bad_chs)
        # update with old bad channels
        if old_bads is not None and not redo:
            new_bads = sorted(set(old_bads).union(new_bads))
        # print change
        if old_bads is None:
            print("-> %s" % new_bads)
        else:
            print("%s -> %s" % (old_bads, new_bads))
        # write new bad channels
        text = '\n'.join(new_bads)
        with open(path, 'w') as fid:
            fid.write(text)

    def mtime(self, subject, session, bad_chs=True):
        path = self.path.format(subject=subject, session=session)
        if exists(path):
            mtime = getmtime(path)
            if not bad_chs:
                return mtime
            path = self.bads_path.format(subject=subject, session=session)
            if exists(path):
                return max(mtime, getmtime(path))


class CachedRawPipe(RawPipe):

    _bad_chs_affect_cache = False

    def __init__(self, name, source, path, log):
        assert isinstance(source, RawPipe)
        path = path.format(raw=name, subject='{subject}', session='{session}')
        RawPipe.__init__(self, name, path, log)
        self.source = source

    def cache(self, subject, session):
        "Make sure the cache is up to date"
        path = self.path.format(subject=subject, session=session)
        if not exists(path) or getmtime(path) < self.mtime(subject, session, self._bad_chs_affect_cache):
            self.log.debug("make raw %s for %s/%s...", self.name, subject, session)

            raw = self._make(subject, session)
            dir_path = dirname(path)
            if not exists(dir_path):
                mkdir(dir_path)
            raw.save(path)

    def load(self, subject, session, add_bads=True, preload=False):
        self.cache(subject, session)
        return RawPipe.load(self, subject, session, add_bads, preload)

    def load_bad_channels(self, subject, session):
        return self.source.load_bad_channels(subject, session)

    def _make(self, subject, session):
        raise NotImplementedError

    def make_bad_channels(self, subject, session, bad_chs, redo):
        self.source.make_bad_channels(subject, session, bad_chs, redo)

    def mtime(self, subject, session, bad_chs=True):
        return self.source.mtime(subject, session, bad_chs)


class RawFilter(CachedRawPipe):

    def __init__(self, name, source, path, log, args, kwargs):
        CachedRawPipe.__init__(self, name, source, path, log)
        self.args = args
        self.kwargs = kwargs

    def filter_ndvar(self, ndvar):
        axis = ndvar.get_axis('time')
        sfreq = 1. / ndvar.time.tstep
        x = ndvar.x.swapaxes(axis, 0) if axis else ndvar.x
        x = mne.filter.filter_data(x, sfreq, *self.args, **self.kwargs)
        if axis:
            x = x.swapaxes(axis, 0)
        return NDVar(x, ndvar.dims, ndvar.info.copy(), ndvar.name)

    def _make(self, subject, session):
        raw = self.source.load(subject, session, preload=True)
        raw.filter(*self.args, **self.kwargs)
        return raw


class RawICA(CachedRawPipe):
    """ICA raw pipe

    Notes
    -----
    To avoid unwanted data loss, the ICA does not check raw source mtime.
    However, if bad channels change the ICA is automatically recomputed.
    """

    def __init__(self, name, source, path, ica_path, log, session, kwargs):
        CachedRawPipe.__init__(self, name, source, path, log)
        if isinstance(session, basestring):
            self.session = (session,)
        else:
            assert isinstance(session, tuple)
        self.ica_path = ica_path
        self.session = session
        self.kwargs = kwargs

    def load_bad_channels(self, subject, session=None):
        bad_chs = set()
        for session in self.session:
            bad_chs.update(self.source.load_bad_channels(subject, session))
        return sorted(bad_chs)

    def load_ica(self, subject):
        path = self.ica_path.format(subject=subject)
        if not exists(path):
            raise RuntimeError("ICA file does not exist for raw=%r, "
                               "subject=%r. Run e.make_ica_selection() to "
                               "create it." % (self.name, subject))
        return mne.preprocessing.read_ica(path)

    def make_ica(self, subject):
        path = self.ica_path.format(subject=subject)
        raw = self.source.load(subject, self.session[0], add_bads=False)
        bad_channels = self.load_bad_channels(subject)
        raw.info['bads'] = bad_channels
        if exists(path):
            ica = mne.preprocessing.read_ica(path)
            picks = mne.pick_types(raw.info, ref_meg=False)
            if ica.ch_names == [raw.ch_names[i] for i in picks]:
                return path
            self.log.info("%s/%s: ICA outdated due to change in bad channels",
                          self.name, subject)
        self.log.debug("%s/%s: computing ICA decomposition", self.name, subject)

        for session in self.session[1:]:
            raw_ = self.source.load(subject, session, False)
            raw_.info['bads'] = bad_channels
            raw.append(raw_)

        ica = mne.preprocessing.ICA(max_iter=256, **self.kwargs)
        # reject presets from meeg-preprocessing
        ica.fit(raw, reject={'mag': 5e-12, 'grad': 5000e-13, 'eeg': 300e-6})
        ica.save(path)
        return path

    def _make(self, subject, session):
        raw = self.source.load(subject, session, preload=True)
        ica = self.load_ica(subject)
        ica.apply(raw)
        return raw
