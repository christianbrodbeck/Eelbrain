# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
'''
MneExperiment is a base class for managing an mne experiment.


Epochs
------

Epochs are defined as dictionaries containing the following entries
(**mandatory**/optional):

**stimvar** : str
    The name of the variable which defines the relevant stimuli (i.e., the
    variable on which the stim value is chosen: the relevant events are
    found by ``idx = (ds[stimvar] == stim)``.
**stim** : str
    Value of the stimvar relative to which the epoch is defined. Can combine
    multiple names with '|'.
**name** : str
    A name for the epoch; when the resulting data is added to a Dataset, this
    name is used.
**tmin** : scalar
    Start of the epoch.
**tmax** : scalar
    End of the epoch.
reject_tmin : scalar
    Alternate start time for rejection (amplitude and eye-tracker).
reject_tmax : scalar
    Alternate end time for rejection (amplitude and eye-tracker).
decim : int
    Decimate the data by this factor (i.e., only keep every ``decim``'th
    sample)
tag : str
    Optional tag to identify epochs that differ in ways not captured by the
    above.


Epochs can be
specified in the :attr:`MneExperiment.epochs` dictionary. All keys in this
dictionary have to be of type :class:`str`, values have to be :class:`dict`s
containing the epoch specification. If an epoch is specified in
:attr:`MneExperiment.epochs`, its name (key) can be used in the epochs
argument to various methods. Example::

    # in MneExperiment subclass definition
    class experiment(MneExperiment):
        epochs = {'adjbl': dict(name='bl', stim='adj', tstart=-0.1, tstop=0)}
        ...

The :meth:`MneExperiment.get_epoch_str` method produces A label for each
epoch specification, which is used for filenames. Data which is excluded from
artifact rejection is parenthesized. For example, ``"noun[(-100)0,500]"``
designates data form -100 to 500 ms relative to the stimulus 'noun', with only
the interval form 0 to 500 ms used for rejection.

'''

from collections import defaultdict
from operator import add
import os
from Queue import Queue
import re
import shutil
import subprocess
from threading import Thread
from warnings import warn

import numpy as np

import mne
from mne.baseline import rescale
from mne.minimum_norm import (make_inverse_operator, apply_inverse,
                              apply_inverse_epochs)
from mne.coreg import scale_labels

from .. import fmtxt
from ..data import load
from ..data import plot
from ..data import save
from ..data import testnd
from ..data import Var, NDVar, combine
from ..data.data_obj import isdatalist, UTS, DimensionMismatchError
from .. import ui
from ..utils import keydefaultdict
from ..utils import subp
from ..utils.mne_utils import is_fake_mri, split_label
from .experiment import FileTree


__all__ = ['MneExperiment', 'LabelCache']

analysis_source = 'source_{raw}_{proj}_{rej}_{inv}'
analysis_sensor = 'sensor_{raw}_{proj}_{rej}'


class LabelCache(dict):
    def __getitem__(self, path):
        if path in self:
            return super(LabelCache, self).__getitem__(path)
        else:
            label = mne.read_label(path)
            self[path] = label
            return label


temp = {
        # basic dir
        'meg-sdir': os.path.join('{root}', 'meg'),  # contains subject-name folders for MEG data
        'meg-dir': os.path.join('{meg-sdir}', '{subject}'),
        'mri-sdir': os.path.join('{root}', 'mri'),  # contains subject-name folders for MRI data
        'mri-dir': os.path.join('{mri-sdir}', '{mrisubject}'),
        'bem-dir': os.path.join('{mri-dir}', 'bem'),
        'raw-dir': os.path.join('{meg-dir}', 'raw'),

        # raw
        'experiment': '???',
        # use iir with "l-h" labels, "hp..." labels are legacy
        'raw': ('clm', '0-40', '1-40', 'lp40', 'hp1-lp40'),
        # key necessary for identifying raw file info (used for bad channels):
        'raw-key': '{subject}',
        'raw-base': os.path.join('{raw-dir}', '{subject}_{experiment}_{raw}'),
        'raw-file': '{raw-base}-raw.fif',
        'raw-evt-file': '{raw-base}-evts.pickled',
        'trans-file': os.path.join('{raw-dir}', '{mrisubject}-trans.fif'),  # mne p. 196

        # log-files (eye-tracker etc.)
        'log-dir': os.path.join('{meg-dir}', 'logs'),
        'log-rnd': '{log-dir}/rand_seq.mat',
        'log-data-file': '{log-dir}/data.txt',
        'log-file': '{log-dir}/log.txt',
        'edf-file': os.path.join('{log-dir}', '*.edf'),

        # mne secondary/forward modeling
        'proj': '',
        'cov': 'bl',
        'proj-file': '{raw-base}_{proj}-proj.fif',
        'proj-plot': '{raw-base}_{proj}-proj.pdf',
        'cov-file': '{raw-base}_{cov}-{cov-rej}-{proj}-cov.fif',
        'bem-file': os.path.join('{bem-dir}', '{mrisubject}-*-bem.fif'),
        'bem-sol-file': os.path.join('{bem-dir}', '{mrisubject}-*-bem-sol.fif'),
        'src-file': os.path.join('{bem-dir}', '{mrisubject}-{src}-src.fif'),
        'fwd-file': '{raw-base}_{mrisubject}-fwd.fif',

        # epochs
        'epoch-stim': None,  # the stimulus/i selected by the epoch
        'epoch-desc': None,  # epoch description
        'epoch-bare': None,  # epoch description without decim or rej
        'epoch-nodecim': None,  # epoch description without decim parameter
        'epoch-rej': None,  # epoch description for rejection purposes
        'rej-file': os.path.join('{meg-dir}', 'epoch_sel', '{raw}_'
                                 '{experiment}_{epoch-rej}_sel.pickled'),

        'common_brain': 'fsaverage',

        # evoked
        'evoked-dir': os.path.join('{meg-dir}', 'evoked_{raw}_{proj}'),
        'evoked-file': os.path.join('{evoked-dir}', '{experiment}_'
                                    '{epoch-desc}_{model}_evoked.pickled'),

        # Source space
        'annot': 'aparc',
        'label': '???',
        'label-dir': os.path.join('{mri-dir}', 'label', '{annot}'),
        'hemi': ('lh', 'rh'),
        'label-file': os.path.join('{label-dir}', '{hemi}.{label}.label'),

        # (method) plots
        'plot-dir': os.path.join('{root}', 'plots'),
        'plot-file': os.path.join('{plot-dir}', '{analysis}', '{name}.{ext}'),

        # result output files
        # group/subject
        'kind': '',  # analysis kind (movie, plot, ...)
        'analysis': '',  # analysis name (source, ...)
        'name': '',  # file name
        'ext': 'pickled',  # file extension

        'res-gdir': os.path.join('{root}', 'res_{kind}_{experiment}',
                                 '{analysis}'),
        'res-dir': os.path.join('{res-gdir}', '{group}'),
        'res-file': os.path.join('{res-dir}', '{name}.{ext}'),

        # besa
        'besa-root': os.path.join('{root}', 'besa'),
        'besa-trig': os.path.join('{besa-root}', '{subject}', '{subject}_'
                                  '{experiment}_{epoch-bare}_triggers.txt'),
        'besa-evt': os.path.join('{besa-root}', '{subject}', '{subject}_'
                                 '{experiment}_{epoch-nodecim}.evt'),
         }


class MneExperiment(FileTree):
    """Class for managing data for an experiment

    """
    # Experiment Constants
    # ====================

    # Bad channels dictionary: (sub, exp) -> list of str
    bad_channels = defaultdict(list)

    # Default values for epoch definitions
    epoch_default = {'stimvar': 'stim', 'tmin':-0.1, 'tmax': 0.6, 'decim': 5}

    # named epochs
    epochs = {'bl': {'stim': 'adj', 'tmin':-0.2, 'tmax':0},
              'epoch': {'stim': 'adj'}}
    # Rejection
    # =========
    # how to reject data epochs.
    # kind : 'auto', 'manual', 'make'
    #     How the rejection is derived; 'auto': use the parameters to do the
    #     selection on the fly; 'manual': manually create a rejection file (use
    #     the selection GUI .make_rej()); 'make' a rejection file
    #     is created by the user
    # cov-rej : str
    #     rej setting to use for cov under this setting.
    #
    # For manual rejection
    # ^^^^^^^^^^^^^^^^^^^^
    # eog_sns : list of str
    #     The sensors to plot separately in the rejection GUI.
    #     The default is the two MEG sensors closest to the eyes
    #     for Abu Dhabi KIT data. For NY KIT data those are
    #     ['MEG 143', 'MEG 151'].
    # decim : int
    #     Decim factor for the rejection GUI (default is 5).
    #
    # For automatic rejection
    # ^^^^^^^^^^^^^^^^^^^^^^^
    # threshod : None | dict
    #     the reject argument when loading epochs:
    # edf : list of str
    #     How to use eye tracker information in rejection. True
    #     causes edf files to be loaded but not used
    #     automatically.
    _epoch_rejection = {'': {'kind': None},
                        'man': {'kind': 'manual',
                                'eog_sns': ['MEG 087', 'MEG 130'],
                                'decim': 5,
                                },
                        'et': {'kind': 'auto',
                               'threshold': dict(mag=3e-12),
                               'edf': ['EBLINK'],
                               },
                        'threshold': {'kind': 'auto',
                                      'threshold': dict(mag=3e-12)}
                        }
    epoch_rejection = {}

    exclude = {}  # field_values to exclude (e.g. subjects)

    groups = {}

    # whether to look for and load eye tracker data when loading raw files
    has_edf = defaultdict(lambda: False)

    # projection definition:
    # "base": 'raw' for raw file, or epoch name
    # "rej": rejection setting to use (only applies for epoch projs)
    # r.g. {'ironcross': {'base': 'adj', 'rej': 'man'}}
    projs = {}

    # Pattern for subject names
    subject_re = re.compile('R\d{4}$')

    # state variables that are always shown in self.__repr__():
    _repr_kwargs = ('subject', 'rej')

    # Where to search for subjects (defined as a template name). If the
    # experiment searches for subjects automatically, it scans this directory
    # for subfolders matching subject_re.
    _subject_loc = 'meg-sdir'

    # basic templates to use. Can be a string referring to a templates
    # dictionary in the module level _temp dictionary, or a templates
    # dictionary
    _templates = temp
    # specify additional templates
    _values = {}
    # specify defaults for specific fields (e.g. specify the initial subject
    # name)
    _defaults = {
                 'experiment': 'experiment_name',
                 # this should be a key in the epochs class attribute (see
                 # above)
                 'epoch': 'epoch'}

    def __init__(self, root=None, **state):
        """
        Parameters
        ----------
        root : str | None
            the root directory for the experiment (usually the directory
            containing the 'meg' and 'mri' directories)
        """
        # create attributes
        self.groups = self.groups.copy()
        self.exclude = self.exclude.copy()
        self._mri_subjects = keydefaultdict(lambda k: k)
        self._label_cache = LabelCache()
        self._templates = self._templates.copy()
        self._templates.update(self._values)

        # store epoch rejection settings
        epoch_rejection = self._epoch_rejection.copy()
        epoch_rejection.update(self.epoch_rejection)
        self.epoch_rejection = epoch_rejection

        FileTree.__init__(self, **state)
        self.set_root(root, True)

        # regiser variables with complex behavior
        self._register_field('rej', self.epoch_rejection.keys(),
                             post_set_handler=self._post_set_rej)
        self._register_field('group', self.groups.keys() + ['all'], 'all',
                             eval_handler=self._eval_group)
        self._register_field('epoch', self.epochs.keys(),
                             set_handler=self.set_epoch)
        self._register_value('inv', 'free-2-dSPM',
                             set_handler=self._set_inv_as_str)
        self._register_value('model', '', eval_handler=self._eval_model)
        self._register_field('src', ('ico-4', 'vol-10', 'vol-7'),
                             post_set_handler=self._post_set_src)

        # Define make handlers
        self._bind_make('evoked-file', self.make_evoked)
        self._bind_make('raw-file', self.make_raw)
        self._bind_make('cov-file', self.make_cov)
        self._bind_make('src-file', self.make_src)
        self._bind_make('fwd-file', self.make_fwd)

        # set initial values
        self._increase_depth()
        self.set_env()

    def __iter__(self):
        "Iterate state through subjects and yield each subject name."
        for subject in self.iter():
            yield subject

    @property
    def _epoch_state(self):
        epochs = self._params['epochs']
        if len(epochs) != 1:
            err = ("This function is only implemented for single epochs (got "
                   "%s)" % self.get('epoch'))
            raise NotImplementedError(err)
        return epochs[0]

    def add_epochs_stc(self, ds, src='epochs', dst='stc', ndvar=True):
        """
        Transform epochs contained in ds into source space (adds a list of mne
        SourceEstimates to ds)

        Parameters
        ----------
        ds : Dataset
            The Dataset containing the mne Epochs for the desired trials.
        src : str
            Name of the source epochs in ds.
        dst : str
            Name of the source estimates to be created in ds.
        ndvar : bool
            Add the source estimates as NDVar instead of a list of
            SourceEstimate objects.
        """
        subject = ds['subject']
        if len(subject.cells) != 1:
            err = ("ds must have a subject variable with exaclty one subject")
            raise ValueError(err)
        subject = subject.cells[0]
        self.set(subject=subject)

        epochs = ds[src]
        inv = self.load_inv(epochs)
        stc = apply_inverse_epochs(epochs, inv, **self._params['apply_inv_kw'])

        if ndvar:
            subject = self.get('mrisubject')
            kind, grade = self._params['src']
            stc = load.fiff.stc_ndvar(stc, subject, kind, grade, 'stc')

        ds[dst] = stc

    def add_evoked_label(self, ds, label, hemi='lh', src='stc'):
        """
        Extract the label time course from a list of SourceEstimates.

        Parameters
        ----------
        label :
            the label's bare name (e.g., 'insula').
        hemi : 'lh' | 'rh' | 'bh' | False
            False assumes that hemi is a Factor in ds.
        src : str
            Name of the variable in ds containing the SourceEstimates.

        Returns
        -------
        ``None``
        """
        if hemi in ['lh', 'rh']:
            self.set(hemi=hemi)
            key = label + '_' + hemi
        else:
            key = label
            if hemi != 'bh':
                assert 'hemi' in ds

        self.set(label=label)

        x = []
        for case in ds.itercases():
            if hemi == 'bh':
                lbl_l = self.load_label(subject=case['subject'], hemi='lh')
                lbl_r = self.load_label(hemi='rh')
                lbl = lbl_l + lbl_r
            else:
                if hemi is False:
                    self.set(hemi=case['hemi'])
                lbl = self.load_label(subject=case['subject'])

            stc = case[src]
            x.append(stc.in_label(lbl).data.mean(0))

        time = UTS(stc.tmin, stc.tstep, stc.shape[1])
        ds[key] = NDVar(np.array(x), dims=('case', time))

    def add_evoked_stc(self, ds, ind_stc=True, ind_ndvar=False, morph_stc=False,
                       morph_ndvar=False, baseline=None):
        """
        Add source estimates to a dataset with evoked data.

        Parameters
        ----------
        ds : Dataset
            The Dataset containing the Evoked objects.
        ind_stc : bool
            Add source estimates on individual brains as list of
            :class:`mne.SourceEstimate`.
        ind_ndvar : bool
            Add source estimates on individual brains as :class:`NDVar`.
        morph_stc : bool
            Add source estimates morphed to the common brain as list of
            :class:`mne.SourceEstimate`.
        morph_ndvar : bool
            Add source estimates morphed to the common brain as :class:`NDVar`.
        baseline : None | str | tuple
            Baseline correction in source space.

        Notes
        -----
        Assumes that all Evoked of the same subject share the same inverse
        operator.
        """
        n_subjects = ds.eval('len(subject.cells)')
        ind = (ind_stc or ind_ndvar)
        morph = (morph_stc or morph_ndvar)
        if not (ind or morph):
            return

        if isinstance(baseline, str):
            raise NotImplementedError("Baseline form different epoch")

        # find from subjects
        common_brain = self.get('common_brain')
        from_subjects = {}
        for subject in ds.eval('subject.cells'):
            if is_fake_mri(self.get('mri-dir', subject=subject)):
                subject_from = common_brain
            elif ind_ndvar and n_subjects > 1:
                err = ("Subject %r has its own MRI; An ndvar can only be "
                       "created form indivdual stcs if all stcs were "
                       "estimated on the common brain." % subject)
                raise ValueError(err)
            else:
                subject_from = subject
            from_subjects[subject] = subject_from

        # find vars to work on
        do = []
        for name in ds:
            if isinstance(ds[name][0], mne.fiff.Evoked):
                do.append(name)

        # prepare data containers
        invs = {}
        if ind:
            stcs = defaultdict(list)
        if morph:
            mstcs = defaultdict(list)

        # convert evoked objects
        mri_sdir = self.get('mri-sdir')
        for case in ds.itercases():
            subject = case['subject']
            subject_from = from_subjects[subject]

            # create stcs from sns data
            for name in do:
                evoked = case[name]

                # get inv
                if subject in invs:
                    inv = invs[subject]
                else:
                    inv = self.load_inv(evoked, subject=subject)
                    invs[subject] = inv

                # apply inv
                stc = apply_inverse(evoked, inv, **self._params['apply_inv_kw'])

                # baseline correction
                if baseline:
                    rescale(stc._data, stc.times, baseline, 'mean', copy=False)

                if ind:
                    stcs[name].append(stc)

                if morph:
                    stc = mne.morph_data(subject_from, common_brain, stc, 4,
                                         subjects_dir=mri_sdir)
                    mstcs[name].append(stc)

        # add to Dataset
        if len(do) > 1:
            keys = ('%%s_%s' % d for d in do)
        else:
            keys = ('%s',)
        kind, grade = self._params['src']
        for name, key in zip(do, keys):
            if ind_stc:
                ds[key % 'stc'] = stcs[name]
            if ind_ndvar:
                if n_subjects == 1:
                    subject = ds['subject'].cells[0]
                else:
                    subject = common_brain
                ndvar = load.fiff.stc_ndvar(stcs[name], subject, kind, grade)
                ds[key % 'src'] = ndvar
            if morph_stc:
                ds[key % 'stcm'] = mstcs[name]
            if morph_ndvar:
                ndvar = load.fiff.stc_ndvar(mstcs[name], common_brain, kind,
                                            grade)
                ds[key % 'srcm'] = ndvar

    def cache_events(self, redo=False):
        """Create the 'raw-evt-file'.

        This is done automatically the first time the events are loaded, but
        caching them will allow faster loading times form the beginning.
        """
        evt_file = self.get('raw-evt-file')
        exists = os.path.exists(evt_file)
        if exists and redo:
            os.remove(evt_file)
        elif exists:
            return

        self.load_events(add_proj=False)

    def get_epoch_str(self, stimvar=None, stim=None, tmin=None, tmax=None,
                      reject_tmin=None, reject_tmax=None, decim=None,
                      name=None, tag=None, rej_epoch=None):
        """Produces a descriptor for a single epoch specification

        Parameters
        ----------
        stimvar : str
            The name of the variable on which stim is defined (not included in
            the label).
        stim : str
            The stimulus name.
        tmin : scalar
            Start of the epoch data in seconds.
        tmax : scalar
            End of the epoch data in seconds.
        reject_tmin : None | scalar
            Set an alternate tmin for epoch rejection.
        reject_tmax : None | scalar
            Set an alternate tmax for epoch rejection.
        decim : None | int
            Decimate epoch data.
        name : None | str
            Name the epoch (not included in the label).
        tag : None | str
            Optional tag for epoch string.
        rej_epoch : None | str
            Use rejection from another epoch (only for rejection by rej-file;
            needs to have same triggers).
        """
        desc = '%s[' % stim
        if reject_tmin is None:
            desc += '%i_' % (tmin * 1000)
        else:
            desc += '(%i)%i_' % (tmin * 1000, reject_tmin * 1000)

        if reject_tmax is None:
            desc += '%i' % (tmax * 1000)
        else:
            desc += '(%i)%i' % (reject_tmax * 1000, tmax * 1000)

        if (decim is not None) and (decim != 1):
            desc += '|%i' % decim

        desc += '{rej}'
        if rej_epoch is not None:
            desc += '-%s' % rej_epoch

        if tag is not None:
            desc += '|%s' % tag

        return desc + ']'

    def get_field_values(self, field, exclude=True):
        """Find values for a field taking into account exclusion

        Parameters
        ----------
        field : str
            Field for which to find values.
        exclude : bool
            Exclude values based on experiment.exclude.
        """
        if field == 'mrisubject':
            subjects = self.get_field_values('subject', exclude=exclude)
            mrisubjects = sorted(self._mri_subjects[s] for s in subjects)
            common_brain = self.get('common_brain')
            if common_brain:
                mrisubjects.insert(0, common_brain)
            return mrisubjects
        elif field == 'group':
            values = ['all']
            values.extend(self.groups.keys())
            return values
        else:
            values = list(FileTree.get_field_values(self, field))
            if exclude:
                exclude = self.exclude.get(field, None)
            if exclude:
                values = [v for v in values if not v in exclude]

        return values

    def iter(self, fields='subject', group=None, **kwargs):
        """
        Cycle the experiment's state through all values on the given fields

        Parameters
        ----------
        fields : list | str
            Field(s) over which should be iterated.
        exclude : dict  {str: str, str: iterator over str, ...}
            Values to exclude from the iteration with {name: value} and/or
            {name: (sequence of values, )} entries.
        values : dict  {str: iterator over str}
            Fields with custom values to iterate over (instead of the
            corresponding field values) with {name: (sequence of values)}
            entries.
        group : None | str
            If iterating over subjects, use this group ('all' or a name defined
            in experiment.groups).
        prog : bool | str
            Show a progress dialog; str for dialog title.
        mail : bool | str
            Send an email when iteration is finished. Can be True or an email
            address. If True, the notification is sent to :attr:`.owner`.
        others :
            Fields with constant values throughout the iteration.
        """
        if group and (group != 'all') and ('subject' in fields):
            subjects = self.get_field_values('subject')
            group = self.groups[group]
            group_subjects = [s for s in subjects if s in group]
            kwargs.setdefault('values', {})['subject'] = group_subjects

        return FileTree.iter(self, fields, **kwargs)

    def iter_range(self, start=None, stop=None, field='subject'):
        """Iterate through a range on a field with ordered values.

        Parameters
        ----------
        start : None | str
            Start value (inclusive). With ``None``, begin at the first value.
        stop : None | str
            Stop value (inclusive). With ``None``, end with the last value.
        field : str
            Name of the field.

        Yields
        ------
        value : str
            Current field value.
        """
        values = self.get_field_values(field)
        if start is not None:
            start = values.index(start)
        if stop is not None:
            stop = values.index(stop) + 1
        idx = slice(start, stop)
        values = values[idx]

        level = self._increase_depth()
        for value in values:
            self.reset()
            self.set(**{field: value})
            yield value
        self.reset(level - 1)

    def iter_vars(self, *args, **kwargs):
        """Deprecated. Use :attr:`.iter()`"""
        warn("MneExperiment.iter_vars() is deprecated. Use .iter()",
             DeprecationWarning)
        kwargs['mail'] = kwargs.get('notify', False)
        self.iter(*args, **kwargs)

    def label_events(self, ds, experiment, subject):
        """
        Adds T (time) and SOA (stimulus onset asynchrony) to the Dataset.

        Parameters
        ----------
        ds : Dataset
            A Dataset containing events (as returned by
            :func:`load.fiff.events`).
        experiment : str
            Name of the experiment.
        subject : str
            Name of the subject.

        Notes
        -----
        Subclass this method to specify events.
        """
        if 'raw' in ds.info:
            raw = ds.info['raw']
            sfreq = raw.info['sfreq']
            ds['T'] = ds['i_start'] / sfreq
            ds['SOA'] = Var(np.ediff1d(ds['T'].x, 0))
        return ds

    def label_subjects(self, ds):
        """Label the subjects in ds based on .groups

        Parameters
        ----------
        ds : Dataset
            A Dataset with 'subject' entry.
        """
        subject = ds['subject']
        for name, subjects in self.groups.iteritems():
            ds[name] = Var(subject.isin(subjects))

    def load_edf(self, **kwargs):
        """Load the edf file ("edf-file" template)"""
        kwargs['fmatch'] = False
        src = self.get('edf-file', **kwargs)
        edf = load.eyelink.Edf(src)
        return edf

    def load_epochs(self, subject=None, baseline=None, ndvar=False,
                    add_bads=True, reject=True, add_proj=True, cat=None,
                    decim=None, **kwargs):
        """
        Load a Dataset with epochs for a given epoch definition

        Parameters
        ----------
        subject : str
            Subject(s) for which to load evoked files. Can be a group name
            such as 'all' or a single subject.
        epoch : str
            Epoch definition.
        ndvar : bool | str
            Convert epochs to an NDVar with the given name (if True, 'meg' is
            uesed).
        add_bads : False | True | list
            Add bad channel information to the Raw. If True, bad channel
            information is retrieved from self.bad_channels. Alternatively,
            a list of bad channels can be sumbitted.
        reject : bool
            Whether to apply epoch rejection or not. The kind of rejection
            employed depends on the :attr:`.epoch_rejection` class attribute.
        cat : sequence of cell-names
            Only load data for these cells (cells of model).
        decim : None | int
            override the epoch decim factor.
        """
        if subject in self.get_field_values('group'):
            self.set(**kwargs)
            dss = []
            for _ in self.iter(group=subject):
                ds = self.load_epochs(baseline=baseline, ndvar=False,
                                      add_bads=add_bads, reject=reject,
                                      cat=cat)
                dss.append(ds)

            ds = combine(dss)
        else:  # single subject
            self.set(subject=subject, **kwargs)
            ds = self.load_selected_events(add_bads=add_bads, reject=reject,
                                           add_proj=add_proj)
            if reject and self._params['rej']['kind'] == 'auto':
                reject_arg = self._params['rej'].get('threshold', None)
            else:
                reject_arg = None

            if cat:
                model = ds.eval(self.get('model'))
                idx = model.isin(cat)
                ds = ds.sub(idx)

            # load sensor space data
            epoch = self._epoch_state
            target = 'epochs'
            tmin = epoch['tmin']
            tmax = epoch['tmax']
            decim = decim or epoch['decim']
            ds = load.fiff.add_mne_epochs(ds, tmin, tmax, baseline, decim=decim,
                                          target=target, reject=reject_arg)

        if ndvar:
            if ndvar is True:
                ndvar = 'meg'
            else:
                ndvar = str(ndvar)
            ds[ndvar] = load.fiff.epochs_ndvar(ds[target], ndvar)

        return ds

    def load_epochs_stc(self, subject=None, sns_baseline=None, ndvar=False,
                        cat=None):
        """Load a Dataset with stcs for single epochs

        Parameters
        ----------
        subject : str
            Subject(s) for which to load evoked files. Can be a group name
            such as 'all' or a single subject.
        """
        ds = self.load_epochs(subject, baseline=sns_baseline, ndvar=False,
                              cat=cat)
        self.add_epochs_stc(ds, ndvar=ndvar)
        return ds

    def load_events(self, subject=None, add_proj=True, add_bads=True,
                    **kwargs):
        """
        Load events from a raw file.

        Loads events from the corresponding raw file, adds the raw to the info
        dict.

        Parameters
        ----------
        subject : str (state)
            Subject for which to load events.
        add_proj : bool
            Add the projections to the Raw object.
        add_bads : False | True | list
            Add bad channel information to the Raw. If True, bad channel
            information is retrieved from self.bad_channels. Alternatively,
            a list of bad channels can be sumbitted.
        edf : bool
            Loads edf and add it as ``ds.info['edf']``. Edf will only be added
            if ``bool(self.epoch_rejection['edf']) == True``.
        others :
            Update state.
        """
        raw = self.load_raw(add_proj=add_proj, add_bads=add_bads,
                            subject=subject, **kwargs)

        evt_file = self.get('raw-evt-file')
        if os.path.exists(evt_file):
            ds = load.unpickle(evt_file)
        else:
            ds = load.fiff.events(raw)

            subject = self.get('subject')
            if self.has_edf[subject]:  # add edf
                edf = self.load_edf()
                edf.add_t_to(ds)
                ds.info['edf'] = edf

            # cache
            del ds.info['raw']
            save.pickle(ds, evt_file)

        ds.info['raw'] = raw

        subject = subject or self.get('subject')
        experiment = self.get('experiment')

        ds = self.label_events(ds, experiment, subject)
        return ds

    def load_evoked(self, subject=None, baseline=None, ndvar=True, cat=None,
                    **kwargs):
        """
        Load a Dataset with the evoked responses for each subject.

        Parameters
        ----------
        subject : str
            Subject(s) for which to load evoked files. Can be a group name
            such as 'all' or a single subject.
        baseline : None | (tmin, tmax)
            Baseline to apply to evoked response.
        ndvar : bool | str
            Convert the mne Evoked objects to an NDVar. If True, the target
            name is 'meg'.
        cat : sequence of cell-names
            Only load data for these cells (cells of model).
        model : str (state)
            Model according to which epochs are grouped into evoked responses.
        *others* : str
            State parameters.
        """
        if subject in self.get_field_values('group'):
            self.set(**kwargs)
            dss = []
            for _ in self.iter(group=subject):
                ds = self.load_evoked(baseline=baseline, ndvar=False, cat=cat)
                dss.append(ds)

            ds = combine(dss)

            # check consistency
            for name in ds:
                if isinstance(ds[name][0], (mne.fiff.Evoked, mne.SourceEstimate)):
                    lens = np.array([len(e.times) for e in ds[name]])
                    ulens = np.unique(lens)
                    if len(ulens) > 1:
                        err = ["Unequel time axis sampling (len):"]
                        subject = ds['subject']
                        for l in ulens:
                            idx = (lens == l)
                            err.append('%i: %r' % (l, subject[idx].cells))
                        raise DimensionMismatchError(os.linesep.join(err))

        else:  # single subject
            self.set(subject=subject, **kwargs)
            path = self.get('evoked-file', make=True)
            ds = load.unpickle(path)

            if cat:
                model = ds.eval(self.get('model'))
                idx = model.isin(cat)
                ds = ds.sub(idx)

            # baseline correction
            if isinstance(baseline, str):
                raise NotImplementedError
            elif baseline:
                if ds.info.get('evoked', ('evoked',)) != ('evoked',):
                    raise NotImplementedError
                for e in ds['evoked']:
                    rescale(e.data, e.times, baseline, 'mean', copy=False)

        # convert to NDVar
        if ndvar:
            if ndvar is True:
                ndvar = 'meg'

            keys = [k for k in ds if isdatalist(ds[k], mne.fiff.Evoked, False)]
            for k in keys:
                if len(keys) > 1:
                    ndvar_key = '_'.join((k, ndvar))
                else:
                    ndvar_key = ndvar
                ds[ndvar_key] = load.fiff.evoked_ndvar(ds[k])

        return ds

    def load_evoked_stc(self, subject=None, sns_baseline=None,
                        src_baseline=None, sns_ndvar=False, ind_stc=True,
                        ind_ndvar=False, morph_stc=False, morph_ndvar=False,
                        cat=None, **kwargs):
        """Load evoked source estimates.

        Parameters
        ----------
        subject : str
            Subject(s) for which to load evoked files. Can be a group name
            such as 'all' or a single subject.
        sns_baseline : None | str | tuple
            Sensor space baseline correction.
        src_baseline : None | str | tuple
            Source space baseline correctoin.
        ind_stc : bool
            Add source estimates on individual brains as list of
            :class:`mne.SourceEstimate`.
        ind_ndvar : bool
            Add source estimates on individual brains as :class:`NDVar`.
        morph_stc : bool
            Add source estimates morphed to the common brain as list of
            :class:`mne.SourceEstimate`.
        morph_ndvar : bool
            Add source estimates morphed to the common brain as :class:`NDVar`.
        cat : sequence of cell-names
            Only load data for these cells (cells of model).
        *others* : str
            State parameters.
        """
        ds = self.load_evoked(subject=subject, baseline=sns_baseline,
                              ndvar=sns_ndvar, cat=cat, **kwargs)
        self.add_evoked_stc(ds, ind_stc, ind_ndvar, morph_stc, morph_ndvar,
                            src_baseline)
        return ds

    def load_inv(self, fiff, **kwargs):
        """Load the inverse operator

        Parameters
        ----------
        fiff : Raw | Epochs | Evoked | ...
            Object for which to make the inverse operator (provides the mne
            info dictionary).
        others :
            State parameters.
        """
        self.set(**kwargs)

        fwd_file = self.get('fwd-file', make=True)
        fwd = mne.read_forward_solution(fwd_file, surf_ori=True)
        cov = mne.read_cov(self.get('cov-file', make=True))
        if self._params['reg_inv']:
            cov = mne.cov.regularize(cov, fiff.info)
        inv = make_inverse_operator(fiff.info, fwd, cov,
                                    **self._params['make_inv_kw'])
        return inv

    def load_label(self, **kwargs):
        """Load an mne label file."""
        self.set(**kwargs)
        fname = self.get('label-file')
        return self._label_cache[fname]

    def load_raw(self, add_proj=True, add_bads=True, preload=False, **kwargs):
        """
        Load a raw file as mne Raw object.

        Parameters
        ----------
        add_proj : bool
            Add the projections to the Raw object. This does *not* set the
            proj variable.
        add_bads : False | True | list
            Add bad channel information to the Raw. If True, bad channel
            information is retrieved from self.bad_channels. Alternatively,
            a list of bad channels can be sumbitted.
        preload : bool
            Mne Raw parameter.
        """
        self.set(**kwargs)

        if add_proj:
            proj = self.get('proj')
            if proj:
                proj = self.get('proj-file')
            else:
                proj = None
        else:
            proj = None

        raw_file = self.get('raw-file', make=True)
        raw = load.fiff.mne_raw(raw_file, proj, preload=preload)
        if add_bads:
            if add_bads is True:
                key = self.get('raw-key')
                bad_chs = self.bad_channels[key]
            else:
                bad_chs = add_bads

            raw.info['bads'].extend(bad_chs)

        return raw

    def load_selected_events(self, subject=None, reject=True, add_proj=True,
                             add_bads=True, index=True, **kwargs):
        """
        Load events and return a subset based on epoch and rejection

        Parameters
        ----------
        subject : str
            Subject(s) for which to load evoked files. Can be a group name
            such as 'all' or a single subject.
        reject : bool | 'keep'
            Reject bad trials. For True, bad trials are removed from the
            Dataset. For 'keep', the 'accept' variable is added to the Dataset
            and bad trials are kept.
        add_proj : bool
            Add the projections to the Raw object.
        add_bads : False | True | list
            Add bad channel information to the Raw. If True, bad channel
            information is retrieved from self.bad_channels. Alternatively,
            a list of bad channels can be sumbitted.
        index : bool | str
            Index the Dataset before rejection (provide index name as str).
        others :
            Update the experiment state.

        Warning
        -------
        For automatic rejection: Since no epochs are loaded, no rejection
        based on thresholding is performed.
        """
        self.set(**kwargs)
        if subject in self.get_field_values('group'):
            dss = [self.load_selected_events(reject=reject, add_proj=add_proj,
                                             add_bads=add_bads, index=index)
                   for _ in self.iter(group=subject)]
            ds = combine(dss)
            return ds

        self.set(subject=subject)
        epoch = self._epoch_state

        ds = self.load_events(add_proj=add_proj, add_bads=add_bads)
        stimvar = epoch['stimvar']
        stim = epoch['stim']
        stimvar = ds[stimvar]
        if '|' in stim:
            idx = stimvar.isin(stim.split('|'))
        else:
            idx = stimvar == stim
        ds = ds.sub(idx)

        if index:
            idx_name = index if isinstance(index, str) else 'index'
            ds.index(idx_name)

        if reject:
            if reject not in (True, 'keep'):
                raise ValueError("Invalid reject value: %r" % reject)

            if self._params['rej']['kind'] in ('manual', 'make'):
                # if rejections come from different epoch, increase level
                rej_epoch = epoch.get('rej_epoch', None)
                if rej_epoch is not None:
                    level = self._increase_depth()
                    self.set(epoch=rej_epoch)
                path = self.get('rej-file')
                if rej_epoch is not None:
                    self.reset(level - 1)

                if not os.path.exists(path):
                    err = ("The rejection file at %r does not exist. Run "
                           ".make_rej() first." % path)
                    raise RuntimeError(err)

                ds_sel = load.unpickle(path)
                if not np.all(ds['trigger'] == ds_sel['trigger']):
                    err = ("The epoch selection file contains different "
                           "events than the data. Something went wrong...")
                    raise RuntimeError(err)
                if reject == 'keep':
                    ds['accept'] = ds_sel['accept']
                elif reject == True:
                    ds = ds.sub(ds_sel['accept'])
                else:
                    err = ("reject parameter must be bool or 'keep', not "
                           "%r" % reject)
                    raise ValueError(err)
            else:
                use = self._params['rej'].get('edf', False)
                if use:
                    edf = ds.info['edf']
                    tmin = epoch.get('reject_tmin', epoch['tmin'])
                    tmax = epoch.get('reject_tmax', epoch['tmax'])
                    if reject == 'keep':
                        edf.mark(ds, tstart=tmin, tstop=tmax, use=use)
                    else:
                        ds = edf.filter(ds, tstart=tmin, tstop=tmax, use=use)

        return ds

    def load_ttest(self, group=None, c1=None, c0=0, blc='sns', **kwargs):
        """Load a ttest result.

        Parameters
        ----------
        group : str (state)
            The group to use (group name, or subject name for single subject
            ttest)
        c1 : None | str | tuple
            Test condition (cell in model). If None, the grand average is
            used and c0 has to be a scalar.
        c0 : str | scalar
            Control condition (cell on model) or scalar against which to
            compare c1.
        blc : 'sns' | 'src' | None
            Whether to perform baseline correction in sensor space, source
            space, or not at all.
        """
        group = self.get('group', group=group, **kwargs)
        name = self._get_ttest_name(c1, c0, blc, True)
        src = self.get('res-file', kind='data', analysis=analysis_source,
                       name=name, ext='pickled')
        res = load.unpickle(src)
        return res

    def make_besa_evt(self, epoch=None, redo=False):
        """Make the trigger and event files needed for besa

        Parameters
        ----------
        epoch : epoch definition (see module documentation)
            name of the epoch for which to produce the evt files.
        redo : bool
            If besa files already exist, overwrite them.

        Notes
        -----
        Ignores the *decim* epoch parameter.

        Target files are saved relative to the *besa-root* location.
        """
        self.set(epoch=epoch)
        epoch = self._epoch_state

        stim = epoch['stim']
        tmin = epoch['tmin']
        tmax = epoch['tmax']
        rej = self.get('rej')

        trig_dest = self.get('besa-trig', rej='', mkdir=True)
        evt_dest = self.get('besa-evt', rej=rej, mkdir=True)

        if not redo and os.path.exists(evt_dest) and os.path.exists(trig_dest):
            return

        # load events
        ds = self.load_selected_events(reject='keep')
        idx = ds['stim'] == stim
        ds = ds.sub(idx)

        # save triggers
        if redo or not os.path.exists(trig_dest):
            save.meg160_triggers(ds, trig_dest, pad=1)
            if not redo and os.path.exists(evt_dest):
                return
        else:
            ds.index('besa_index', 1)

        # reject bad trials
        ds = ds.sub('accept')

        # save evt
        save.besa_evt(ds, tstart=tmin, tstop=tmax, dest=evt_dest)

    def make_copy(self, temp, field, src, dst, redo=False):
        """Make a copy of a file

        Parameters
        ----------
        temp : str
            Template of the file which to copy.
        field : str
            Field in which the source and target of the link are distinguished.
        src : str
            Value for field on the source file.
        dst : str
            Value for field on the destination filename.
        redo : bool
            If the target file already exists, overwrite it.
        """
        dst_path = self.get(temp, **{field: dst})
        if not redo and os.path.exists(dst_path):
            return

        src_path = self.get(temp, **{field: src})
        shutil.copyfile(src_path, dst_path)

    def make_cov(self, redo=False):
        """Make a noise covariance (cov) file

        Parameters
        ----------
        redo : bool
            If the cov file already exists, overwrite it.
        cov : None | str
            The epoch used for estimating the covariance matrix (needs to be
            a name in .epochs). If None, the experiment state cov is used.
        """
        dest = self.get('cov-file')
        if (not redo) and os.path.exists(dest):
            return

        cov = self.get('cov')
        rej = self.get('cov-rej')
        ds = self.load_epochs(baseline=(None, 0), ndvar=False, decim=1,
                              epoch=cov, rej=rej)
        epochs = ds['epochs']
        cov = mne.cov.compute_covariance(epochs)
        cov.save(dest)

    def make_evoked(self, redo=False, **kwargs):
        """
        Creates datasets with evoked files for the current subject/experiment
        pair.

        Parameters
        ----------
        stimvar : str
            Name of the variable containing the stimulus.
        model : str
            Name of the model. No spaces, order matters.
        epoch : epoch specifications
            See the module documentation.

        """
        dest = self.get('evoked-file', mkdir=True, **kwargs)
        if not redo and os.path.exists(dest):
            return

        epoch_names = [ep['name'] for ep in self._params['epochs']]

        # load the epochs
        epoch = self.get('epoch')
        dss = [self.load_epochs(epoch=name) for name in epoch_names]
        self.set(epoch=epoch)

        # find the events common to all epochs
        idx = reduce(np.intersect1d, (ds['index'] for ds in dss))

        # reduce datasets to common events and aggregate
        model = self.get('model')
        drop = ('i_start', 't_edf', 'T', 'index')
        for i in xrange(len(dss)):
            ds = dss[i]
            index = ds['index']
            ds_idx = index.isin(idx)
            if ds_idx.sum() < len(ds_idx):
                ds = ds[ds_idx]

            dss[i] = ds.aggregate(model, drop_bad=True, drop=drop)

        if len(dss) == 1:
            ds = dss[0]
            ds.rename('epochs', 'evoked')
            ds.info['evoked'] = ('evoked',)
        else:
            for name, ds in zip(epoch_names, dss):
                ds.rename('epochs', name)
            ds = combine(dss)
            ds.info['evoked'] = tuple(epoch_names)

        save.pickle(ds, dest)

    def make_fwd(self, redo=False):
        """Make the forward model"""
        fname = self.get('fwd-file')
        if not redo and os.path.exists(fname):
            return

        info = self.get('raw-file', make=True)
        mri = self.get('trans-file')
        src = self.get('src-file', make=True)
        bem = self.get('bem-sol-file')

        mne.make_forward_solution(info, mri, src, bem, fname, ignore_ref=True,
                                  overwrite=True)

    def _get_ttest_name(self, c1, c0, blc, cov=False):
        blc = blc or ''
        if blc not in ('', 'sns', 'src', 'sns-src'):
            raise ValueError("blc = %r" % blc)

        name = '{epoch}_'
        if c1 is None:
            self.set(model='')
            if not np.isscalar(c0):
                err = ("For tests of the grand average (c1=None), the control "
                       "condition (c0) needs to be a scalar. Got %r." % c0)
                raise TypeError(err)
            name += 'gav'
        elif isinstance(c1, basestring):
            name += c1 + '>' + str(c0)
        elif isinstance(c1, tuple):
            if not isinstance(c0, tuple):
                err = "If c1 is a tuple, c0 must also be a tuple (got %r)" % c0
                raise TypeError(err)
            name += ','.join(c1) + '>' + ','.join(c0)
        else:
            raise TypeError("c1 needs to be None, str or tuple, got %r" % c1)

        if blc:
            name += '_%s-blc' % blc

        if cov:
            name += '_{cov}'

        return name

    def make_ttest(self, group=None, c1=None, c0=0, blc='sns', redo=False,
                   **kwargs):
        """Make a t-test movie

        Parameters
        ----------
        group : str (state)
            The group to use (group name, or subject name for single subject
            ttest)
        c1 : None | str | tuple
            Test condition (cell in model). If None, the grand average is
            used and c0 has to be a scalar.
        c0 : str | scalar
            Control condition (cell on model) or scalar against which to
            compare c1.
        blc : 'sns' | 'src' | 'sns-src' | None
            Whether to perform baseline correction in sensor space, source
            space, or not at all.
        """
        group = self.get('group', group=group, **kwargs)
        name = self._get_ttest_name(c1, c0, blc, True)
        dst = self.get('res-file', kind='data', analysis=analysis_source,
                       name=name, ext='pickled', mkdir=True)
        if not redo and os.path.exists(dst):
            return

        sns_baseline = (None, 0) if blc.startswith('sns') else None
        src_baseline = (None, 0) if blc.endswith('src') else None
        model = self.get('model') or None
        if c1 is None:
            cat = None
            if not src_baseline and not self.get('inv').startswith('fixed'):
                raise ValueError("Grand average test without source space "
                                 "baseline correction")
        elif isinstance(c0, (basestring, tuple)):
            cat = (c1, c0)
        else:
            cat = (c1,)
        cat = cat or None
        if group in self.get_field_values('group'):
            ds = self.load_evoked_stc(group, ind=False, morph=True, ndvar=True,
                                      sns_baseline=sns_baseline,
                                      src_baseline=src_baseline, cat=cat)
            if isinstance(c0, (basestring, tuple)):
                res = testnd.ttest_rel('stcm', model, c1, c0, match='subject',
                                       ds=ds)
            else:
                res = testnd.ttest_1samp('stcm', model, c1, c0, match='subject',
                                         ds=ds)
        elif group in self.get_field_values('subject'):
            ds = self.load_epochs_stc(group, ndvar=True,
                                      sns_baseline=sns_baseline,
                                      src_baseline=src_baseline, cat=cat)
            if isinstance(c0, (basestring, tuple)):
                res = testnd.ttest_ind('stc', model, c1, c0, ds=ds)
            else:
                res = testnd.ttest_1samp('stc', model, c1, c0, ds=ds)
        else:
            err = ("Group %r is neiter a group nor a subject." % group)
            raise ValueError(err)

        save.pickle(res, dst)

    def make_ttest_movie(self, group=None, c1=None, c0=0, blc='sns', p0=.01,
                         p1=0.001, view=1, surf=None, redo=False, dtmin=.005,
                         **kwargs):
        """
        Parameters
        ----------
        group : str (state)
            The group to use (group name, or subject name for single subject
            ttest)
        c1 : None | str | tuple
            Test condition (cell in model). If None, the grand average is
            used and c0 has to be a scalar.
        c0 : str | scalar
            Control condition (cell on model) or scalar against which to
            compare c1.
        blc : 'sns' | 'src' | 'sns-src' | None
            Whether to perform baseline correction in sensor space, source
            space, or not at all.
        p0, p1 : scalar, 0 < p < 1
            P thresholds for color map.
        view : int
            View preset for the movie.
        surf : None | str
            Surface to use (with None, use the view preset).
        """
        group = self.get('group', group=group, **kwargs)
        name = self._get_ttest_name(c1, c0, blc, True)
        plt_name = '_%s' % str(view)
        if surf:
            plt_name += surf
        plt_name += '-{}|{}'.format(str(p0)[2:], str(p1)[2:])
        if dtmin:
            plt_name += '>%ims' % (dtmin * 1000)
        dst = self.get('res-file', kind='movie', analysis=analysis_source,
                       name=name + plt_name, ext='mov', mkdir=True)
        if not redo and os.path.exists(dst):
            return

        src = self.get('res-file', kind='data', analysis=analysis_source,
                       name=name, ext='pickled')

        res = load.unpickle(src)

        if surf is None:
            if view == 1:
                surf = 'inflated'
            elif view == 2:
                surf = 'smoothwm'
            else:
                surf = 'inflated'

        p = plot.brain.stat(res.p, res.t, p0=p0, p1=p1, surf=surf, dtmin=dtmin)
        p.save_movie(dst, view=view)
        p.close()

    def make_labels(self, redo=False, **kwargs):
        """

        Notes
        -----
        Specific to aparc and will set the

        """
        mrisubject = self.get('mrisubject', **kwargs)
        mri_sdir = self.get('mri-sdir')
        mri_dir = self.get('mri-dir')

        # scaled MRI
        if is_fake_mri(mri_dir):
            common_brain = self.get('common_brain')
            self.make_labels(mrisubject=common_brain)
            self.set(mrisubject=mrisubject)
            scale_labels(mrisubject, overwrite=redo, subjects_dir=mri_sdir)
            return

        # original MRI
        label_dir = self.get('label-dir')
        if not os.path.exists(label_dir) or not os.listdir(label_dir):
            annot = self.get('annot')
            subp.mri_annotation2label(mrisubject, annot=annot,
                                      subjects_dir=mri_sdir)

        # split:  (src, axis, (dst1, ...))
        # merge:  (dst, (src1, ...))
        labels = (
                  # split:
                  ('superiortemporal', 1, ('pSTG', 'mSTG', 'aSTG')),
                  ('middletemporal', 1, ('pMTG', 'mMTG', 'aMTG')),
                  ('inferiortemporal', 1, ('pITG', 'mITG', 'aITG')),
                  ('fusiform', 1, ('pFFG', 'mFFG', 'aFFG')),

                  # merge:
                  ('OFC', ('medialorbitofrontal', 'lateralorbitofrontal')),
                  ('IFG', ('parsopercularis', 'parstriangularis',
                           'parsorbitalis')),
                  ('V', ('pericalcarine', 'cuneus', 'lingual',
                         'lateraloccipital')),
                  ('LTL', ('superiortemporal', 'middletemporal',
                           'inferiortemporal')),
                  ('aTL', ('aSTG', 'aMTG', 'aITG')),
                  ('mTL', ('mSTG', 'mMTG', 'mITG')),
                  ('pTL', ('pSTG', 'pMTG', 'pITG')),
                  ('amTL', ('aTL', 'mTL')),
                  ('PTL', ('pSTG', 'bankssts', 'pMTG', 'pITG')))

        for item in labels:
            if len(item) == 3:
                src, axis, dst_names = item
                for _ in self.iter('hemi'):
                    dst_paths = [self.get('label-file', label=name) for name
                                 in dst_names]
                    if (not redo) and all(os.path.exists(pth)
                                          for pth in dst_paths):
                        continue

                    src_label = self.load_label(label=src)

                    labels = split_label(src_label, axis=axis)
                    for name, label, path in zip(dst_names, labels, dst_paths):
                        label.name = name
                        label.save(path)
            else:
                dst, srcs = item
                for _ in self.iter('hemi'):
                    dst_file = self.get('label-file', label=dst)
                    if (not redo) and os.path.exists(dst_file):
                        continue

                    src_labels = (self.load_label(label=name) for name in srcs)
                    label = reduce(add, src_labels)
                    label.save(dst_file)

    def make_link(self, temp, field, src, dst, redo=False):
        """Make a hard link at the file with the dst value on field, linking to
        the file with the src value of field

        Parameters
        ----------
        temp : str
            Template of the file for which to make a link.
        field : str
            Field in which the source and target of the link are distinguished.
        src : str
            Value for field on the source file.
        dst : str
            Value for field on the destination filename.
        redo : bool
            If the target file already exists, overwrite it.
        """
        dst_path = self.get(temp, **{field: dst})
        if not redo and os.path.exists(dst_path):
            return

        src_path = self.get(temp, **{field: src})
        os.link(src_path, dst_path)

    def make_morph_map(self, redo=False, workers=2):
        """Run the mne_make_morph_map utility

        Parameters
        ----------
        redo : bool
            Redo existing files.
        workers : int
            See :meth:`.run_subp()`
        """
        cmd = ["mne_make_morph_maps",
               '--to', self.get('common_brain'),
               '--from', self.get('mrisubject')]

        if redo:
            cmd.appen('--redo')

        self.run_subp(cmd, workers=workers)

    def make_proj(self, save=True, save_plot=True):
        """
        computes the first ``n_mag`` PCA components, plots them, and asks for
        user input (a tuple) on which ones to save.

        Parameters
        ----------
        epochs : mne.Epochs
            epochs which should be used for the PCA
        dest : str(path)
            path where to save the projections
        n_mag : int
            number of components to compute
        save : False | True | tuple
            False: don'r save proj fil; True: manuall pick componentws to
            include in the proj file; tuple: automatically include these
            components
        save_plot : False | str(path)
            target path to save the plot
        """
        proj = self.get('proj')
        opt = self.projs[proj]
        reject = {'mag': 1.5e-11}  # far lower than the data range
        if opt['base'] == 'raw':
            raw = self.load_raw(add_proj=False)

            # select time range of events
            events = load.fiff.events(raw)
            time = events['i_start'] / events.info['samplingrate']
            tstart = time.x.min() - 5
            tstop = time.x.max() + 5

            projs = mne.compute_proj_raw(raw, tstart, tstop, duration=5,
                                         n_grad=0, n_mag=9, n_eeg=0,
                                         reject=reject, n_jobs=1)
            fiff_obj = raw
        else:
            epoch = opt['base']
            rej = opt.get('rej', None)
            rej_ = '' if rej is None else rej
            self.set(epoch=epoch, rej=rej_)
            ds = self.load_epochs(add_proj=False)
            epochs = ds['epochs']
            if rej is None:
                epochs.reject = reject.copy()
                epochs._reject_setup()
            projs = mne.compute_proj_epochs(epochs, n_grad=0, n_mag=9, n_eeg=0)
            fiff_obj = epochs

        # convert projs to NDVar
        picks = mne.epochs.pick_types(fiff_obj.info, exclude='bads')
        sensor = load.fiff.sensor_dim(fiff_obj, picks=picks)
        projs_ndvars = []
        for p in projs:
            d = p['data']['data'][0]
            name = p['desc'][-5:]
            v = NDVar(d, (sensor,), name=name)
            projs_ndvars.append(v)

        # plot PCA components
        proj_file = self.get('proj-file')
        p = plot.Topomap(projs_ndvars, title=proj_file, ncol=3, w=9)
        if save_plot:
            dest = self.get('proj-plot')
            p.figure.savefig(dest)
        if save:
            rm = save
            title = "Select Projections"
            msg = ("which Projections do you want to select? (tuple / 'x' to "
                   "abort)")
            while not isinstance(rm, tuple):
                answer = ui.ask_str(msg, title, default='(0,)')
                rm = eval(answer)
                if rm == 'x': raise

            p.close()
            projs = [projs[i] for i in rm]
            mne.write_proj(proj_file, projs)

    def make_raw(self, redo=False, n_jobs=1, **kwargs):
        """Make a raw file

        Parameters
        ----------
        raw : str
            Name of the raw file to make.
        redo : bool
            If the file already exists, recreate it.
        n_jobs : int
            Number of processes for multiprocessing.

        Notes
        -----
        Due to the electronics of the KIT system sensors, signal lower than
        0.16 Hz is not recorded even when recording at DC.
        """
        dst = self.get('raw-file', **kwargs)
        if not redo and os.path.exists(dst):
            return

        raw_dst = self.get('raw')
        raw_src = 'clm'
        if raw_dst == raw_src:
            err = ("Raw %r can not be made (target same as source)" % raw_dst)
            raise ValueError(err)

        apply_proj = False
        raw = self.load_raw(raw=raw_src, add_proj=apply_proj, add_bads=False,
                            preload=True)
        if apply_proj:
            raw.apply_projector()

        if raw_dst == '0-40':
            raw.filter(None, 40, n_jobs=n_jobs, method='iir')
        elif raw_dst == '1-40':
            raw.filter(1, 40, n_jobs=n_jobs, method='iir')
        elif raw_dst == 'lp40':
            raw.filter(None, 40, n_jobs=n_jobs)
        elif raw_dst == 'hp1-lp40':
            raw.filter(1, 40, n_jobs=n_jobs)
        else:
            raise ValueError('raw = %r' % raw_dst)

        self.set(raw=raw_dst)
        raw.save(dst, overwrite=True)

    def make_rej(self, **kwargs):
        """Show the SelectEpochs GUI to do manual epoch selection for a given
        epoch

        The GUI is opened with the correct file name; if the corresponding
        file exists, it is loaded, and upon saving the correct path is
        the default.

        Parameters
        ----------
        kwargs :
            Kwargs for SelectEpochs
        """
        rej_args = self._params['rej']
        if not rej_args['kind'] == 'manual':
            err = ("Epoch rejection kind for rej=%r is not manual. See the "
                   ".epoch_rejection class attribute." % self.get('rej'))
            raise RuntimeError(err)

        ds = self.load_epochs(ndvar=True, add_bads=False, reject=False,
                              decim=rej_args.get('decim', 5))
        path = self.get('rej-file', mkdir=True)

        from ..wxgui import MEG
        mark = rej_args.get('eog_sns', None)
        bad_chs = self.bad_channels[self.get('raw-key')]
        MEG.SelectEpochs(ds, data='meg', path=path, mark=mark, bad_chs=bad_chs,
                         **kwargs)  # nplots, plotsize,

    def make_src(self, redo=False):
        """Make the source space

        Parameters
        ----------
        redo : bool
            Recreate the source space even if the corresponding file already
            exists.
        """
        dst = self.get('src-file')
        if not redo and os.path.exists(dst):
            return

        src = self.get('src')
        kind, param = src.split('-')

        subject = self.get('mrisubject')
        subjects_dir = self.get('mri-sdir')
        try:
            cfg = mne.coreg.read_mri_cfg(subject, subjects_dir)
            is_scaled = True
        except IOError:
            is_scaled = False

        if is_scaled:
            # make sure the source space exists for the original
            subject_from = cfg['subject_from']
            self.set(mrisubject=subject_from)
            self.make_src()
            self.set(mrisubject=subject)
            mne.scale_source_space(subject, src, subjects_dir=subjects_dir)
        else:
            if kind == 'vol':
                cmd = ['mne_volume_source_space',
                       '--bem', self.get('bem-file'),
                       '--grid', param,
                       '--all',
                       '--src', dst]
                self.run_subp(cmd, workers=0)
            else:
                spacing = kind + param
                mne.setup_source_space(subject, spacing=spacing,
                                       subjects_dir=subjects_dir)

    def makeplt_coreg(self, redo=False, **kwargs):
        """
        Save a coregistration plot

        """
        self.set(**kwargs)

        fname = self.get('plot-file', name='{subject}_{experiment}',
                         analysis='coreg', ext='png', mkdir=True)
        if not redo and os.path.exists(fname):
            return

        from mayavi import mlab
        p = self.plot_coreg()
        p.save_views(fname, overwrite=True)
        mlab.close()

    def next(self, field='subject'):
        """Change field to the next value

        Parameters
        ----------
        field :
        """
        current = self.get(field)
        values = self.get_field_values(field)
        if current in values:
            idx = values.index(current) + 1
            if idx == len(values):
                next_ = values[0]
                print("The last %s was reached; rewinding to "
                      "%r" % (field, next_))
            else:
                next_ = values[idx]
                print("%s: %r -> %r" % (field, current, next_))
        else:
            err = ("The current %s %r is not in %s "
                   "values." % (field, current, field))
            raise RuntimeError(err)

        self.set(**{field: next_})

    def plot_annot(self, annot=None, surf='smoothwm', mrisubject=None,
                   borders=True, label=True):
        mrisubject = self.get('mrisubject', mrisubject=mrisubject, match=False)
        brain = self.plot_brain(surf=surf)
        brain.add_annotation(self.get('annot', annot=annot), borders=borders)

        if label:
            if label is True:
                label = '%s: %s' % (mrisubject, annot)
            from mayavi import mlab
            mlab.text(.05, .05, label, color=(0, 0, 0), figure=brain._f)

        return brain

    def plot_brain(self, surf='smoothwm', mrisubject=None, new=True):
        from mayavi import mlab
        import surfer
        self.set_env()

        mrisubject = self.get('mrisubject', mrisubject=mrisubject, match=False)
        hemi = self.get('hemi')

        opts = dict(background=(1, 1, 1), foreground=(0, 0, 0))
        figure = mlab.figure()
        brain = surfer.Brain(mrisubject, hemi, surf, config_opts=opts,
                             figure=figure)
        self.brain = brain
        return brain

    def plot_coreg(self, **kwargs):
        from ..data.plot.coreg import dev_mri
        self.set(**kwargs)
        raw = mne.fiff.Raw(self.get('raw-file'))
        return dev_mri(raw)

    def rm_fake_mris(self, exclude=[], confirm=False):
        """
        Remove all fake MRIs and trans files

        Parameters
        ----------
        exclude : str | list of str
            Exclude these subjects.

        """
        if isinstance(exclude, basestring):
            exclude = [exclude]
        rmd = []  # dirs
        rmf = []  # files
        sub = []
        for _ in self.iter(['subject']):
            if self.get('subject') in exclude:
                continue
            mri_sdir = self.get('mri-dir')
            if os.path.exists(mri_sdir):
                if is_fake_mri(mri_sdir):
                    rmd.append(mri_sdir)
                    sub.append(self.get('subject'))
                    trans = self.get('trans-file', match=False)
                    if os.path.exists(trans):
                        rmf.append(trans)

        if not rmf and not rmd:
            ui.message("No Fake MRIs Found", "")
            return

        if ui.ask("Delete %i Files?" % (len(rmd) + len(rmf)),
                  '\n'.join(sorted(rmd + rmf)), default=False):
            map(shutil.rmtree, rmd)
            map(os.remove, rmf)
            for s in sub:
                self.set_mri_subject(s, self.get('common_brain'))

    def run_mne_analyze(self, subject=None, modal=False):
        subjects_dir = self.get('mri-sdir')
        subject = subject or self.get('mrisubject')
        fif_dir = self.get('raw-dir', subject=subject)
        subp.run_mne_analyze(fif_dir, subject=subject,
                             subjects_dir=subjects_dir, modal=modal)

    def run_mne_browse_raw(self, subject=None, modal=False):
        fif_dir = self.get('raw-dir', subject=subject)
        subp.run_mne_browse_raw(fif_dir, modal)

    def run_subp(self, cmd, workers=2):
        """
        Add a command to the processing queue.

        Commands should have a form that can be submitted to
        :func:`subprocess.call`.

        Parameters
        ----------
        cmd : list of str
            The command.
        workers : int
            The number of workers to create. For 0, the process is executed in
            interpreter's thread. If > 0, the parameter is only used the
            first time the method is called.

        Notes
        -----
        The task queue can be inspected in the :attr:`queue` attribute
        """
        if cmd is None:
            return

        if workers == 0:
            env = os.environ.copy()
            self.set_env(env)
            mne.utils.run_subprocess(cmd, env=env)
            return

        if not hasattr(self, 'queue'):
            self.queue = Queue()
            env = os.environ.copy()
            self.set_env(env)

            def worker():
                while True:
                    cmd = self.queue.get()
                    subprocess.call(cmd, env=env)
                    self.queue.task_done()

            for _ in xrange(workers):
                t = Thread(target=worker)
                t.daemon = True
                t.start()

        self.queue.put(cmd)

    def set(self, subject=None, **state):
        """
        Set variable values.

        Parameters
        ----------
        subject : str
            Set the `subject` value. The corresponding `mrisubject` is
            automatically set to the corresponding mri subject.
        add : bool
            If the template name does not exist, add a new key. If False
            (default), a non-existent key will raise a KeyError.
        other : str
            All other keywords can be used to set templates.
        """
        if subject is not None:
            state['subject'] = subject
            if 'mrisubject' not in state:
                state['mrisubject'] = self._mri_subjects[subject]

        FileTree.set(self, **state)

    def set_epoch(self, epoch, pad=None):
        """Set the current epoch

        Parameters
        ----------
        epoch : str
            An epoch name for an epoch defined in self.epochs. Several epochs
            can be combined with '|' (but not all functions support linked
            epochs).
        pad : None | scalar
            Pad epochs with this this amount of data (in seconds). Padding is
            not reflected in the epoch descriptors.
        """
        epochs = epoch.split('|')
        epochs.sort()
        epoch = '|'.join(epochs)

        e_descs = []  # full epoch descriptor
        e_descs_nodecim = []  # epoch description without decim
        e_descs_bare = []  # epoch description without decim or rejection
        e_descs_rej = []
        epoch_dicts = []
        stims = set()  # all relevant stims
        for name in epochs:
            # expand epoch description
            ep = self.epoch_default.copy()
            ep.update(self.epochs[name])
            ep['name'] = name

            # make sure stim is ordered
            stim = ep['stim']
            if '|' in stim:
                stim = '|'.join(sorted(set(stim.split('|'))))

            # store expanded epoch
            stims.update(stim.split('|'))
            ep_desc = ep.copy()
            if pad:
                if not 'reject_tmin' in ep:
                    ep['reject_tmin'] = ep['tmin']
                ep['tmin'] -= pad
                if not 'reject_tmax' in ep:
                    ep['reject_tmax'] = ep['tmax']
                ep['tmax'] += pad
            epoch_dicts.append(ep)

            # epoch desc
            desc = self.get_epoch_str(**ep_desc)
            e_descs.append(desc)

            # epoch desc without decim
            ep_desc['decim'] = None
            desc_nd = self.get_epoch_str(**ep_desc)
            e_descs_nodecim.append(desc_nd)

            # epoch for rejection
            ep_rej = ep_desc.copy()
            if 'reject_tmin' in ep_rej:
                ep_rej['tmin'] = ep_rej.pop('reject_tmin')
            if 'reject_tmax' in ep_rej:
                ep_rej['tmax'] = ep_rej.pop('reject_tmax')
            desc_rej = self.get_epoch_str(**ep_rej)
            e_descs_rej.append(desc_rej)

            # bare epoch desc
            ep_desc['reject_tmin'] = None
            ep_desc['reject_tmax'] = None
            desc_b = self.get_epoch_str(**ep_desc)
            desc_b = desc_b.format(rej='')
            e_descs_bare.append(desc_b)

        # store secondary settings
        fields = {'epoch': epoch,
                  'epoch-stim': '|'.join(sorted(stims)),
                  'epoch-desc': '(%s)' % ','.join(sorted(e_descs)),
                  'epoch-nodecim': '(%s)' % ','.join(sorted(e_descs_nodecim)),
                  'epoch-bare': '(%s)' % ','.join(sorted(e_descs_bare)),
                  'epoch-rej': '(%s)' % ','.join(sorted(e_descs_rej))}
        self._fields.update(fields)
        self._params['epochs'] = epoch_dicts

    def _eval_group(self, group):
        if group not in self.get_field_values('group'):
            if group not in self.get_field_values('subject'):
                raise ValueError("No group or subject named %r" % group)
        return group

    def _eval_model(self, model):
        model = [v.strip() for v in model.split('%')]
        model.sort()
        return '%'.join(model)

    def _post_set_rej(self, rej):
        rej_args = self.epoch_rejection[rej]
        self._params['rej'] = rej_args
        cov_rej = rej_args.get('cov-rej', rej)
        self._fields['cov-rej'] = cov_rej

    def _post_set_src(self, src):
        kind, grade = src.split('-')
        grade = int(grade)
        self._params['src'] = (kind, grade)

    def set_env(self, env=os.environ):
        """
        Set environment variables

        for mne/freesurfer:

         - SUBJECTS_DIR
        """
        env['SUBJECTS_DIR'] = self.get('mri-sdir')

    def set_inv(self, ori='free', depth=0.8, reg=False, snr=2, method='dSPM',
                pick_normal=False):
        """Alternative method to set the ``inv`` state.
        """
        items = [ori, depth if depth else None, 'reg' if reg else None, snr,
                 method, 'pick_normal' if pick_normal else None]
        inv = '-'.join(map(str, filter(None, items)))
        self.set(inv=inv)

    def _set_inv_as_str(self, inv):
        """
        Notes
        -----
        inv composed of the following elements, delimited with '-':

         1) 'free' | 'fixed' | float
         2) depth weighting (optional)
         3) regularization 'reg' (optional)
         4) snr
         5) method
         6) pick_normal:  'pick_normal' (optional)
        """
        make_kw = {}
        apply_kw = {}
        args = inv.split('-')
        ori = args.pop(0)
        if ori == 'fixed':
            make_kw['fixed'] = True
            make_kw['loose'] = None
        elif ori == 'free':
            make_kw['loose'] = 1
        else:
            ori = float(ori)
            if not 0 <= ori <= 1:
                err = ('First value of inv (loose parameter) needs to be '
                       'in [0, 1]')
                raise ValueError(err)
            make_kw['loose'] = ori

        method = args.pop(-1)
        if method == 'pick_normal':
            apply_kw['pick_normal'] = True
            method = args.pop(-1)
        if method in ("MNE", "dSPM", "sLORETA"):
            apply_kw['method'] = method
        else:
            err = ('Setting inv with invalid method: %r' % method)
            raise ValueError(err)

        snr = float(args.pop(-1))
        apply_kw['lambda2'] = 1. / snr ** 2

        regularize_inv = False
        if args:
            arg = args.pop(-1)
            if arg == 'reg':
                regularize_inv = True
                if args:
                    depth = args.pop(-1)
                else:
                    depth = None
            else:
                depth = arg

            if depth is not None:
                make_kw['depth'] = float(depth)

        if args:
            raise ValueError("Too many parameters in inv %r" % inv)

        self._fields['inv'] = inv
        self._params['make_inv_kw'] = make_kw
        self._params['apply_inv_kw'] = apply_kw
        self._params['reg_inv'] = regularize_inv

    def set_mri_subject(self, subject, mri_subject=None):
        """
        Reassign a subject's MRI

        Parameters
        ----------
        subject : str
            The (MEG) subject name.
        mri_subject : None | str
            The corresponding MRI subject. None resets to the default
            (mri_subject = subject)
        """
        if mri_subject is None:
            del self._mri_subjects[subject]
        else:
            self._mri_subjects[subject] = mri_subject
        if subject == self.get('subject'):
            self._state['mrisubject'] = mri_subject

    def set_root(self, root, find_subjects=False):
        root = os.path.expanduser(root)
        self._fields['root'] = root
        if not find_subjects:
            return

        subjects = set()
        sub_dir = self.get(self._subject_loc)
        if os.path.exists(sub_dir):
            for dirname in os.listdir(sub_dir):
                isdir = os.path.isdir(os.path.join(sub_dir, dirname))
                if isdir and self.subject_re.match(dirname):
                    subjects.add(dirname)
        else:
            err = ("Subjects directory not found: %r. Initialize with "
                   "parse_subjects=False, or specifiy proper directory in "
                   "experiment._subject_loc." % sub_dir)
            raise IOError(err)

        subjects = sorted(subjects)
        self._field_values['subject'] = subjects

        mrisubjects = [self._mri_subjects[s] for s in subjects]
        common_brain = self.get('common_brain')
        if common_brain:
            mrisubjects.insert(0, common_brain)
        self._field_values['mrisubject'] = mrisubjects

        if len(subjects) == 0:
            print("Warning: no subjects found in %r" % sub_dir)
            return

        # on init, subject is not in fields
        subject = self._fields.get('subject', None)
        if subject not in subjects:
            subject = subjects[0]
        self.set(subject=subject, add=True)

    def show_subjects(self):
        """Print a table with the MRI subject corresponding to each subject"""
        table = fmtxt.Table('ll')
        table.cells('subject', 'mrisubject')
        table.midrule()
        for _ in self.iter('subject'):
            table.cell(self.get('subject'))
            table.cell(self.get('mrisubject'))
        return table

    def show_summary(self, templates=['raw-file'], missing='-', link=' > ',
                     count=True):
        """
        Compile a table about the existence of files by subject

        Parameters
        ----------
        templates : list of str
            The names of the path templates whose existence to list
        missing : str
            The value to display for missing files.
        link : str
            String between file names.
        count : bool
            Add a column with a number for each subject.
        """
        if not isinstance(templates, (list, tuple)):
            templates = [templates]

        results = {}
        experiments = set()
        mris = {}
        for _ in self.iter_temp('raw-key'):
            items = []
            sub = self.get('subject')
            exp = self.get('experiment')
            mri_subject = self.get('mrisubject')
            if sub not in mris:
                if mri_subject == sub:
                    mri_dir = self.get('mri-dir')
                    if not os.path.exists(mri_dir):
                        mris[sub] = 'missing'
                    elif is_fake_mri(mri_dir):
                        mris[sub] = 'fake'
                    else:
                        mris[sub] = 'own'
                else:
                    mris[sub] = mri_subject

            for temp in templates:
                path = self.get(temp, match=False)
                if '*' in path:
                    try:
                        _ = os.path.exists(self.get(temp, match=True))
                        items.append(temp)
                    except IOError:
                        items.append(missing)

                else:
                    if os.path.exists(path):
                        items.append(temp)
                    else:
                        items.append(missing)

            desc = link.join(items)
            results.setdefault(sub, {})[exp] = desc
            experiments.add(exp)

        table = fmtxt.Table('l' * (2 + len(experiments) + count))
        if count:
            table.cell()
        table.cells('Subject', 'MRI')
        experiments = list(experiments)
        table.cells(*experiments)
        table.midrule()

        for i, subject in enumerate(sorted(results)):
            if count:
                table.cell(i)
            table.cell(subject)
            table.cell(mris[subject])

            for exp in experiments:
                table.cell(results[subject].get(exp, '?'))

        return table
