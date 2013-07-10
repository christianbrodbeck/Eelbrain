'''
mne_experiment is a base class for managing an mne experiment.


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
    A name for the epoch; when the resulting data is added to a dataset, this
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
specified in the :attr:`mne_experiment.epochs` dictionary. All keys in this
dictionary have to be of type :class:`str`, values have to be :class:`dict`s
containing the epoch specification. If an epoch is specified in
:attr:`mne_experiment.epochs`, its name (key) can be used in the epochs
argument to various methods. Example::

    # in mne_experiment subclass definition
    class experiment(mne_experiment):
        epochs = {'adjbl': dict(name='bl', stim='adj', tstart=-0.1, tstop=0)}
        ...

The :meth:`mne_experiment.get_epoch_str` method produces A label for each
epoch specification, which is used for filenames. Data which is excluded from
artifact rejection is parenthesized. For example, ``"noun[(-100)0,500]"``
designates data form -100 to 500 ms relative to the stimulus 'noun', with only
the interval form 0 to 500 ms used for rejection.

'''

from collections import defaultdict
from glob import glob
import itertools
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
from mne.transforms.coreg import scale_labels

from .. import fmtxt
from .. import load
from .. import plot
from .. import save
from .. import ui
from ..analyze import testnd
from ..utils import keydefaultdict
from ..utils import subp
from ..utils.com import send_email, Notifier
from ..utils.mne_utils import is_fake_mri, split_label
from .data import (var, ndvar, combine, isdatalist, DimensionMismatchError,
                   UTS)


__all__ = ['mne_experiment', 'LabelCache']

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


def _etree_expand(node, state):
    for tk, tv in node.iteritems():
        if tk == '.':
            continue

        for k, v in state.iteritems():
            name = '{%s}' % tk
            if str(v).startswith(name):
                tv[k] = {'.': v.replace(name, '')}
        if len(tv) > 1:
            _etree_expand(tv, state)


def _etree_node_repr(node, name, indent=0):
    head = ' ' * indent
    out = [(name, head + node['.'])]
    for k, v in node.iteritems():
        if k == '.':
            continue

        out.extend(_etree_node_repr(v, k, indent=indent + 3))
    return out


_temp = {
    'v0': {
        # basic dir
        'meg-sdir': os.path.join('{root}', 'meg'),  # contains subject-name folders for MEG data
        'meg-dir': os.path.join('{meg-sdir}', '{subject}'),
        'mri-sdir': os.path.join('{root}', 'mri'),  # contains subject-name folders for MRI data
        'mri-dir': os.path.join('{mri-sdir}', '{mrisubject}'),
        'bem-dir': os.path.join('{mri-dir}', 'bem'),
        'raw-dir': os.path.join('{meg-dir}', 'raw'),

        # raw
        'raw': 'raw',
        # key necessary for identifying raw file info (used for bad channels):
        'raw-key': '{subject}',
        'raw-base': os.path.join('{raw-dir}', '{subject}_{experiment}_{raw}'),
        'raw-file': '{raw-base}-raw.fif',
        'raw-evt-file': '{raw-base}-evts.pickled',
        'trans-file': os.path.join('{raw-dir}', '{mrisubject}-trans.fif'),  # mne p. 196

        # log-files (eye-tracker etc.)
        'log-dir': os.path.join('{meg-dir}', 'logs'),
        'log-data-file': '{log-dir}/data.txt',
        'log-file': '{log-dir}/log.txt',
        'edf-file': os.path.join('{log-dir}', '*.edf'),

        # mne secondary/forward modeling
        'proj': '',
        'cov': 'bl',
        'proj-file': '{raw-base}_{proj}-proj.fif',
        'proj-plot': '{raw-base}_{proj}-proj.pdf',
        'cov-file': '{raw-base}_{cov}-{rej}-{proj}-cov.fif',
        'bem-file': os.path.join('{bem-dir}', '{mrisubject}-*-bem-sol.fif'),
        'src-file': os.path.join('{bem-dir}', '{mrisubject}-ico-4-src.fif'),
        'fwd-file': '{raw-base}-{mrisubject}_{cov}_{proj}-fwd.fif',
        # inv:
        # 1) 'free' | 'fixed' | float
        # 2) depth weighting (optional)
        # 3) regularization 'reg' (optional)
        # 4) snr
        # 5) method
        # 6) pick_normal:  'pick_normal' (optional)
        'inv': 'free-0.8-2-dSPM',

        # epochs
        'epoch': 'epoch',  # epoch name
        'rej': 'man',  # rejection
        'epoch-stim': None,  # the stimulus/i selected by the epoch
        'epoch-desc': None,  # epoch description
        'epoch-bare': None,  # epoch description without decim or rej
        'epoch-nodecim': None,  # epoch description without decim parameter
        'epoch-sel-file': os.path.join('{meg-dir}', 'epoch_sel', '{raw}_'
                                       '{experiment}_{epoch-nodecim}_sel.'
                                       'pickled'),

        'common_brain': 'fsaverage',

        # evoked
        'evoked-dir': os.path.join('{meg-dir}', 'evoked_{raw}_{proj}'),
        'evoked-file': os.path.join('{evoked-dir}', '{experiment}_'
                                    '{epoch-desc}_{model}_evoked.pickled'),

        # Source space
        'annot': 'aparc',
        'label-dir': os.path.join('{mri-dir}', 'label', '{annot}'),
        'hemi': ('lh', 'rh'),
        'label-file': os.path.join('{label-dir}', '{hemi}.{label}.label'),

        # (method) plots
        'plot-dir': os.path.join('{root}', 'plots'),
        'plot-file': os.path.join('{plot-dir}', '{analysis}', '{name}.{ext}'),

        # result output files
        # group/subject
        'group': 'all',  # analysis performed over (can be single subject)
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
    }



class mne_experiment(object):
    """Class for managing data for an experiment

    Methods
    -------
    Methods for getting information about the experiment:

    .print_tree()
        see templates and their dependency
    .state()
        print variables with their current value
    .list_files()
        List the presence or absence of files for a list of file templates.
    .list_values()
        print a table for all iterable varibales, i.e., those variables for
        which the experiment stores multiple values.
    .subjects_table()
        Each subject with corresponding MRI subject.
    .summary()
        check for the presence of files for a given templates
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
    #     the selection GUI .make_epoch_selection()); 'make' a rejection file
    #     is created by the user
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
                               }
                        }
    epoch_rejection = {}

    exclude = {}  # field_values to exclude (e.g. subjects)

    groups = {}

    owner = None  # email address as string (for notification)

    # Pattern for subject names
    subject_re = re.compile('R\d{4}$')

    _fmt_pattern = re.compile('\{([\w-]+)\}')
    # state variables that are always shown in self.__repr__():
    _repr_vars = ['subject']

    # Where to search for subjects (defined as a template name). If the
    # experiment searches for subjects automatically, it scans this directory
    # for subfolders matching subject_re.
    _subject_loc = 'meg-sdir'

    # basic templates to use. Can be a string referring to a templates
    # dictionary in the module level _temp dictionary, or a templates
    # dictionary
    _templates = 'v0'
    # modify certain template entries from the outset (e.g. specify the initial
    # subject name)
    _defaults = {
                 'experiment': 'experiment_name',
                 # this should be a key in the epochs class attribute (see
                 # above)
                 'epoch': 'epoch'}

    def __init__(self, root=None, parse_subjects=True, **kwargs):
        """
        Parameters
        ----------
        root : str | None
            the root directory for the experiment (usually the directory
            containing the 'meg' and 'mri' directories)
        parse_subjects : bool
            Find MEG subjects using :attr:`_subjects_loc`
        """
        self._parse_subjects = parse_subjects

        # copy class attributes
        self.groups = self.groups.copy()
        self.exclude = self.exclude.copy()

        # epoch rejection settings
        epoch_rejection = self._epoch_rejection.copy()
        epoch_rejection.update(self.epoch_rejection)
        self.epoch_rejection = epoch_rejection

        # find template values:
        temps = self._get_templates(root=root)
        field_values = {}
        field_values['rej'] = tuple(self.epoch_rejection.keys())
        field_values['group'] = self.groups.keys()
        secondary = []  # exclude when resetting
        for k in temps:
            v = temps[k]
            if v is None:
                secondary.append(k)
            elif isinstance(v, tuple):
                field_values[k] = v
                temps[k] = v[0]
            elif not isinstance(v, basestring):
                err = ("Invalid templates field value: %r. Need None, tuple "
                       "or string" % v)
                raise TypeError(err)

        self._field_values = field_values
        self._secondary_fields = tuple(secondary)
        self._state = temps

        # set variables with derived settings
        epoch = temps.get('epoch', None)
        inv = temps.get('inv', None)
        rej = temps.get('rej', self.epoch_rejection.keys()[0])
        self.set(epoch=epoch, inv=inv, rej=rej, add=True)

        # find experiment data structure
        self._mri_subjects = keydefaultdict(lambda k: k)
        self.set(root=root, match=parse_subjects, add=True)

        # set initial values
        self._label_cache = LabelCache()

        # set defaults for any existing field names
        for k in self._state.keys():
            temp = self._state[k]
            for field in self._fmt_pattern.findall(temp):
                if field not in self._state:
                    self._state[field] = ''

        self._initial_state = self._state.copy()

        owner = getattr(self, 'owner', None)
        if owner:
            self.notification = Notifier(owner, 'mne_experiment task')

        self.set(**kwargs)

    def __iter__(self):
        "Iterate state through subjects and yield each subject name."
        for state in self.iter():
            yield state['subject']

    @property
    def _epoch_state(self):
        epochs = self._epochs_state
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
        ds : dataset
            The dataset containing the mne Epochs for the desired trials.
        src : str
            Name of the source epochs in ds.
        dst : str
            Name of the source estimates to be created in ds.
        ndvar : bool
            Add the source estimates as ndvar instead of a list of
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
        stc = apply_inverse_epochs(epochs, inv, **self._apply_inv_kw)

        if ndvar:
            subject = self.get('mrisubject')
            stc = load.fiff.stc_ndvar(stc, subject, 'stc')

        ds[dst] = stc

    def add_evoked_label(self, ds, label, hemi='lh', src='stc'):
        """
        Extract the label time course from a list of SourceEstimates.

        Parameters
        ----------
        label :
            the label's bare name (e.g., 'insula').
        hemi : 'lh' | 'rh' | 'bh' | False
            False assumes that hemi is a factor in ds.
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
        ds[key] = ndvar(np.array(x), dims=('case', time))

    def add_evoked_stc(self, ds, ind='stc', morph='stcm', baseline=None,
                       ndvar=False):
        """
        Add an stc (ndvar) to a dataset with an evoked list.

        Assumes that all Evoked of the same subject share the same inverse
        operator.

        Parameters
        ----------
        ds : dataset
            The dataset containing the Evoked objects.
        ind: bool | str
            Add stcs on individual brains (with str as name, otherwise 'stc').
        morph : bool | str
            Add stcs for data morphed to the common brain (with str as name,
            otherwise 'stcm').
        baseline : None | str | tuple
            Baseline correction in source space.
        ndvar : bool
            Add stcs as ndvar (default is list of mne SourceEstimate objects).
            For individual brain stcs, this option only applies for datasets
            with a single subject.
        """
        if not (ind or morph):
            return

        if isinstance(baseline, str):
            raise NotImplementedError

        # find vars to work on
        do = []
        for name in ds:
            if isinstance(ds[name][0], mne.fiff.Evoked):
                do.append(name)

        # prepare data containers
        invs = {}
        if ind:
            ind = 'stc' if (ind == True) else str(ind)
            stcs = defaultdict(list)
        if morph:
            morph = 'stcm' if (morph == True) else str(morph)
            mstcs = defaultdict(list)

        # convert evoked objects
        common_brain = self.get('common_brain')
        mri_sdir = self.get('mri-sdir')
        for case in ds.itercases():
            subject = case['subject']
            if is_fake_mri(self.get('mri-dir')):
                subject_from = common_brain
            else:
                subject_from = subject

            for name in do:
                evoked = case[name]

                # get inv
                if subject in invs:
                    inv = invs[subject]
                else:
                    inv = self.load_inv(evoked, subject=subject)
                    invs[subject] = inv

                # apply inv
                stc = apply_inverse(evoked, inv, **self._apply_inv_kw)

                # baseline correction
                if baseline:
                    rescale(stc._data, stc.times, baseline, 'mean', copy=False)

                if ind:
                    stcs[name].append(stc)

                if morph:
                    stc = mne.morph_data(subject_from, common_brain, stc, 4,
                                         subjects_dir=mri_sdir)
                    mstcs[name].append(stc)

        # add to dataset
        for name in do:
            if ind:
                if len(do) > 1:
                    key = '%s_%s' % (ind, do)
                else:
                    key = ind

                if ndvar and (len(ds['subject'].cells) == 1):
                    subject = ds['subject'].cells[0]
                    ds[key] = load.fiff.stc_ndvar(stcs[name], subject)
                else:
                    ds[key] = stcs[name]
            if morph:
                if len(do) > 1:
                    key = '%s_%s' % (morph, do)
                else:
                    key = morph

                if ndvar:
                    ds[key] = load.fiff.stc_ndvar(mstcs[name], common_brain)
                else:
                    ds[key] = mstcs[name]

    def _get_templates(self, root=None, **kwargs):
        if isinstance(self._templates, str):
            t = _temp[self._templates].copy()
        else:
            t = self._templates.copy()

        if not isinstance(t, dict):
            err = ("Templates mus be dictionary; got %s" % type(t))
            raise TypeError(err)

        t.update(self._defaults)
        t.update(kwargs)

        # make sure we have a valid root
        if root is not None:
            t['root'] = os.path.expanduser(root)
        elif 'root' not in t:
            msg = "Please select the meg directory of your experiment"
            root = ui.ask_dir("Select Root Directory", msg, True)
            t['root'] = root

        if not os.path.exists(t['root']):
            raise IOError("Specified root path does not exist: %r" % root)

        return t

    def __repr__(self):
        args = [repr(self.get('root'))]
        kwargs = []

        for k in self._repr_vars:
            v = self.get(k)
            kwargs.append((k, repr(v)))

        proper_mrisubject = self._mri_subjects[self._state['subject']]
        for k in sorted(self._state):
            if k == 'root':
                continue
            if k == 'mrisubject' and self._state[k] == proper_mrisubject:
                continue
            if '{' in self._state[k]:
                continue
            if k in self._repr_vars:
                continue

            v = self._state[k]
            if v != self._initial_state[k]:
                kwargs.append((k, repr(v)))

        args.extend('='.join(pair) for pair in kwargs)
        args = ', '.join(args)
        return "%s(%s)" % (self.__class__.__name__, args)

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

    def expand_template(self, temp, values=()):
        """
        Expand a template until all its subtemplates are neither in
        field names or in ``values``

        Parameters
        ----------
        values : container (implements __contains__)
            values which should not be expanded (in addition to
        """
        temp = self._state.get(temp, temp)

        while True:
            stop = True
            for name in self._fmt_pattern.findall(temp):
                if (name in values) or (self._field_values.get(name, False)):
                    pass
                else:
                    temp = temp.replace('{%s}' % name, self._state[name])
                    stop = False

            if stop:
                break

        return temp

    def find_keys(self, temp):
        """
        Find all terminal keys that are relevant for a template.

        Returns
        -------
        keys : set
            All terminal keys that are relevant for foormatting temp.
        """
        keys = set()
        temp = self._state.get(temp, temp)

        for key in self._fmt_pattern.findall(temp):
            value = self._state[key]
            if self._fmt_pattern.findall(value):
                keys = keys.union(self.find_keys(value))
            else:
                keys.add(key)

        return keys

    def format(self, temp, vmatch=True, **kwargs):
        """
        Returns the template temp formatted with current values. Formatting
        retrieves values from self._state iteratively
        """
        self.set(match=vmatch, **kwargs)

        while True:
            variables = self._fmt_pattern.findall(temp)
            if variables:
                temp = temp.format(**self._state)
            else:
                break

        path = os.path.expanduser(temp)
        return path

    def get(self, temp, fmatch=True, vmatch=True, match=True, mkdir=False,
            make=False, **kwargs):
        """
        Retrieve a formatted template

        With match=True, '*' are expanded to match a file,
        and if there is not a unique match, an error is raised. With
        mkdir=True, the directory containing the file is created if it does not
        exist.

        Parameters
        ----------
        temp : str
            Name of the requested template.
        fmatch : bool
            "File-match":
            If the template contains asterisk ('*'), use glob to fill it in.
            An IOError is raised if the pattern fits 0 or >1 files.
        vmatch : bool
            "Value match":
            Require existence of the assigned value (only applies for variables
            with field values)
        match : bool
            Do any matching (i.e., match=False sets fmatch as well as vmatch
            to False).
        mkdir : bool
            If the directory containing the file does not exist, create it.
        make : bool
            If a requested file does not exists, make it if possible.
        kwargs :
            Set any state values.
        """
        if not match:
            fmatch = vmatch = False

        path = self.format('{%s}' % temp, vmatch=vmatch, **kwargs)

        # assert the presence of the file
        if fmatch and ('*' in path):
            paths = glob(path)
            if len(paths) == 1:
                path = paths[0]
            elif len(paths) > 1:
                err = "More than one files match %r: %r" % (path, paths)
                raise IOError(err)
            else:
                raise IOError("No file found for %r" % path)

        # create the directory
        if mkdir:
            dirname = os.path.dirname(path)
            if not os.path.exists(dirname):
                os.makedirs(dirname)

        if make and not os.path.exists(path):
            if temp == 'evoked-file':
                self.make_evoked()
            elif temp == 'fwd-file':
                self.make_fwd()
            elif temp == 'cov-file':
                self.make_cov()

        return path

    def get_epoch_str(self, stimvar=None, stim=None, tmin=None, tmax=None,
                      reject_tmin=None, reject_tmax=None, decim=None,
                      name=None, tag=None):
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

        values = self._field_values[field]
        if exclude:
            exclude = self.exclude.get(field, None)
        if exclude:
            values = [v for v in values if not v in exclude]

        return values

    def iter(self, fields=['subject'], exclude={}, values={}, group=None,
             mail=False, prog=False, **constants):
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
        if mail is True:
            mail = self.owner

        state_ = self._state.copy()

        # set constants
        self.set(**constants)

        if isinstance(fields, basestring):
            fields = [fields]
        fields = list(set(fields).difference(constants).union(values))

        # gather possible values to iterate over
        field_values = {k: self.get_field_values(k) for k in fields}
        if group and (group != 'all') and ('subject' in field_values):
            subjects = field_values['subject']
            group = self.groups[group]
            field_values['subject'] = [s for s in subjects if s in group]
        field_values.update(values)

        # exclude values
        for k in exclude:
            ex = exclude[k]
            if isinstance(ex, basestring):
                ex = (ex,)
            field_values[k] = [v for v in field_values[k] if not v in ex]

        # pick out the fields to iterate, but drop excluded cases:
        v_lists = []
        for field in fields:
            v_lists.append(field_values[field])

        if len(v_lists):
            if prog:
                i_max = np.prod(map(len, v_lists))
                if not isinstance(prog, str):
                    prog = "MNE Experiment Iterator"
                progm = ui.progress_monitor(i_max, prog, "")
                prog = True

            for v_list in itertools.product(*v_lists):
                values = dict(zip(fields, v_list))
                if prog:
                    progm.message(' | '.join(map(str, v_list)))
                self.set(**values)
                yield self._state
                if prog:
                    progm.advance()
        else:
            yield self._state

        self._state.update(state_)

        if mail:
            send_email(mail, "Eelbrain Task Done", "I did as you desired, "
                       "my master.")

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
        initial_value = self.get(field)

        values = self.get_field_values(field)
        if start is not None:
            start = values.index(start)
        if stop is not None:
            stop = values.index(stop) + 1
        idx = slice(start, stop)
        values = values[idx]
        for value in values:
            self.set(**{field: value})
            yield value

        self.set(**{field: initial_value})

    def iter_temp(self, temp, exclude={}, values={}, mail=False, prog=False,
                  **constants):
        """
        Iterate through all paths conforming to a template given in ``temp``.

        Parameters
        ----------
        temp : str
            Name of a template in the mne_experiment.templates dictionary, or
            a path template with variables indicated as in ``'{var_name}'``
        """
        # if the name is an existing template, retrieve it
        keep = constants.keys() + values.keys()
        temp = self.expand_template(temp, values=keep)

        # find variables for iteration
        variables = set(self._fmt_pattern.findall(temp))

        for state in self.iter(variables, exclude=exclude, values=values,
                               mail=mail, prog=prog, **constants):
            path = temp.format(**state)
            yield path

    def iter_vars(self, *args, **kwargs):
        """Deprecated. Use :attr:`.iter()`"""
        warn("mne_experiment.iter_vars() is deprecated. Use .iter()",
             DeprecationWarning)
        kwargs['mail'] = kwargs.get('notify', False)
        self.iter(*args, **kwargs)


    def label_events(self, ds, experiment, subject):
        """
        Adds T (time) and SOA (stimulus onset asynchrony) to the dataset.

        Parameters
        ----------
        ds : dataset
            A dataset containing events (as returned by
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
            ds['SOA'] = var(np.ediff1d(ds['T'].x, 0))
        return ds

    def label_subjects(self, ds):
        """Label the subjects in ds based on .groups

        Parameters
        ----------
        ds : dataset
            A dataset with 'subject' entry.
        """
        subject = ds['subject']
        for name, subjects in self.groups.iteritems():
            ds[name] = var(subject.isin(subjects))

    def list_files(self, files=['raw-file'], count=True, fields=['subject'],
                   **kwargs):
        """
        Compile a table about the existence of files by subject

        Parameters
        ----------
        files : str | list of str
            The names of the path templates whose existence to list.
        count : bool
            Add a column with a number for each subject.
        fields : str | list of str
            The names of the variables for which to list files (i.e., for each
            unique combination of ``fields``, list ``files``).
        """
        if not isinstance(files, (list, tuple)):
            files = [files]
        if not isinstance(fields, (list, tuple)):
            fields = [fields]

        ncol = (len(fields) + len(files))
        table = fmtxt.Table('r' * bool(count) + 'l' * ncol)
        if count:
            table.cell()
        for name in fields + files:
            table.cell(name.capitalize())
        table.midrule()

        for i, _ in enumerate(self.iter(fields, **kwargs)):
            if count:
                table.cell(i)

            for field in fields:
                table.cell(self.get(field))

            for temp in files:
                path = self.get(temp)
                if os.path.exists(path):
                    table.cell(temp)
                else:
                    table.cell('-')

        return table

    def list_values(self, str_out=False):
        """
        Generate a table for all iterable varibales

        i.e., those variables for which the experiment stores multiple values.

        Parameters
        ----------
        str_out : bool
            Return the table as a string (instead of printing it).
        """
        lines = []
        for key in self._field_values:
            values = self.get_field_values(key)
            line = '%s:' % key
            head_len = len(line) + 1
            while values:
                v = repr(values.pop(0))
                if values:
                    v += ','
                if len(v) < 80 - head_len:
                    line += ' ' + v
                else:
                    lines.append(line)
                    line = ' ' * head_len + v

                if not values:
                    lines.append(line)

        table = os.linesep.join(lines)
        if str_out:
            return table
        else:
            print table

    def load_edf(self, **kwargs):
        """Load the edf file ("edf-file" template)"""
        kwargs['fmatch'] = False
        src = self.get('edf-file', **kwargs)
        edf = load.eyelink.Edf(src)
        return edf

    def load_epochs(self, subject=None, baseline=None, ndvar=False,
                    add_bads=True, reject=True, cat=None, decim=None,
                    **kwargs):
        """
        Load a dataset with epochs for a given epoch definition

        Parameters
        ----------
        subject : str
            Subject(s) for which to load evoked files. Can be 'all'a group name
            or a single subject.
        epoch : str
            Epoch definition.
        ndvar : bool | str
            Convert epochs to an ndvar with the given name (if True, 'MEG' is
            uesed).
        add_bads : False | True | list
            Add bad channel information to the Raw. If True, bad channel
            information is retrieved from self.bad_channels. Alternatively,
            a list of bad channels can be sumbitted.
        reject : bool
            Whether to apply epoch rejection or not. The kind of rejection
            employed depends on the :attr:`.epoch_rejection` class attribute.
        cat : list of str or tuple
            Only load evoked for a certain category (cells on model).
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
            ds = self.load_selected_events(add_bads=add_bads, reject=reject)
            if reject and self._rej_args['kind'] == 'auto':
                reject_arg = self._rej_args.get('threshold', None)
            else:
                reject_arg = None

            if cat:
                model = ds.eval(self.get('model'))
                idx = model.isin(cat)
                ds = ds.subset(idx)

            # load sensor space data
            epoch = self._epoch_state
            target = 'epochs'
            tmin = epoch['tmin']
            tmax = epoch['tmax']
            decim = decim or epoch['decim']
            ds = load.fiff.add_mne_epochs(ds, target=target, tmin=tmin,
                                          tmax=tmax, reject=reject_arg,
                                          baseline=baseline, decim=decim)

        if ndvar:
            if ndvar is True:
                ndvar = 'meg'
            else:
                ndvar = str(ndvar)
            ds[ndvar] = load.fiff.epochs_ndvar(ds[target], name=ndvar)

        return ds

    def load_epochs_stc(self, subject=None, sns_baseline=None,
                        src_baseline=None, ndvar=False, cat=None):
        """Load a dataset with stcs for single epochs

        Parameters
        ----------

        """
        ds = self.load_epochs(subject, baseline=sns_baseline, ndvar=False,
                              cat=cat)
        self.add_epochs_stc(ds, baseline=src_baseline, ndvar=ndvar)
        return ds

    def load_events(self, subject=None, add_proj=True, add_bads=True, edf=True,
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

            # add edf
            edf = self.load_edf()
            edf.add_T_to(ds)
            ds.info['edf'] = edf

            # cache
            del ds.info['raw']
            save.pickle(ds, evt_file)

        ds.info['raw'] = raw

        if subject is None:
            subject = self._state['subject']
        experiment = self.get('experiment')

        ds = self.label_events(ds, experiment, subject)
        return ds

    def load_evoked(self, subject=None, ndvar=False, baseline=None, cat=None,
                    **kwargs):
        """
        Load a dataset with evoked files for each subject.

        Load data previously created with :meth:`mne_experiment.make_evoked`.

        Parameters
        ----------
        subject : str
            Subject(s) for which to load evoked files. Can be 'all'a group name
            or a single subject.
        ndvar : bool | str
            Convert the mne Evoked objects to an ndvar. If True, the target
            name is 'meg'.
        model : str (state)
            Model according to which epochs are grouped into evoked responses.
        others :
            State parameters.
        """
        if subject in self.get_field_values('group'):
            self.set(**kwargs)
            dss = []
            for _ in self.iter(group=subject):
                ds = self.load_evoked(ndvar=False, baseline=baseline, cat=cat)
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
                ds = ds.subset(idx)

            # baseline correction
            if isinstance(baseline, str):
                raise NotImplementedError
            elif baseline:
                if ds.info.get('evoked', ('evoked',)) != ('evoked',):
                    raise NotImplementedError
                for e in ds['evoked']:
                    rescale(e.data, e.times, baseline, 'mean', copy=False)

        # convert to ndvar
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

    def load_evoked_stc(self, subject=None, ind=True, morph=False, ndvar=False,
                        sns_baseline=None, src_baseline=None, cat=None,
                        **kwargs):
        """
        Parameters
        ----------
        subject : 'all' | str
            With 'all', a dataset with all subjects is loaded, otherwise a
            single subject.
        ind: bool | str
            Add stcs on individual brains (with str as name, otherwise 'stc').
        morph : bool | str
            Add stcs for data morphed to the common brain (with str as name,
            otherwise 'stcm').
        ndvar : bool
            Add stcs as ndvar (default is list of mne SourceEstimate objects).
            For individual brain stcs, this option only applies for datasets
            with a single subject.
        sns_baseline : None | str | tuple
            Sensor space baseline correction.
        src_baseline : None | str | tuple
            Source space baseline correctoin.
        others : str
            State parameters.
        """
        ds = self.load_evoked(subject=subject, baseline=sns_baseline,
                              ndvar=False, cat=cat, **kwargs)
        self.add_evoked_stc(ds, ind=ind, morph=morph, baseline=src_baseline,
                            ndvar=ndvar)
        return ds

    def load_inv(self, fiff, **kwargs):
        """Load the inverse operator

        Parameters
        ----------
        fiff : Raw | Epochs | Evoked | ...
            Object for which to make the inverse operator (provides the mne
            info dictionary).
        """
        self.set(**kwargs)

        fwd_file = self.get('fwd-file', make=True)
        fwd = mne.read_forward_solution(fwd_file, surf_ori=True)
        cov = mne.read_cov(self.get('cov-file', make=True))
        if self._regularize_inv:
            cov = mne.cov.regularize(cov, fiff.info)
        inv = make_inverse_operator(fiff.info, fwd, cov, **self._make_inv_kw)
        return inv

    def load_label(self, **kwargs):
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

        raw_file = self.get('raw-file')
        raw = load.fiff.Raw(raw_file, proj, preload=preload)
        if add_bads:
            if add_bads is True:
                key = self.get('raw-key')
                bad_chs = self.bad_channels[key]
            else:
                bad_chs = add_bads

            raw.info['bads'].extend(bad_chs)

        return raw

    def load_selected_events(self, reject=True, add_proj=True, add_bads=True,
                             index=True, **kwargs):
        """
        Load events and return a subset based on epoch and rejection

        Parameters
        ----------
        reject : bool | 'keep'
            Reject bad trials. For True, bad trials are removed from the
            dataset. For 'keep', the 'accept' variable is added to the dataset
            and bad trials are kept.
        add_proj : bool
            Add the projections to the Raw object.
        add_bads : False | True | list
            Add bad channel information to the Raw. If True, bad channel
            information is retrieved from self.bad_channels. Alternatively,
            a list of bad channels can be sumbitted.
        index : bool | str
            Index the dataset before rejection (provide index name as str).
        others :
            Update the experiment state.

        Warning
        -------
        For automatic rejection: Since no epochs are loaded, no rejection
        based on thresholding is performed.
        """
        self.set(**kwargs)
        epoch = self._epoch_state

        ds = self.load_events(add_proj=add_proj, add_bads=add_bads)
        stimvar = epoch['stimvar']
        stim = epoch['stim']
        stimvar = ds[stimvar]
        if '|' in stim:
            idx = stimvar.isin(stim.split('|'))
        else:
            idx = stimvar == stim
        ds = ds.subset(idx)

        if index:
            idx_name = index if isinstance(index, str) else 'index'
            ds.index(idx_name)

        if reject:
            if reject not in (True, 'keep'):
                raise ValueError("Invalie reject value: %r" % reject)

            if self._rej_args['kind'] in ('manual', 'make'):
                path = self.get('epoch-sel-file')
                if not os.path.exists(path):
                    err = ("The rejection file at %r does not exist. Run "
                           ".make_epoch_selection() first." % path)
                    raise RuntimeError(err)
                ds_sel = load.unpickle(path)
                if not np.all(ds['eventID'] == ds_sel['eventID']):
                    err = ("The epoch selection file contains different "
                           "events than the data. Something went wrong...")
                    raise RuntimeError(err)
                if reject == 'keep':
                    ds['accept'] = ds_sel['accept']
                elif reject == True:
                    ds = ds.subset(ds_sel['accept'])
                else:
                    err = ("reject parameter must be bool or 'keep', not "
                           "%r" % reject)
                    raise ValueError(err)
            else:
                use = self._rej_args.get('edf', False)
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
        ds = ds.subset(idx)

        # save triggers
        if redo or not os.path.exists(trig_dest):
            save.meg160_triggers(ds, trig_dest, pad=1)
            if not redo and os.path.exists(evt_dest):
                return
        else:
            ds.index('besa_index', 1)

        # reject bad trials
        ds = ds.subset('accept')

        # save evt
        save.besa_evt(ds, tstart=tmin, tstop=tmax, dest=evt_dest)

    def make_cov(self, cov=None, redo=False):
        """Make a noise covariance (cov) file

        Parameters
        ----------
        cov : None | str
            The epoch used for estimating the covariance matrix (needs to be
            a name in .epochs). If None, the experiment state cov is used.
        """
        if (cov is not None) and not isinstance(cov, str):
            raise TypeError("cov should be None or str, no %r" % cov)

        cov = self.get('cov', cov=cov)
        dest = self.get('cov-file')
        if (not redo) and os.path.exists(dest):
            return

        self.set(epoch=cov)
        epoch = self._epoch_state
        stimvar = epoch['stimvar']
        stim = epoch['stim']
        tmin = epoch['tmin']
        tmax = epoch['tmax']

        ds = self.load_events()

        # decimate events
        ds = ds.subset(ds[stimvar] == stim)
        edf = ds.info['edf']
        ds = edf.filter(ds, use=['EBLINK'], tstart=tmin, tstop=tmax)

        # create covariance matrix
        epochs = load.fiff.mne_Epochs(ds, baseline=(None, 0), preload=True,
                                      reject={'mag':3e-12}, tmin=tmin,
                                      tmax=tmax)
        cov = mne.cov.compute_covariance(epochs)
        cov.save(dest)

    def make_epoch_selection(self, **kwargs):
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
        if not self._rej_args['kind'] == 'manual':
            err = ("Epoch rejection kind for rej=%r is not manual. See the "
                   ".epoch_rejection class attribute." % self.get('rej'))
            raise RuntimeError(err)

        ds = self.load_epochs(ndvar=True, add_bads=False, reject=False,
                              decim=self._rej_args.get('decim', 5))
        path = self.get('epoch-sel-file', mkdir=True)

        from ..wxgui.MEG import SelectEpochs
        ROI = self._rej_args.get('eog_sns', None)
        bad_chs = self.bad_channels[self.get('raw-key')]
        SelectEpochs(ds, data='meg', path=path, ROI=ROI, bad_chs=bad_chs,
                     **kwargs)  # nplots, plotsize,

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

        epoch_names = [ep['name'] for ep in self._epochs_state]

        # load the epochs
        epoch = self.get('epoch')
        dss = [self.load_epochs(epoch=name) for name in epoch_names]
        self.set(epoch=epoch)

        # find the events common to all epochs
        idx = reduce(np.intersect1d, (ds['index'] for ds in dss))

        # reduce datasets to common events and compress
        model = self.get('model')
        drop = ('i_start', 't_edf', 'T', 'index')
        for i in xrange(len(dss)):
            ds = dss[i]
            index = ds['index']
            ds_idx = index.isin(idx)
            if ds_idx.sum() < len(ds_idx):
                ds = ds[ds_idx]

            dss[i] = ds.compress(model, drop_bad=True, drop=drop)

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

    def make_filter(self, dest='lp40', hp=None, lp=40, src='raw',
                    apply_proj=False, redo=False, n_jobs=1, **kwargs):
        """
        Make a filtered raw file

        Parameters
        ----------
        dest, src : str
            `raw` names for target and source raw file.
        hp, lp : None | int
            High-pass and low-pass parameters.
        apply_proj : bool
            Apply the projections to the Raw data before filtering.
        redo : bool
            If the target file already exists, redo and overwrite it.
        n_jobs : int
            Number of processes. Warning: PyShell does not support
            multiprocessing and will crash with values > 1.
        kwargs :
            mne.fiff.Raw.filter() kwargs.
        """
        dest_file = self.get('raw-file', raw=dest)
        if (not redo) and os.path.exists(dest_file):
            return

        raw = self.load_raw(add_proj=apply_proj, add_bads=False, preload=True,
                            raw=src)
        if apply_proj:
            raw.apply_projector()
        raw.filter(hp, lp, n_jobs=n_jobs, **kwargs)
        raw.save(dest_file, overwrite=True)

    def make_fwd(self, thread=False):
        """Make the forward model

        Parameters
        ----------
        thread : bool
            Process files in the background. Warning: nothing will prevent the
            user from destroying the ongoing process by terminating the Python
            interpreter. Check :attr:`.queue.unfinished_tasks`.
        """
        cmd = self.make_fwd_cmd()
        self.run_subp(cmd, workers=int(thread))

    def make_fwd_cmd(self, redo=False):
        """
        Returns the mne_do_forward_solution command.

        Relevant templates:
        - raw-file
        - mrisubject
        - src
        - bem
        - trans

        Returns
        -------
        cmd : None | list of str
            The command to run mne_do_forward_solution as it would be
            submitted to subprocess.call(). None if redo=False and the target
            file already exists.

        """
        fwd = self.get('fwd-file')

        cmd = ["mne_do_forward_solution",
               '--subject', self.get('mrisubject'),
               '--src', self.get('src-file'),
               '--bem', self.get('bem-file'),
               '--mri', self.get('trans-file'),
               '--meas', self.get('raw-file'),  # provides sensor locations and coordinate transformation between the MEG device coordinates and MEG head-based coordinates.
               '--fwd', fwd,
               '--megonly']
        if redo:
            cmd.append('--overwrite')
        return cmd

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
            cat += (c1,)
        cat = cat or None
        if group in self.get_field_values('group'):
            ds = self.load_evoked_stc(group, ind=False, morph=True, ndvar=True,
                                      sns_baseline=sns_baseline,
                                      src_baseline=src_baseline, cat=cat)
            res = testnd.ttest('stcm', model, c1, c0, match='subject', ds=ds)
        elif group in self.get_field_values('subject'):
            ds = self.load_epochs_stc(group, ndvar=True,
                                      sns_baseline=sns_baseline,
                                      src_baseline=src_baseline, cat=cat)
            res = testnd.ttest('stc', model, c1, c0, ds=ds)
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
            scale_labels(mrisubject, common_brain, overwrite=redo,
                         subjects_dir=mri_sdir)
            return

        # original MRI
        label_dir = self.get('label-dir')
        if not os.path.exists(label_dir):
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
        temp :
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

    def make_proj_for_epochs(self, epochs, n_mag=5, save=True, save_plot=True):
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
        projs = mne.compute_proj_epochs(epochs, n_grad=0, n_mag=n_mag, n_eeg=0)
        self.ui_select_projs(projs, epochs, save=save, save_plot=save_plot)

    def make_raw(self, raw='hp1-lp40', redo=False, n_jobs=1):
        """Make a raw file

        Parameters
        ----------
        raw : str
            Name of the raw file to make.
        redo : bool
            If the file already exists, recreate it.
        n_jobs : int
            Number of processes for multiprocessing.
        """
        self.reset()
        if raw == 'lp40':
            self.make_filter(raw, hp=None, lp=40, n_jobs=n_jobs, src='clm',
                             redo=redo)
        elif raw == 'hp1-lp40':
            self.make_filter(raw, hp=1, lp=40, n_jobs=n_jobs, src='clm',
                             redo=redo)
        elif raw == 'hp.2-lp40':
            self.make_filter(raw, hp=0.2, lp=40, n_jobs=n_jobs, src='clm',
                             redo=redo, l_trans_bandwidth=0.05,
                             filter_length='20s')
        elif raw == 'hp.1-lp40':
            self.make_filter(raw, hp=0.1, lp=40, n_jobs=n_jobs, src='clm',
                             redo=redo, l_trans_bandwidth=0.05,
                             filter_length='20s')
        else:
            raise ValueError('raw = %r' % raw)

    def ui_select_projs(self, projs, fif_obj, save=True, save_plot=True):
        """
        Plots proj, and asks the user which ones to save.

        Parameters
        ----------
        proj : list
            The projections.
        fif_obj : mne fiff object
            Provides info dictionary for extracting sensor locations.
        save : False | True | tuple
            False: don'r save proj fil; True: manuall pick componentws to
            include in the proj file; tuple: automatically include these
            components
        save_plot : False | str(path)
            target path to save the plot
        """
        picks = mne.epochs.pick_types(fif_obj.info, exclude='bads')
        sensor = load.fiff.sensor_dim(fif_obj, picks=picks)

        # plot PCA components
        PCA = []
        for p in projs:
            d = p['data']['data'][0]
            name = p['desc'][-5:]
            v = ndvar(d, (sensor,), name=name)
            PCA.append(v)

        proj_file = self.get('proj-file')
        p = plot.topo.topomap(PCA, size=1, title=proj_file)
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

    def parse_dirs(self, subjects=[], parse_subjects=True, subject=None):
        """Find subject names by looking through the directory structure.

        Parameters
        ----------
        subjects : list of str
            Subjects to add initially.
        parse_subjects : bool
            Look for subjects as folders in the directory specified in
            :attr:`._subject_loc`.
        subject : str
            Proposed subject state value (if the new value is not found in the
            new subjects, an error is raised and nothing is changed.
        """
        subjects = set(subjects)

        # find subjects
        if parse_subjects:
            pattern = self.subject_re
            sub_dir = self.get(self._subject_loc)
            if os.path.exists(sub_dir):
                for dirname in os.listdir(sub_dir):
                    isdir = os.path.isdir(os.path.join(sub_dir, dirname))
                    if isdir and pattern.match(dirname):
                        subjects.add(dirname)
            else:
                err = ("MEG subjects directory not found: %r. Initialize with "
                       "parse_subjects=False, or specifiy proper directory in "
                       "experiment._subject_loc." % sub_dir)
                raise IOError(err)

        if len(subjects) == 0:
            err = "No subjects found in %r" % sub_dir
            raise IOError(err)
        subjects = sorted(subjects)
        self._field_values['subject'] = subjects

        mrisubjects = [self._mri_subjects[s] for s in subjects]
        common_brain = self.get('common_brain')
        if common_brain:
            mrisubjects.insert(0, common_brain)
        self._field_values['mrisubject'] = mrisubjects

        if subject is None:
            subject = self._state.get('subject', None)
            if subject not in subjects:
                subject = subjects[0]
        elif subject not in subjects:
            err = ("Subject not found: %r" % subject)
            raise ValueError(err)
        self.set(subject, add=True)  # allow setting mrisubject the first time

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
        from ..plot.coreg import dev_mri
        self.set(**kwargs)
        raw = mne.fiff.Raw(self.get('raw-file'))
        return dev_mri(raw)

    def print_tree(self, root='root'):
        """
        Print a tree of the filehierarchy implicit in the templates

        Parameters
        ----------
        root : str
            Name of the root template (e.g., 'besa-root').
        """
        tree = {'.': root}
        root_temp = '{%s}' % root
        for k, v in self._state.iteritems():
            if str(v).startswith(root_temp):
                tree[k] = {'.': v.replace(root_temp, '')}
        _etree_expand(tree, self._state)
        nodes = _etree_node_repr(tree, root)
        name_len = max(len(n) for n, _ in nodes)
        path_len = max(len(p) for _, p in nodes)
        pad = ' ' * (80 - name_len - path_len)
        print os.linesep.join(n.ljust(name_len) + pad + p.ljust(path_len) for n, p in nodes)

    def pull(self, src_root, names=['raw-file', 'log-dir'], **kwargs):
        """OK 12/8/12
        Copies all items matching a template from another root to the current
        root.

        .. warning:: Implemented by creating a new instance of the same class with
            ``src_root`` as root and calling its ``.push()`` method.
            This determines available templates and field_values.

        src_root : str(path)
            root of the source experiment
        names : list of str
            list of template names to copy.
            tested for 'raw-file' and 'log-dir'.
            Should work for any template with an exact match; '*' is not
            implemented and will raise an error.
        **kwargs** :
            see :py:meth:`push`

        """
        subjects = self.get_field_values('subjects')
        e = self.__class__(src_root, subjects=subjects,
                           mri_subjects=self._mri_subjects)
        e.push(self.get('root'), names=names, **kwargs)

    def push(self, dst_root, names=[], overwrite=False, missing='warn'):
        """OK 12/8/12
        Copy certain branches of the directory tree.

        name : str | list of str
            name(s) of the template(s) of the files that should be copied
        overwrite : bool
            What to do if the target file already exists (overwrite it with the
            source file or keep it)
        missing : 'raise' | 'warn' | 'ignor'
            What to do about missing source files(raise an error, print a
            warning, or ignore them)

        """
        assert missing in ['raise', 'warn', 'ignore']

        if isinstance(names, basestring):
            names = [names]

        for name in names:
            for src in self.iter_temp(name):
                if '*' in src:
                    raise NotImplementedError("Can't fnmatch here yet")

                if os.path.exists(src):
                    dst = self.get(name, root=dst_root, match=False, mkdir=True)
                    self.set(root=self.get('root'))
                    if os.path.isdir(src):
                        if os.path.exists(dst):
                            if overwrite:
                                shutil.rmtree(dst)
                                shutil.copytree(src, dst)
                            else:
                                pass
                        else:
                            shutil.copytree(src, dst)
                    elif overwrite or not os.path.exists(dst):
                        shutil.copy(src, dst)
                elif missing == 'warn':
                    print "Skipping (missing): %r" % src
                elif missing == 'raise':
                    raise IOError("Missing: %r" % src)

    def rename(self, old, new):
        """
        Rename a files corresponding to a pattern (or template)

        Parameters
        ----------
        old : str
            Template for the files to be renamed. Can interpret '*', but will
            raise an error in cases where more than one file fit the pattern.
        new : str
            Template for the new names.

        Examples
        --------
        The following command will collect a specific file for each subject and
        place it in a common folder:

        >>> e.rename('{root}/{subject}/info.txt',
                     '/some_other_place/{subject}s_info.txt'
        """
        new = self.expand_template(new)
        files = []
        for old_name in self.iter_temp(old):
            if '*' in old_name:
                matches = glob(old_name)
                if len(matches) == 1:
                    old_name = matches[0]
                elif len(matches) > 1:
                    err = ("Several files fit the pattern %r" % old_name)
                    raise ValueError(err)

            if os.path.exists(old_name):
                new_name = self.format(new)
                files.append((old_name, new_name))

        if not files:
            print "No files found for %r" % old
            return

        old_pf = os.path.commonprefix([pair[0] for pair in files])
        new_pf = os.path.commonprefix([pair[1] for pair in files])
        n_pf_old = len(old_pf)
        n_pf_new = len(new_pf)

        table = fmtxt.Table('lll')
        table.cells('Old', '', 'New')
        table.midrule()
        table.caption("%s -> %s" % (old_pf, new_pf))
        for old, new in files:
            table.cells(old[n_pf_old:], '->', new[n_pf_new:])

        print table

        msg = "Rename %s files (confirm with 'yes')? " % len(files)
        if raw_input(msg) == 'yes':
            for old, new in files:
                dirname = os.path.dirname(new)
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                os.rename(old, new)

    def reset(self, exclude=['subject', 'experiment', 'root']):
        """
        Reset the experiment to the state at the end of __init__.

        Note that variables which were added to the experiment after __init__
        are lost. 'epoch' is always excluded (can not be reset).

        Parameters
        ----------
        exclude : collection
            Exclude these variables from the reset (i.e., retain their current
            value)
        """
        exclude = set(exclude)
        exclude.update(self._secondary_fields)
        # dependent variables
        if 'subject' in exclude:
            exclude.add('mrisubject')

        save = {k:self._state[k] for k in exclude}
        self._state = self._initial_state.copy()
        self._state.update(save)

    def rm(self, temp, exclude={}, values={}, v=False, **constants):
        """
        Remove all files corresponding to a template

        Asks for confirmation before deleting anything. Uses glob, so
        individual templates can be set to '*'.

        Parameters
        ----------
        temp : str
            The template.
        exclude : dict
            Exclude specific values by field.
        values : dict
            Provide specific values by field.
        v : bool
            Verbose mode (print all filename patterns that are searched).
        others :
            Set fields.
        """
        files = []
        for fname in self.iter_temp(temp, exclude=exclude, values=values,
                                    **constants):
            fnames = glob(fname)
            if v:
                print "%s -> %i" % (fname, len(fnames))
            if fnames:
                files.extend(fnames)
            elif os.path.exists(fname):
                files.append(fname)

        if files:
            root = self.get('root')
            print "root: %s\n" % root
            root_len = len(root)
            for name in files:
                if name.startswith(root):
                    print name[root_len:]
                else:
                    print name
            msg = "Delete %i files (confirm with 'yes')? " % len(files)
            if raw_input(msg) == 'yes':
                for path in files:
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
        else:
            print "No files found for %r" % temp

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
        if (subject is None) and (self._state['subject'] is None):
            fif_dir = self.get('meg-sdir')
            subject = None
        else:
            fif_dir = self.get('raw-dir', subject=subject)
            subject = self.get('{mrisubject}')

        subp.run_mne_analyze(fif_dir, subject=subject,
                             subjects_dir=subjects_dir, modal=modal)

    def run_mne_browse_raw(self, subject=None, modal=False):
        if (subject is None) and (self._state['subject'] is None):
            fif_dir = self.get('meg-sdir')
        else:
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
            subprocess.call(cmd, env=env)
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

    def set(self, subject=None, match=True, add=False, **kwargs):
        """
        Set variable values.

        Parameters
        ----------
        subject : str
            Set the `subject` value. The corresponding `mrisubject` is
            automatically set to the corresponding mri subject.
        match : bool
            Require existence of the assigned value (only applies for variables
            for which values are defined as field values). When setting
            root, find subjects by looking through folder structure.
        add : bool
            If the template name does not exist, add a new key. If False
            (default), a non-existent key will raise a KeyError.
        all other : str
            All other keywords can be used to set templates.
        """
        # If not changing root, update subject.
        # If changing root, delay updating subjects to take into account
        # changes in field_value['subject']
        root = kwargs.get('root', None)
        if subject is not None:
            if (root is not None) and self._parse_subjects:
                pass  # set subject with root
            else:
                kwargs['subject'] = subject
                if 'mrisubject' not in kwargs:
                    kwargs['mrisubject'] = self._mri_subjects[subject]

        # extract fields that need special treatment
        epoch = kwargs.pop('epoch', None)
        if epoch:
            epochs = sorted(epoch.split('|'))
            kwargs['epoch'] = epoch = '|'.join(epochs)

            e_descs = []  # full epoch descriptor
            e_descs_nodecim = []  # epoch description without decim
            e_descs_bare = []  # epoch description without decim or rejection
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
                epoch_dicts.append(ep)

                # epoch desc
                desc = self.get_epoch_str(**ep)
                e_descs.append(desc)

                # epoch desc without decim
                ep_nd = ep.copy()
                ep_nd['decim'] = None
                desc_nd = self.get_epoch_str(**ep_nd)
                e_descs_nodecim.append(desc_nd)

                # bare epoch desc
                ep_nd['reject_tmin'] = None
                ep_nd['reject_tmax'] = None
                desc_b = self.get_epoch_str(**ep_nd)
                desc_b = desc_b.format(rej='')
                e_descs_bare.append(desc_b)


            epoch_stim = '|'.join(sorted(stims))
            epoch_desc = '(%s)' % ','.join(sorted(e_descs))
            epoch_nodecim = '(%s)' % ','.join(sorted(e_descs_nodecim))
            epoch_bare = '(%s)' % ','.join(sorted(e_descs_bare))

        rej = kwargs.get('rej', None)
        if rej:
            if rej not in self.epoch_rejection:
                err = ("No settings for rej=%r in self.epoch_rejection" % rej)
                raise ValueError(err)
            rej_args = self.epoch_rejection[rej]

        # special attributes derived from inv
        inv = kwargs.get('inv', None)
        if inv is not None:
            make_kw = {}
            apply_kw = {}
            args = inv.split('-')
            # 1) 'free' | 'fixed' | float
            # 2) depth weighting (optional)
            # 3) regularization 'reg' (optional)
            # 4) snr
            # 5) method
            # 6) pick_normal:  'pick_normal' | nothing
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

        # test fields with entries in field_values
        if match:
            for k, v in kwargs.iteritems():
                if v is None:
                    pass
                elif k not in self._field_values:
                    pass
                elif '*' in v:
                    pass
                elif (k == 'group') and (v in self.get_field_values('subject')):
                    pass
                elif v not in self.get_field_values(k):
                    err = ("Variable {k!r} has no value {v!r}. In order to "
                           "see valid values use e.list_values(); In order to "
                           "set a non-existent value, use e.set({k!s}={v!r}, "
                           "match=False).".format(k=k, v=v))
                    raise ValueError(err)

        # clean model
        model = kwargs.get('model', None)
        if model is not None:
            kwargs['model'] = '%'.join(sorted(model.split('%')))

        # update state ---
        for k, v in kwargs.iteritems():
            if add or k in self._state:
                if v is not None:
                    self._state[k] = v
            else:
                raise KeyError("No variable named %r" % k)

        # set subject after updating root
        if (root is not None) and self._parse_subjects:
            self.parse_dirs(subject=subject)

        # store secondary settings
        if epoch:
            self._state['epoch-stim'] = epoch_stim
            self._state['epoch-desc'] = epoch_desc
            self._state['epoch-nodecim'] = epoch_nodecim
            self._state['epoch-bare'] = epoch_bare
            self._epochs_state = epoch_dicts
        if rej:
            self._rej_args = rej_args
        if inv:
            self._make_inv_kw = make_kw
            self._apply_inv_kw = apply_kw
            self._regularize_inv = regularize_inv

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

    def show_in_finder(self, key, **kwargs):
        "Reveals the file corresponding to the ``key`` template in the Finder."
        fname = self.get(key, **kwargs)
        subprocess.call(["open", "-R", fname])

    def state(self, temp=None, empty=False):
        """
        Examine the state of the experiment.

        Parameters
        ----------
        temp : None | str
            Only show variables relevant to this template.
        empty : bool
            Show empty variables (items whose value is the empty string '').

        Returns
        -------
        state : Table
            Table of (relevant) variables and their values.
        """
        table = fmtxt.Table('lll')
        table.cells('Key', '*', 'Value')
        table.caption('*: Value is modified from initialization state.')
        table.midrule()

        if temp is None:
            keys = (k for k in self._state if not '{' in self._state[k])
        else:
            keys = self.find_keys(temp)

        for k in sorted(keys):
            v = self._state[k]
            if v != self._initial_state[k]:
                mod = '*'
            else:
                mod = ''

            if empty or mod or v:
                table.cells(k, mod, repr(v))

        return table

    def subjects_table(self):
        """Print a table with the MRI subject corresponding to each subject"""
        table = fmtxt.Table('ll')
        table.cells('subject', 'mrisubject')
        table.midrule()
        for _ in self.iter('subject'):
            table.cell(self.get('subject'))
            table.cell(self.get('mrisubject'))
        return table

    def summary(self, templates=['raw-file'], missing='-', link=' > ',
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
