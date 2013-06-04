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
    Value of the stimvar relative to which the epoch is defined.
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


Epochs can be specified directly in the relevant function, or they can be
specified in the :attr:`mne_experiment.epochs` dictionary. All keys in this
dictionary have to be of type :class:`str`, values have to be :class:`dict`s
containing the epoch specification. If an epoch is specified in
:attr:`mne_experiment.epochs`, its name (key) can be used in the epochs
argument to various methods. Example::

    # in mne_experiment subclass definition
    class experiment(mne_experiment):
        epochs = {'adjbl': dict(name='bl', stim='adj', tstart=-0.1, tstop=0)}
        ...

    # use as argument:
    epochs=[dict(name=evoked', stim='noun', tmin=-0.1, tmax=0.5,
                 reject_tmin=0), 'adjbl']

The :meth:`mne_experiment.get_epoch_str` method produces A label for each
epoch specification, which is used for filenames. Data which is excluded from
artifact rejection is parenthesized. For example, ``"noun[(-100)0,500]"``
designates data form -100 to 500 ms relative to the stimulus 'noun', with only
the interval form 0 to 500 ms used for rejection.

'''

from collections import defaultdict
import cPickle as pickle
from glob import glob
import itertools
from operator import add
import os
from Queue import Queue
import re
import shutil
import subprocess
from threading import Thread

import numpy as np

import mne
from mne.minimum_norm import (make_inverse_operator, apply_inverse,
                              apply_inverse_epochs)

from .. import fmtxt
from .. import load
from .. import plot
from .. import save
from .. import ui
from ..utils import keydefaultdict
from ..utils import common_prefix, subp
from ..utils.com import send_email, Notifier
from ..utils.mne_utils import is_fake_mri
from .data import (dataset, factor, var, ndvar, combine, isfactor, align1,
                   DimensionMismatchError, UTS)


__all__ = ['mne_experiment', 'LabelCache']



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
        'meg_dir': os.path.join('{root}', 'meg'),  # contains subject-name folders for MEG data
        'meg_sdir': os.path.join('{meg_dir}', '{subject}'),
        'mri_dir': os.path.join('{root}', 'mri'),  # contains subject-name folders for MRI data
        'mri_sdir': os.path.join('{mri_dir}', '{mrisubject}'),
        'bem_dir': os.path.join('{mri_sdir}', 'bem'),
        'raw_sdir': os.path.join('{meg_sdir}', 'raw'),
        'eeg_sdir': os.path.join('{meg_sdir}', 'raw_eeg'),
        'log_sdir': os.path.join('{meg_sdir}', 'logs', '{subject}_{experiment}'),

        # raw
        'mrk': os.path.join('{raw_sdir}', '{subject}_{experiment}_marker.txt'),
        'elp': os.path.join('{raw_sdir}', '{subject}_HS.elp'),
        'hsp': os.path.join('{raw_sdir}', '{subject}_HS.hsp'),
        'raw': 'raw',
        'raw-base': os.path.join('{raw_sdir}', '{subject}_{experiment}_{raw}'),
        'raw-file': '{raw-base}-raw.fif',
        'raw-txt': os.path.join('{raw_sdir}', '{subject}_{experiment}_*raw.txt'),

        'trans': os.path.join('{raw_sdir}', '{mrisubject}-trans.fif'),  # mne p. 196

        # eye-tracker
        'edf': os.path.join('{log_sdir}', '*.edf'),

        # mne raw-derivatives analysis
        'proj': '',
        'proj-file': '{raw-base}_{proj}-proj.fif',
        'proj_plot': '{raw-base}_{proj}-proj.pdf',
        'cov-file': '{raw-base}_{cov}-{proj}-cov.fif',
        'fwd': '{raw-base}-{mrisubject}_{cov}_{proj}-fwd.fif',
        'cov': 'bl',

        # epochs
        'epoch-sel-file': os.path.join('{meg_sdir}', 'epoch_sel', '{raw}_'
                                       '{experiment}_{epoch}_sel.pickled'),

        # fwd model
        'common_brain': 'fsaverage',
        'fid': os.path.join('{bem_dir}', '{mrisubject}-fiducials.fif'),
        'bem': os.path.join('{bem_dir}', '{mrisubject}-*-bem-sol.fif'),
        'src': os.path.join('{bem_dir}', '{mrisubject}-ico-4-src.fif'),
        'bem_head': os.path.join('{bem_dir}', '{mrisubject}-head.fif'),

        # evoked
        'evoked_dir': os.path.join('{meg_sdir}', 'evoked'),
        'evoked': os.path.join('{evoked_dir}', '{raw}_{experiment}_{model}',
                               '{epoch}_{proj}_evoked.pickled'),

        # Souce space
        'labeldir': 'label',
        'hemi': 'lh',
        'label-file': os.path.join('{mri_sdir}', '{labeldir}', '{hemi}.{label}.label'),
        'morphmap': os.path.join('{mri_dir}', 'morph-maps', '{subject}-{common_brain}-morph.fif'),

        # EEG
        'vhdr': os.path.join('{eeg_sdir}', '{subject}_{experiment}.vhdr'),
        'eegfif': os.path.join('{eeg_sdir}', '{subject}_{experiment}_raw.fif'),
        'eegfilt': os.path.join('{eeg_sdir}', '{subject}_{experiment}_filt_raw.fif'),

        # output files
        'plot_dir': os.path.join('{root}', 'plots'),
        'plot_png': os.path.join('{plot_dir}', '{analysis}', '{name}.png'),
        'res': os.path.join('{root}', 'res_{kind}', '{analysis}', '{name}{suffix}.{ext}'),
        'kind': '',
        'analysis': '',
        'name': '',
        'suffix': '',
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
    .subjects_table()
        Each subject with corresponding MRI subject.
    .summary()
        check for the presence of files for a given templates
    """
    # Experiment Constants
    # --------------------
    # Bad channels dictionary: (sub, exp) -> list of int
    bad_channels = defaultdict(list)
    # Default values for epoch definitions
    epoch_default = {'stimvar': 'stim', 'tmin':-0.1, 'tmax': 0.6, 'decim': 5,
                     'name': 'epochs'}

    # named epochs
    epochs = {'bl': {'stim': 'adj', 'tmin':-0.2, 'tmax':0},
              'epoch': {'stim': 'adj'}}
    # how to reject data epochs. See the  explanation accompanying the values
    # below:
    epoch_rejection = {
                       # Whether to use manual supervision. If True, each epoch
                       # needs a rejection file which can be created using
                       # .make_epoch_selection(). If False, epoch are rejected
                       # automatically.
                       'manual': False,
                       # The sensors to plot separately in the rejection GUI.
                       # The default is the two MEG sensors closest to the eyes
                       # for Abu Dhabi KIT data.
                       'eog_sns': ['MEG 087', 'MEG 130'],
                       # the reject argument when loading epochs:
                       'threshold': dict(mag=3e-12),
                       # How to use eye tracker information:
                       'edf': ['EBLINK']}

    subjects_has_own_mri = ()
    subject_re = re.compile('R\d{4}$')

    # Constants that should not need to be modified
    # ---------------------------------------------
    _experiments = []
    _fmt_pattern = re.compile('\{([\w-]+)\}')
    _mri_loc = 'mri_dir'  # location of subject mri folders
    _repr_vars = ['subject', 'experiment']  # state variables that are shown in self.__repr__()
    _subject_loc = 'meg_dir'  # location of subject folders

    # basic templates to use. Can be a string referring to a templates
    # dictionary in the module level _temp dictionary, or a templates
    # dictionary
    _templates = 'v0'
    # modify certain template entries from the outset (e.g. specify the initial
    # subject name)
    _defaults = {
                 # this should be a key in the epochs class attribute (see
                 # above)
                 'epoch': 'epoch'}

    def __init__(self, root=None, parse_subjects=True, subjects=[],
                 mri_subjects={}):
        """
        Parameters
        ----------
        root : str
            the root directory for the experiment (usually the directory
            containing the 'meg' and 'mri' directories)
        parse_subjects : bool
            Find MEG subjects using :attr:`_subjects_loc`
        parse_mri : bool
            Find MRI subjects using :attr:`_mri_loc`
        subjects : list of str
            Provide additional MEG subjects.
        mri_subjects : dict, {subject: mrisubject}
            Provide additional MRI subjects.

        """
        if root:
            root = os.path.expanduser(root)
            if not os.path.exists(root):
                raise IOError("Path does not exist: %r" % root)
        else:
            msg = "Please select the meg directory of your experiment"
            root = ui.ask_dir("Select Root Directory", msg, True)

        # settings
        self.root = root

        self._log_path = os.path.join(root, 'mne-experiment.pickle')

        self._state = self.get_templates(root=root)
        epoch = self._state.get('epoch', None)
        self.set(epoch=epoch)

        # find experiment data structure
        self._mri_subjects = keydefaultdict(lambda k: k)
        self._mri_subjects.update(mri_subjects)
        self.set(root=root, add=True)
        self.var_values = {'hemi': ('lh', 'rh')}
        self.exclude = {}

        self.parse_dirs(parse_subjects=parse_subjects, subjects=subjects)
        self._update_var_values()

        # set initial values
        self._label_cache = LabelCache()
        for k, v in self.var_values.iteritems():
            if v:
                self._state[k] = v[0]

        # set defaults for any existing templates
        for k in self._state.keys():
            temp = self._state[k]
            for name in self._fmt_pattern.findall(temp):
                if name not in self._state:
                    self._state[name] = ''

        self._initial_state = self._state.copy()

        owner = getattr(self, 'owner', False)
        if owner:
            self.notification = Notifier(owner, 'mne_experiment task')

    def _process_epochs_arg(self, epochs):
        """Fill in named epochs and set the 'epoch' and 'stim' templates"""
        epochs = list(epochs)  # make sure we don't modify the input object
        e_descs = []  # full epoch descriptor
        stims = set()  # all relevant stims
        for i in xrange(len(epochs)):
            # expand epoch description
            ep = epochs[i]
            if isinstance(ep, str):
                ep = self.epochs[ep]
            ep = ep.copy()
            for k, v in self.epoch_default.iteritems():
                if k not in ep:
                    ep[k] = v

            # make sure stim is ordered
            stim = ep['stim']
            if '|' in stim:
                stim = '|'.join(sorted(set(stim.split('|'))))

            # store expanded epoch
            stims.update(stim.split('|'))
            epochs[i] = ep
            desc = self.get_epoch_str(**ep)
            e_descs.append(desc)

        ep_str = '(%s)' % ','.join(sorted(e_descs))
        stim = '|'.join(sorted(stims))

        self.set(stim=stim, add=True)
        self._state['epoch'] = ep_str
        self._epochs_state = epochs

    def _update_var_values(self):
        subjects = self.var_values['subject']
        self.var_values.update(mrisubject=[self._mri_subjects[s]
                                           for s in subjects],
                               experiment=list(self._experiments))

    def add_epochs_stc(self, ds, method='dSPM', ori='free', depth=0.8,
                       reg=False, snr=2., pick_normal=False, src='epochs',
                       dst='stc', asndvar=True):
        """
        Transform epochs contained in ds into source space (adds a list of mne
        SourceEstimates to ds)

        Parameters
        ----------
        ds : dataset
            The dataset containing the mne Epochs for the desired trials.
        method : 'MNE' | 'dSPM' | 'sLORETA'
            MNE method.
        ori : 'free' | ...
            Orientation constraint.
        depth : None | scalar
            Depth weighting factor.
        reg : bool
            Regularize the noise covariance matrix.
        snr : scalar
            Estimated signal to noise ration (used for inverse tranformation).
        pick_normal : bool
            apply_inverse parameter.
        src : str
            Name of the source epochs in ds.
        dst : str
            Name of the source estimates to be created in ds.
        asndvar : bool
            Add the source estimates as ndvar instead of a list of
            SourceEstimate objects.
        """
        subject = ds['subject']
        if len(subject.cells) != 1:
            err = ("ds must have a subject variable with exaclty one subject")
            raise ValueError(err)
        subject = subject.cells[0]

        inv_name = method + '-' + ori
        self.set(inv_name=inv_name, subject=subject)
        lambda2 = 1. / snr ** 2

        epochs = ds[src]
        inv = self.get_inv(epochs, depth=depth, reg=reg)
        stc = apply_inverse_epochs(epochs, inv, lambda2, method,
                                   pick_normal=pick_normal)

        if asndvar:
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

    def add_evoked_stc(self, ds, method='dSPM', ori='free', depth=0.8,
                       reg=False, snr=3., ind='stc', morph='stcm'):
        """
        Add an stc (ndvar) to a dataset with an evoked list.

        Assumes that all Evoked of the same subject share the same inverse
        operator.

        Parameters
        ----------
        ind: False | str
            Keep list of SourceEstimate objects on individual brains with that
            name.
        morph : False | bool
            Add ndvar for data morphed to the common brain with this name.

        """
        if not (ind or morph):
            return

        inv_name = method + '-' + ori
        self.set(inv_name=inv_name)
        lambda2 = 1. / snr ** 2

        # find vars to work on
        do = []
        for name in ds:
            if isinstance(ds[name][0], mne.fiff.Evoked):
                do.append(name)

        invs = {}
        if ind:
            ind = str(ind)
            stcs = defaultdict(list)
        if morph:
            morph = str(morph)
            mstcs = defaultdict(list)

        for case in ds.itercases():
            subject = case['subject']
            if subject in self.subjects_has_own_mri:
                subject_from = subject
            else:
                subject_from = self.get('common_brain')

            for name in do:
                evoked = case[name]

                # get inv
                if subject in invs:
                    inv = invs[subject]
                else:
                    self.set(subject=subject)
                    inv = self.get_inv(evoked, depth=depth, reg=reg)
                    invs[subject] = inv

                stc = apply_inverse(evoked, inv, lambda2, method)
                if ind:
                    stcs[name].append(stc)

                if morph:
                    stc = mne.morph_data(subject_from, self.get('common_brain'), stc, 4)
                    mstcs[name].append(stc)

        for name in do:
            if ind:
                if ind in ds:
                    key = '%s_%s' % (ind, do)
                else:
                    key = ind
                ds[key] = stcs[name]
            if morph:
                if morph in ds:
                    key = '%s_%s' % (morph, do)
                else:
                    key = morph
                ds[key] = load.fiff.stc_ndvar(mstcs[name], self.get('common_brain'))

    def get_templates(self):
        if isinstance(self._templates, str):
            t = _temp[self._templates]
        else:
            t = self._templates

        if not isinstance(t, dict):
            err = ("Templates mus be dictionary; got %s" % type(t))
            raise TypeError(err)

        t.update(self._defaults)
        return t

    def __repr__(self):
        args = [repr(self.root)]
        kwargs = []

        for k in self._repr_vars:
            v = self.get(k)
            kwargs.append((k, repr(v)))

        for k in sorted(self._state):
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

    def combine_labels(self, target, sources=(), redo=False):
        """Combine several freesurfer labels into one label

        Parameters
        ----------
        target : str
            name of the target label.
        sources : sequence of str
            names of the source labels.
        redo : bool
            If the target file already exists, redo the operation and replace
            it.
        """
        tgt = self.get('label-file', label=target)
        if (not redo) and os.path.exists(tgt):
            return

        srcs = (self.load_label(label=name) for name in sources)
        label = reduce(add, srcs)
        label.save(tgt)

    def expand_template(self, temp, values=()):
        """
        Expand a template until all its subtemplates are neither in
        self.var_values nor in ``values``

        Parameters
        ----------
        values : container (implements __contains__)
            values which should not be expanded (in addition to
        """
        temp = self._state.get(temp, temp)

        while True:
            stop = True
            for name in self._fmt_pattern.findall(temp):
                if (name in values) or (self.var_values.get(name, False)):
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

    def get(self, temp, fmatch=True, vmatch=True, match=True, mkdir=False, **kwargs):
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
            in self.var_values)
        match : bool
            Do any matching (i.e., match=False sets fmatch as well as vmatch
            to False).
        mkdir : bool
            If the directory containing the file does not exist, create it.
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

        if tag is not None:
            desc += '|%s' % tag

        return desc + ']'

    def get_inv(self, fiff, depth=0.8, reg=False, **kwargs):
        self.set(**kwargs)

        inv_name = self.get('inv_name')
        method, ori = inv_name.split('-')
        if ori == 'free':
            fwd_kwa = dict(surf_ori=True)
            inv_kwa = dict(loose=None, depth=depth)
        elif ori == 'loose':
            fwd_kwa = dict(surf_ori=True)
            inv_kwa = dict(loose=0.2, depth=depth)
        elif ori == 'fixed':
            fwd_kwa = dict(force_fixed=True)
            inv_kwa = dict(fixed=True, loose=None, depth=depth)
        else:
            raise ValueError('ori=%r' % ori)

        fwd = mne.read_forward_solution(self.get('fwd'), **fwd_kwa)
        cov = mne.read_cov(self.get('cov-file'))
        if reg:
            cov = mne.cov.regularize(cov, fiff.info, mag=reg)
        inv = make_inverse_operator(fiff.info, fwd, cov, **inv_kwa)
        return inv

    def iter_temp(self, temp, constants={}, values={}, exclude={}, prog=False):
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

        for state in self.iter_vars(variables, constants=constants,
                                    values=values, exclude=exclude, prog=prog):
            path = temp.format(**state)
            yield path

    def iter_vars(self, variables=['subject'], constants={}, values={},
                  exclude={}, prog=False, notify=False):
        """
        Cycle the experiment's state through all values on the given variables

        Parameters
        ----------
        variables : list | str
            Variable(s) over which should be iterated.
        constants : dict(name -> value)
            Variables with constant values throughout the iteration.
        values : dict(name -> (list of values))
            Variables with values to iterate over instead of the corresponding
            `mne_experiment.var_values`.
        exclude : dict(name -> (list of values))
            Values to exclude from the iteration.
        prog : bool | str
            Show a progress dialog; str for dialog title.
        """
        if notify is True:
            notify = self.owner

        state_ = self._state.copy()

        # set constants
        self.set(**constants)

        if isinstance(variables, str):
            variables = [variables]
        variables = list(set(variables).difference(constants).union(values))

        # gather possible values to iterate over
        var_values = self.var_values.copy()
        var_values.update(values)

        # exclude values
        for k in exclude:
            var_values[k] = set(var_values[k]).difference(exclude[k])

        # pick out the variables to iterate, but drop excluded cases:
        v_lists = []
        for v in variables:
            values = set(var_values[v]).difference(self.exclude.get(v, ()))
            v_lists.append(sorted(values))

        if len(v_lists):
            if prog:
                i_max = np.prod(map(len, v_lists))
                if not isinstance(prog, str):
                    prog = "MNE Experiment Iterator"
                progm = ui.progress_monitor(i_max, prog, "")
                prog = True

            for v_list in itertools.product(*v_lists):
                values = dict(zip(variables, v_list))
                if prog:
                    progm.message(' | '.join(map(str, v_list)))
                self.set(**values)
                yield self._state
                if prog:
                    progm.advance()
        else:
            yield self._state

        self._state.update(state_)

        if notify:
            send_email(notify, "Eelbrain Task Done", "I did as you desired, "
                       "my master.")

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

    def load_edf(self, **kwargs):
        """Load the edf file ("edf" template)"""
        kwargs['fmatch'] = False
        src = self.get('edf', **kwargs)
        edf = load.eyelink.Edf(src)
        return edf

    def load_epochs(self, epoch=None, asndvar=False, subject=None, reject=True):
        """
        Load a dataset with epochs for a given epoch definition

        Parameters
        ----------
        epoch : dict
            Epoch definition.
        asndvar : bool | str
            Convert epochs to an ndvar with the given name (if True, 'MEG' is
            uesed).
        subject : None | str
            Subject for which to load the data.
        reject : bool
            Whether to apply epoch rejection or not. The kind of rejection
            employed depends on the ``.epoch_rejection`` class attribute.
        """
        self.set(subject=subject, epoch=epoch)
        epoch = self._epochs_state[0]

        stimvar = epoch['stimvar']
        stim = epoch['stim']
        tmin = epoch.get('reject_tmin', epoch['tmin'])
        tmax = epoch.get('reject_tmax', epoch['tmax'])

        ds = self.load_events(subject)
        stimvar = ds[stimvar]
        if '|' in stim:
            idx = stimvar.isin(stim.split('|'))
        else:
            idx = stimvar == stim
        ds = ds.subset(idx)
        reject_arg = None
        if reject:
            if self.epoch_rejection.get('manual', False):
                reject_arg = None
                path = self.get('epoch-sel-file')
                if not os.path.exists(path):
                    err = ("The rejection file at %r does not exist. Run "
                           ".make_epoch_selection() first.")
                    raise RuntimeError(err)
                ds_sel = load.unpickle(path)
                if not np.all(ds['eventID'] == ds_sel['eventID']):
                    err = ("The epoch selection file contains different "
                           "events than the data. Something went wrong...")
                    raise RuntimeError(err)
                ds = ds.subset(ds_sel['accept'])
            else:
                reject_arg = self.epoch_rejection.get('threshold', None)
                use = self.epoch_rejection.get('edf', False)
                if use:
                    edf = ds.info['edf']
                    ds = edf.filter(ds, tstart=tmin, tstop=tmax, use=use)

        # load sensor space data
        target = epoch['name']
        tmin = epoch['tmin']
        tmax = epoch['tmax']
        decim = epoch['decim']
        ds = load.fiff.add_mne_epochs(ds, target=target, tmin=tmin, tmax=tmax,
                                      reject=reject_arg, baseline=None,
                                      decim=decim)
        if asndvar:
            if asndvar is True:
                asndvar = 'MEG'
            else:
                asndvar = str(asndvar)
            ds[asndvar] = load.fiff.epochs_ndvar(ds[target], name=asndvar)

        return ds

    def load_events(self, subject=None, experiment=None, add_proj=True, edf=True):
        """
        Load events from a raw file.

        Loads events from the corresponding raw file, adds the raw to the info
        dict.

        Parameters
        ----------
        subject, experiment, raw : None | str
            Call self.set(...).
        add_proj : bool
            Add the projections to the Raw object. This does *not* set the
            proj variable.
        edf : bool
            Loads edf and add it to the info dict.

        """
        raw = self.load_raw(add_proj=add_proj, subject=subject,
                            experiment=experiment)

        ds = load.fiff.events(raw)

        if subject is None:
            subject = self._state['subject']
        if experiment is None:
            experiment = self._state['experiment']

        # add edf
        if edf:
            edf = self.load_edf()
            edf.add_T_to(ds)
            ds.info['edf'] = edf

        ds = self.label_events(ds, experiment, subject)

        return ds

    def load_evoked(self, model='ref%side', epochs=None, to_ndvar=False):
        """
        Load a dataset with evoked files for each subject.

        Load data previously created with :meth:`mne_experiment.make_evoked`.

        Parameters
        ----------
        to_ndvar : bool | str
            Convert the mne Evoked objects to an ndvar. If True, it is assumed
            the relavent varibale is 'evoked'. If this is not the case, the
            actual name can be supplied as a string instead. The target name
            is always 'MEG'.
        """
        self.set(model=model, epochs=epochs)
        epochs = self._epochs_state

        dss = []
        for _ in self.iter_vars(['subject']):
            fname = self.get('evoked')
            ds = pickle.load(open(fname))
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

        if to_ndvar:
            if to_ndvar is True:
                to_ndvar = 'evoked'
            evoked = ds[to_ndvar]
            ds['MEG'] = load.fiff.evoked_ndvar(evoked)

        return ds

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
                key = (self.get('subject'), self.get('experiment'))
                bad_chs = self.bad_channels[key]
            else:
                bad_chs = add_bads

            raw.info['bads'].extend(bad_chs)

        return raw

    def load_sensor_data(self, epochs=None, random=('subject',),
                         all_subjects=False):
        """
        Load sensor data in the form of an Epochs object contained in a dataset
        """
        if all_subjects:
            dss = (self.load_sensor_data(epochs=epochs, random=random)
                   for _ in self.iter_vars())
            ds = combine(dss)
            return ds

        self.set(epochs=epochs)
        epochs = self._epochs_state
        if len(set(ep['name'] for ep in epochs)) < len(epochs):
            raise ValueError("All epochs need a unique name")

        stim_epochs = defaultdict(list)
        # {(stimvar, stim) -> [name1, ...]}
        kwargs = {}
        e_names = []  # all target epoch names
        for ep in epochs:
            name = ep['name']
            stimvar = ep['stimvar']
            stim = ep['stim']

            e_names.append(name)
            stim_epochs[(stimvar, stim)].append(name)

            kw = dict(reject={'mag': 3e-12}, baseline=None, preload=True)
            for k in ('tmin', 'tmax', 'reject_tmin', 'reject_tmax', 'decim'):
                if k in ep:
                    kw[k] = ep[k]
            kwargs[name] = kw

        # constants
        ds = self.load_events()
        edf = ds.info['edf']

        dss = {}  # {tgt_epochs -> dataset}
        for (stimvar, stim), names in stim_epochs.iteritems():
            eds = ds.subset(ds[stimvar] == stim)
            eds.index()
            for name in names:
                dss[name] = eds

        for name in e_names:
            ds = dss[name]
            kw = kwargs[name]
            tstart = kw.get('reject_tmin', kw['tmin'])
            tstop = kw.get('reject_tmax', kw['tmax'])
            ds = edf.filter(ds, tstart=tstart, tstop=tstop, use=['EBLINK'])
            dss[name] = ds

        idx = reduce(np.intersect1d, (ds['index'].x for ds in dss.values()))

        for name in e_names:
            ds = dss[name]
            ds = align1(ds, idx)
            ds = load.fiff.add_mne_epochs(ds, **kw)
            ds.index()
            dss[name] = ds

        idx = reduce(np.intersect1d, (ds['index'].x for ds in dss.values()))

        for name in e_names:
            ds = dss[name]
            ds = align1(ds, idx)
            dss[name] = ds

        ds = dss.pop(e_names.pop(0))
        for name in e_names:
            eds = dss[name]
            ds[name] = eds[name]

        return ds

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
        epoch = self._epochs_state[0]
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
        """Show the SelectEpochs GUI to do manual epoch selection for a given epoch

        The GUI is opened with the correct file name; if the corresponding
        file exists, it is loaded, and upon saving the correct path is
        the default.

        Parameters
        ----------
        kwargs :
            Kwargs for SelectEpochs
        """
        if not self.epoch_rejection.get('manual', False):
            err = ("Epoch rejection is automatic. See the .epoch_rejection "
                   "class attribute.")
            raise RuntimeError(err)

        ds = self.load_epochs(asndvar=True, reject=False)
        path = self.get('epoch-sel-file', mkdir=True)

        from ..wxgui.MEG import SelectEpochs
        ROI = self.epoch_rejection.get('eog_sns', None)
        gui = SelectEpochs(ds, path=path, ROI=ROI, **kwargs)  # nplots, plotsize,

    def make_evoked(self, model='ref%side', epochs=None, random=('subject',),
                    redo=False):
        """
        Creates datasets with evoked files for the current subject/experiment
        pair.

        Parameters
        ----------
        stimvar : str
            Name of the variable containing the stimulus.
        model : str
            Name of the model. No spaces, order matters.
        epochs : list of epoch specifications
            See the module documentation.

        """
        self.set(epochs=epochs, model=model)
        epochs = self._epochs_state
        dest_fname = self.get('evoked', mkdir=True)
        if not redo and os.path.exists(dest_fname):
            return

        stim_epochs = defaultdict(list)
        kwargs = {}
        e_names = []
        for ep in epochs:
            name = ep['name']
            if name in kwargs:
                raise ValueError("Duplicate epoch name.")
            stimvar = ep['stimvar']
            stim = ep['stim']

            e_names.append(name)
            stim_epochs[(stimvar, stim)].append(name)

            kw = dict(reject={'mag': 3e-12}, baseline=None, preload=True)
            for k in ('tmin', 'tmax', 'reject_tmin', 'reject_tmax', 'decim'):
                if k in ep:
                    kw[k] = ep[k]
            kwargs[name] = kw

        # constants
        sub = self.get('subject')
        model_name = self.get('model')

        ds = self.load_events()
        edf = ds.info['edf']
        if model_name == '':
            model = None
            cells = ((),)
            model_names = []
        else:
            model = ds.eval(model_name)
            cells = model.cells
            if isfactor(model):
                model_names = model.name
            else:
                model_names = model.base_names

        dss = {}
        for (stimvar, stim), names in stim_epochs.iteritems():
            d = ds.subset(ds[stimvar] == stim)
            for name in names:
                dss[name] = d

        evokeds = defaultdict(list)
        factors = defaultdict(list)
        ns = []

        for cell in cells:
            cell_dss = {}
            if model is None:
                for name, ds in dss.iteritems():
                    ds.index()
                    cell_dss[name] = ds
            else:
                n = None
                for name, ds in dss.iteritems():
                    idx = (ds.eval(model_name) == cell)
                    if idx.sum() == 0:
                        break
                    cds = ds.subset(idx)
                    cds.index()
                    cell_dss[name] = cds
                    if n is None:
                        n = cds.n_cases
                    else:
                        if cds.n_cases != n:
                            err = "Can't index due to unequal trial counts"
                            raise RuntimeError(err)

                if len(cell_dss) < len(dss):
                    continue

            for name in e_names:
                ds = cell_dss[name]
                kw = kwargs[name]
                tstart = kw.get('reject_tmin', kw['tmin'])
                tstop = kw.get('reject_tmax', kw['tmax'])
                ds = edf.filter(ds, tstart=tstart, tstop=tstop, use=['EBLINK'])
                cell_dss[name] = ds

            idx = reduce(np.intersect1d, (ds['index'].x for ds in cell_dss.values()))
            if idx.sum() == 0:
                continue

            for name in e_names:
                ds = cell_dss[name]
                ds = align1(ds, idx)
                kw = kwargs[name]
                ds = load.fiff.add_mne_epochs(ds, **kw)
                cell_dss[name] = ds

            idx = reduce(np.intersect1d, (ds['index'].x for ds in cell_dss.values()))
            n = len(idx)
            if n == 0:
                continue

            for name in e_names:
                ds = cell_dss[name]
                ds = align1(ds, idx)
                epochs = ds['epochs']
                evoked = epochs.average()
                assert evoked.nave == n
                evokeds[name].append(evoked)

            # store values
            if isinstance(model_names, str):
                factors[model_names].append(cell)
            else:
                for name, v in zip(model_names, cell):
                    factors[name].append(v)
            factors['subject'].append(sub)
            ns.append(n)

        ds_ev = dataset()
        ds_ev['n'] = var(ns)
        for name, values in factors.iteritems():
            if name in random:
                ds_ev[name] = factor(values, random=True)
            else:
                ds_ev[name] = factor(values)
        for name in e_names:
            ds_ev[name] = evokeds[name]

        save.pickle(ds_ev, dest_fname)

    def make_filter(self, dest='lp40', hp=None, lp=40, n_jobs=3, src='raw',
                    apply_proj=False, redo=False, **kwargs):
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
        fwd = self.get('fwd')

        cmd = ["mne_do_forward_solution",
               '--subject', self.get('mrisubject'),
               '--src', self.get('src'),
               '--bem', self.get('bem'),
               '--mri', self.get('trans'),
               '--meas', self.get('raw-file'),  # provides sensor locations and coordinate transformation between the MEG device coordinates and MEG head-based coordinates.
               '--fwd', fwd,
               '--megonly']
        if redo:
            cmd.append('--overwrite')
        return cmd

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
            dest = self.get('proj_plot')
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

        fname = self.get('plot_png', name='{subject}_{experiment}',
                         analysis='coreg', mkdir=True)
        if not redo and os.path.exists(fname):
            return

        from mayavi import mlab
        p = self.plot_coreg()
        p.save_views(fname, overwrite=True)
        mlab.close()

    def parse_dirs(self, subjects=[], parse_subjects=True):
        """
        find subject names by looking through the directory
        structure.

        """
        subjects = set(subjects)

        # find subjects
        if parse_subjects:
            pattern = self.subject_re
            sub_dir = self.get(self._subject_loc)
            if os.path.exists(sub_dir):
                for fname in os.listdir(sub_dir):
                    isdir = os.path.isdir(os.path.join(sub_dir, fname))
                    if isdir and pattern.match(fname):
                        subjects.add(fname)
            else:
                err = ("MEG subjects directory not found: %r. Initialize with "
                       "parse_subjects=False, or specifiy proper directory in "
                       "experiment._subject_loc." % sub_dir)
                raise IOError(err)

        self.var_values['subject'] = list(subjects)

    def plot_coreg(self, **kwargs):
        self.set(**kwargs)
        raw = mne.fiff.Raw(self.get('raw-file'))
        return plot.coreg.dev_mri(raw)

    def print_tree(self):
        tree = {'.': 'root'}
        for k, v in self._state.iteritems():
            if str(v).startswith('{root}'):
                tree[k] = {'.': v.replace('{root}', '')}
        _etree_expand(tree, self._state)
        nodes = _etree_node_repr(tree, 'root')
        name_len = max(len(n) for n, _ in nodes)
        path_len = max(len(p) for _, p in nodes)
        pad = ' ' * (80 - name_len - path_len)
        print os.linesep.join(n.ljust(name_len) + pad + p.ljust(path_len) for n, p in nodes)

    def pull(self, src_root, names=['raw-file', 'log_sdir'], **kwargs):
        """OK 12/8/12
        Copies all items matching a template from another root to the current
        root.

        .. warning:: Implemented by creating a new instance of the same class with
            ``src_root`` as root and calling its ``.push()`` method.
            This determines available templates and var_values.

        src_root : str(path)
            root of the source experiment
        names : list of str
            list of template names to copy.
            tested for 'raw-file' and 'log_sdir'.
            Should work for any template with an exact match; '*' is not
            implemented and will raise an error.
        **kwargs** :
            see :py:meth:`push`

        """
        subjects = self.var_values['subjects']
        e = self.__class__(src_root, subjects=subjects,
                           mri_subjects=self._mri_subjects)
        e.push(self.root, names=names, **kwargs)

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
                    self.set(root=self.root)
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

        old_pf = common_prefix([pair[0] for pair in files])
        new_pf = common_prefix([pair[1] for pair in files])
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

    def reset(self, exclude=['subject', 'experiment']):
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
        exclude.update(('epoch', 'stim'))
        # dependent variables
        if 'subject' in exclude:
            exclude.add('mrisubject')

        save = {k:self._state[k] for k in exclude}
        self._state = self._initial_state.copy()
        self._state.update(save)

    def rm(self, temp, values={}, exclude={}, v=False, **kwargs):
        """
        Remove all files corresponding to a template

        Asks for confirmation before deleting anything. Uses glob, so
        individual templates can be set to '*'.

        Parameters
        ----------
        temp : str
            The template.
        v : bool
            Verbose mode (print all filename patterns that are searched).
        """
        files = []
        for fname in self.iter_temp(temp, constants=kwargs, values=values,
                                    exclude=exclude):
            fnames = glob(fname)
            if v:
                print "%s -> %i" % (fname, len(fnames))
            if fnames:
                files.extend(fnames)
            elif os.path.exists(fname):
                files.append(fname)

        if files:
            root = self.root
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
        for _ in self.iter_vars(['subject']):
            if self.get('subject') in exclude:
                continue
            mri_sdir = self.get('mri_sdir')
            if os.path.exists(mri_sdir):
                if is_fake_mri(mri_sdir):
                    rmd.append(mri_sdir)
                    sub.append(self.get('subject'))
                    trans = self.get('trans', match=False)
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
        subjects_dir = self.get('mri_dir')
        if (subject is None) and (self._state['subject'] is None):
            fif_dir = self.get('meg_dir')
            subject = None
        else:
            fif_dir = self.get('raw_sdir', subject=subject)
            subject = self.get('{mrisubject}')

        subp.run_mne_analyze(fif_dir, subject=subject,
                             subjects_dir=subjects_dir, modal=modal)

    def run_mne_browse_raw(self, subject=None, modal=False):
        if (subject is None) and (self._state['subject'] is None):
            fif_dir = self.get('meg_dir')
        else:
            fif_dir = self.get('raw_sdir', subject=subject)

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
            The number of workers to create. This parameter is only used the
            first time the method is called.

        Notes
        -----
        The task queue can be inspected in the :attr:`queue` attribute
        """
        if cmd is None:
            return

        if not hasattr(self, 'queue'):
            self.queue = Queue()

            def worker():
                while True:
                    cmd = self.queue.get()
                    subprocess.call(cmd)
                    self.queue.task_done()

            for _ in xrange(workers):
                t = Thread(target=worker)
                t.daemon = True
                t.start()

        self.queue.put(cmd)

    def set(self, subject=None, experiment=None, match=False, add=False,
            **kwargs):
        """
        Set variable values.

        Parameters
        ----------
        subject: str
            Set the `subject` value. The corresponding `mrisubject` is
            automatically set to the corresponding mri subject.
        match : bool
            Require existence of the assigned value (only applies for variables
            in self.var_values)
        add : bool
            If the template name does not exist, add a new key. If False
            (default), a non-existent key will raise a KeyError.
        all other : str
            All other keywords can be used to set templates.

        """
        if experiment is not None:
            kwargs['experiment'] = experiment
        if subject is not None:
            kwargs['subject'] = subject
            if not 'mrisubject' in kwargs:
                kwargs['mrisubject'] = self._mri_subjects[subject]

        # remove epoch(s)
        epoch = kwargs.pop('epoch', None)
        epochs = kwargs.pop('epochs', None)

        # test var_value
        if match:
            for k, v in kwargs.iteritems():
                if ((k in self.var_values) and (not '*' in v)
                     and v not in self.var_values[k]):
                    raise ValueError("Variable %r has not value %r" % (k, v))

        # set state
        for k, v in kwargs.iteritems():
            if add or k in self._state:
                if v is not None:
                    self._state[k] = v
            else:
                raise KeyError("No variable named %r" % k)

        if epoch and epochs:
            err = "Can's set 'epoch' and 'epochs' at the same time"
            raise RuntimeError(err)
        elif epoch:
            self._process_epochs_arg([epoch])
        elif epochs:
            self._process_epochs_arg(epochs)

    _cell_order = ()
    _cell_fullname = True  # whether or not to include factor name in cell

    def set_cell(self, cell):
        """
        cell : dict
            a {factor: cell} dictionary. Uses self._cell_order to determine te
            order of factors.

        """
        parts = []
        for k in cell:
            if k not in self._cell_order:
                err = ("Cell name not specified in self._cell_order: "
                       "%r" % k)
                raise ValueError(err)

        for f in self._cell_order:
            if f in cell:
                if self._cell_fullname:
                    parts.append('%s=%s' % (f, cell[f]))
                else:
                    parts.append(cell[f])
        name = '-'.join(parts)
        self.set(cell=name, add=True)

    def set_env(self):
        """
        Set environment variables for free for freesurfer etc.

        """
        os.environ['SUBJECTS_DIR'] = self.get('mri_dir')

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
        self._update_var_values()

    def show_in_finder(self, key, **kwargs):
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
        for _ in self.iter_vars('subject'):
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
        for _ in self.iter_vars(['subject', 'experiment']):
            items = []
            sub = self.get('subject')
            exp = self.get('experiment')
            mri_subject = self.get('mrisubject')
            if sub not in mris:
                if mri_subject == sub:
                    mri_dir = self.get('mri_sdir')
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
