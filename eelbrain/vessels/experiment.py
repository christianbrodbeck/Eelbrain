'''
mne_experiment is a base class for managing an mne experiment.


Epochs
------

Epochs are defined as dictionaries containing the following entries
(**mandatory**/optional):

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
import fnmatch
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
from mne.minimum_norm import make_inverse_operator

from .. import fmtxt
from .. import load
from .. import plot
from .. import save
from .. import ui
from ..utils import subp
from ..utils.print_funcs import printlist
from ..utils.kit import split_label
from .data import (dataset, factor, var, ndvar, combine, isfactor, align1,
                   DimensionMismatchError)


__all__ = ['mne_experiment', 'LabelCache']



_kit2fiff_args = {'sfreq':1000, 'lowpass':100, 'highpass':0,
                  'stimthresh':2.5, 'stim':xrange(168, 160, -1)}


class LabelCache(dict):
    def __getitem__(self, path):
        if path in self:
            return super(LabelCache, self).__getitem__(path)
        else:
            label = mne.read_label(path)
            self[path] = label
            return label


class Labels(object):
    _acro = {}
    def __init__(self, lbl_dir):
        for lbl in os.listdir(lbl_dir):
            name, ext = os.path.splitext(lbl)
            if ext == '.label':
                name = name.replace('-', '_')
                path = os.path.join(lbl_dir, lbl)
                setattr(self, name, path)


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



class mne_experiment(object):
    """

    Option Attributes
    -----------------

    auto_launch_mne : bool | None
        If the requested file does not exist, open mne so the user can
        create it. With
        ``None``, the application asks each time.
        Currently affects only "trans" files.

    """
    auto_launch_mne = False
    bad_channels = defaultdict(list)  # (sub, exp) -> list
    epochs = {}
    subjects_has_own_mri = ()
    subject_re = re.compile('R\d{4}$')
    # the default value for the common_brain (can be overridden using the set
    # method after __init__()):
    _common_brain = 'fsaverage'
    _experiments = []
    _fmt_pattern = re.compile('\{(\w+)\}')
    _mri_loc = 'mri_dir'  # location of subject mri folders
    _repr_vars = ['subject', 'experiment']  # state variables that are shown in self.__repr__()
    _subject_loc = 'meg_dir'  # location of subject folders
    def __init__(self, root=None, parse_subjects=True, parse_mri=True,
                 subjects=[], mri_subjects={},
                 kit2fiff_args=_kit2fiff_args):
        """
        root : str
            the root directory for the experiment (i.e., the directory
            containing the 'meg' and 'mri' directories)

        fwd : None | dict
            dictionary specifying the forward model parameters

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
        self._kit2fiff_args = kit2fiff_args

        self._log_path = os.path.join(root, 'mne-experiment.pickle')

        # find experiment data structure
        self.state = self.get_templates()
        self.add_to_state(common_brain=self._common_brain)
        self.var_values = {'hemi': ('lh', 'rh')}
        self.exclude = {}

        self.add_to_state(root=root, raw='{raw_raw}', labeldir='label', hemi='lh')
        self.parse_dirs(parse_subjects=parse_subjects, parse_mri=parse_mri,
                        subjects=subjects, mri_subjects=mri_subjects)

        # set initial values
        self._label_cache = LabelCache()
        for k, v in self.var_values.iteritems():
            if v:
                self.state[k] = v[0]

        # set defaults for any existing templates
        for k in self.state.keys():
            temp = self.state[k]
            for name in self._fmt_pattern.findall(temp):
                if name not in self.state:
                    self.state[name] = '<%s not set>' % name

    def _process_epochs_arg(self, epochs):
        "Fill in named epochs and set the 'epoch' template"
        epochs = list(epochs)
        e_descs = []  # full epoch descriptor
        for i in xrange(len(epochs)):
            ep = epochs[i]
            if isinstance(ep, str):
                ep = self.epochs[ep]
                epochs[i] = ep
            desc = self.get_epoch_str(**ep)
            e_descs.append(desc)

        ep_str = '(%s)' % ','.join(sorted(e_descs))
        self.set(epoch=ep_str)
        return epochs

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

        time = var(stc.times, name='time')
        ds[key] = ndvar(np.array(x), dims=('case', time))

    def add_evoked_stc(self, ds, method='sLORETA', ori='free', ind=True,
                       morph=True, names={'evoked': 'stc'}):
        """
        Add an stc (ndvar) to a dataset with an evoked list.

        Assumes that all Evoked of the same subject share the same inverse
        operator.

        Parameters
        ----------
        ind: bool
            Keep list of SourceEstimate objects on individual brains.
        morph : bool
            Add ndvar for data morphed to the common brain.

        """
        if not (ind or morph):
            return

        inv_name = method + '-' + ori
        self.set(inv_name=inv_name)

        # find vars to work on
        do = []
        for name in ds:
            if isinstance(ds[name][0], mne.fiff.Evoked):
                do.append(name)

        invs = {}
        if ind:
            stcs = defaultdict(list)
        if morph:
            mstcs = defaultdict(list)

        for case in ds.itercases():
            subject = case['subject']
            if subject in self.subjects_has_own_mri:
                subject_from = subject
            else:
                subject_from = self._common_brain

            for name in do:
                evoked = case[name]

                # get inv
                if subject in invs:
                    inv = invs[subject]
                else:
                    self.set(subject=subject)
                    inv = self.get_inv(evoked, depth=0.8)
                    invs[subject] = inv

                stc = mne.minimum_norm.apply_inverse(evoked, inv, 1 / 9., method)
                if ind:
                    stcs[name].append(stc)

                if morph:
                    stc = mne.morph_data(subject_from, self._common_brain, stc, 4)
                    mstcs[name].append(stc)

        for name in do:
            if name in names:
                s_name = names[name]
                m_name = s_name + 'm'
            else:
                i = 0
                while name + 's' + '_' * i in ds:
                    i += 1
                s_name = name + 's' + '_' * i
                im = 0
                while s_name + 'm' + '_' * im in ds:
                    im += 1
                m_name = s_name + 'm' + '_' * im

            if ind:
                ds[s_name] = stcs[name]
            if morph:
                ds[m_name] = load.fiff.stc_ndvar(mstcs[name], self._common_brain)

    def add_to_state(self, **kv):
        for k, v in kv.iteritems():
            if k in self.state:
                raise KeyError("%r already in state" % k)
            self.state[k] = v

    def get_templates(self):
        # rub: need to distinguish files from
        t = {
             # basic dir
             'meg_dir': os.path.join('{root}', 'meg'),  # contains subject-name folders for MEG data
             'meg_sdir': os.path.join('{meg_dir}', '{subject}'),
             'mri_dir': os.path.join('{root}', 'mri'),  # contains subject-name folders for MRI data
             'mri_sdir': os.path.join('{mri_dir}', '{mrisubject}'),
             'raw_sdir': os.path.join('{meg_sdir}', 'raw'),
             'eeg_sdir': os.path.join('{meg_sdir}', 'raw_eeg'),
             'log_sdir': os.path.join('{meg_sdir}', 'logs', '{subject}_{experiment}'),

             # raw
             'mrk': os.path.join('{raw_sdir}', '{subject}_{experiment}_marker.txt'),
             'elp': os.path.join('{raw_sdir}', '{subject}_HS.elp'),
             'hsp': os.path.join('{raw_sdir}', '{subject}_HS.hsp'),
             'rawtxt': os.path.join('{raw_sdir}', '{subject}_{experiment}_*raw.txt'),
             'raw_raw': os.path.join('{raw_sdir}', '{subject}_{experiment}'),
             'rawfif': '{raw}_raw.fif',  # for subp.kit2fiff
             'trans': os.path.join('{raw_sdir}', '{mrisubject}-trans.fif'),  # mne p. 196

             # eye-tracker
             'edf': os.path.join('{log_sdir}', '*.edf'),

             # mne raw-derivatives analysis
             'proj': '',
             'proj_file': '{raw}_{proj}-proj.fif',
             'proj_plot': '{raw}_{proj}-proj.pdf',
             'cov': '{raw}_{cov_name}-{proj}-cov.fif',
             'fwd': '{raw}_{cov_name}-{proj}-fwd.fif',

             # fwd model
             'fid': os.path.join('{mri_sdir}', 'bem', '{mrisubject}-fiducials.fif'),
             'bem': os.path.join('{mri_sdir}', 'bem', '{mrisubject}-5120-bem-sol.fif'),
             'src': os.path.join('{mri_sdir}', 'bem', '{mrisubject}-ico-4-src.fif'),
             'bem_head': os.path.join('{mri_sdir}', 'bem', '{mrisubject}-head.fif'),

             # mne's stc.save() requires stub filename and will add '-?h.stc'
             'evoked_dir': os.path.join('{meg_sdir}', 'evoked_{experiment}_{model}'),
             'evoked': os.path.join('{evoked_dir}', '{epoch}_{proj}-evoked.pickled'),
             'stc_dir': os.path.join('{meg_sdir}', 'stc_{cov_name}-{proj}_{inv_name}'),
             'stc': os.path.join('{stc_dir}', '{experiment}_{cell}_{epoch}'),
             'stc_morphed': os.path.join('{stc_dir}', '{experiment}_{cell}_{common_brain}'),
             'label_file': os.path.join('{mri_sdir}', '{labeldir}', '{hemi}.{label}.label'),
             'morphmap': os.path.join('{mri_dir}', 'morph-maps', '{subject}-{common_brain}-morph.fif'),

             # EEG
             'vhdr': os.path.join('{eeg_sdir}', '{subject}_{experiment}.vhdr'),
             'eegfif': os.path.join('{eeg_sdir}', '{subject}_{experiment}_raw.fif'),
             'eegfilt': os.path.join('{eeg_sdir}', '{subject}_{experiment}_filt_raw.fif'),

             # output files
             'plot_dir': os.path.join('{root}', 'plots'),
             'plot_png': os.path.join('{plot_dir}', '{analysis}', '{name}.png'),

             # BESA
             'besa_triggers': os.path.join('{meg_sdir}', 'besa', '{subject}_{experiment}_{analysis}_triggers.txt'),
             'besa_evt': os.path.join('{meg_sdir}', 'besa', '{subject}_{experiment}_{analysis}.evt'),
             }

        return t

    def __repr__(self):
        args = [repr(self.root)]
        kwargs = []

        for k in self._repr_vars:
            v = self.get(k)
            kwargs.append((k, repr(v)))

        args.extend('='.join(pair) for pair in kwargs)
        args = ', '.join(args)
        return "mne_experiment(%s)" % args

    def combine_labels(self, target, sources=[], redo=False):
        """
        target : str
            name of the target label.
        sources : list of str
            names of the source labels.

        """
        tgt = self.get('label_file', label=target)
        if (not redo) and os.path.exists(tgt):
            return

        srcs = (self.load_label(label=name) for name in sources)
        label = reduce(add, srcs)
        label.save(tgt)

    def do_kit2fiff(self, do='ask', aligntol=xrange(5, 40, 5), redo=False):
        """OK 12/7/2
        find any raw txt files that have not been converted

        do : bool | 'ask',
            whether to automatically convert raw txt files

        **assumes:**

         - all files in the subjects' raw folder
         - filename of schema "<s>_<e>_raw.txt"

        """
        assert do in [True, False, 'ask']

        raw_txt = []
        for _ in self.iter_vars(['subject']):
            subject = self.get('subject')
            temp = self.get('rawtxt', experiment='*', match=False)
            tdir, tname = os.path.split(temp)
            fnames = fnmatch.filter(os.listdir(tdir), tname)
            for fname in fnames:
                fs, fexp, _ = fname.split('_', 2)
                fifpath = self.get('rawfif', raw='{raw_raw}', subject=fs, experiment=fexp, match=False)
                if redo or not os.path.exists(fifpath):
                    raw_txt.append((subject, fexp, fname))

        if len(raw_txt) == 0:
            print "No files found for conversion"
            return

        table = fmtxt.Table('lll')
        table.cells("subject", "experiment", 'file')
        for line in raw_txt:
            table.cells(*line)

        print table
        if do == 'ask':
            do = raw_input('convert missing (y)?') in ['y', 'Y', '\n']

        if do:
            aligntols = {}
            failed = []
            prog = ui.progress_monitor(len(raw_txt), "kit2fiff", "")
            for subject, experiment, fname in raw_txt:
                prog.message(subject + ' ' + experiment)
                self.set(subject=subject, experiment=experiment)
                key = '_'.join((subject, experiment))
                for at in aligntol:
                    try:
                        subp.kit2fiff(self, aligntol=at, overwrite=redo,
                                      **self._kit2fiff_args)
                        aligntols[key] = at
                    except RuntimeError:
                        if at < max(aligntol):
                            pass
                        else:
                            failed.append(fname)
                    else:
                        break
                prog.advance()

            print aligntols

            if len(failed) > 0:
                table = fmtxt.Table('l')
                table.cell("Failed")
                table.cells(*failed)
                print table
        else:
            return raw_txt

    def format(self, temp, vmatch=True, **kwargs):
        """
        Returns the template temp formatted with current values. Formatting
        retrieves values from self.state and self.templates iteratively

        TODO: finish

        """
        self.set(match=vmatch, **kwargs)

        while True:
            variables = self._fmt_pattern.findall(temp)
            if variables:
                temp = temp.format(**self.state)
            else:
                break

        path = os.path.expanduser(temp)
        return path

    def get(self, temp, fmatch=True, vmatch=True, match=True, mkdir=False, **kwargs):
        """
        Retrieve a formatted path by template name.
        With match=True, '*' are expanded to match a file,
        and if there is not a unique match, an error is raised. With
        mkdir=True, the directory containing the file is created if it does not
        exist.

        name : str
            name (code) of the requested file
        subject : None | str
            (MEG) subject for which to retrieve the path (if None, the current
            subject is used)
        experiment : None | str
            experiment for which to retrieve the path (if None, the current
            experiment is used)
        analysis : str
            ... (currently unused)
        match : bool
            require that the file exists. If the path cotains '*', the path is
            extended to the actual file. If not file is found, an IOError is
            raised.
        mkdir : bool
            if the directory containing the file does not exist, create it

        """
        if not match:
            fmatch = vmatch = False

        path = self.format('{%s}' % temp, vmatch=vmatch, **kwargs)

        # assert the presence of the file
        directory, fname = os.path.split(path)
        if fmatch and ('*' in fname):
            if not os.path.exists(directory):
                err = ("Directory does not exist: %r" % directory)
                raise IOError(err)

            match = fnmatch.filter(os.listdir(directory), fname)
            if len(match) == 1:
                path = os.path.join(directory, match[0])
            elif len(match) > 1:
                err = "More than one files match %r: %r" % (path, match)
                raise IOError(err)
            else:
                raise IOError("No file found for %r" % path)
        elif mkdir and not os.path.exists(directory):
            os.makedirs(directory)

        # special cases that can create the file in question
        if match and temp == 'trans':
            if not os.path.exists(path):
                if self.auto_launch_mne is None:
                    a = ui.ask("Launch mne_analyze for Coordinate-Coregistration?",
                               "The 'trans' file for %r, %r does not exist. Should "
                               "mne_analyzed be launched to create it?" %
                               (self.state['subject'], self.state['experiment']),
                               cancel=False, default=True)
                else:
                    a = bool(self.auto_launch_mne)
                if a:
                    # take snapshot of files in raw_sdir
                    raw_sdir = self.get('raw_sdir')
                    flist = os.listdir(raw_sdir)

                    # allow the user to create the file
                    ui.show_help(subp.run_mne_analyze)
                    print "Opening mne_analyze for generating %r" % path
                    subp.run_mne_analyze(self.get('mri_dir'),
                                         raw_sdir,
                                         mri_subject=self.get('mrisubject'),
                                         modal=True)

                    # rename the file if possible
                    newf = set(os.listdir(raw_sdir)).difference(flist)
                    newf = filter(lambda x: str.endswith(x, '-trans.fif'), newf)
                    if len(newf) == 1:
                        src = os.path.join(raw_sdir, newf[0])
                        os.rename(src, path)

                    if not os.path.exists(path):
                        err = ("Error creating file; %r does not exist" % path)
                        raise IOError(err)
                else:
                    err = ("No trans file for %r, %r" %
                           (self.state['subject'], self.state['experiment']))
                    raise IOError(err)

        return path

    def get_epoch_str(self, stim=None, tmin=None, tmax=None, reject_tmin=None,
                      reject_tmax=None, name=None):
        "Produces a descriptor for a single epoch specification"
        desc = '%s[' % stim
        if reject_tmin is None:
            desc += '%i,' % (tmin * 1000)
        else:
            desc += '(%i)%i,' % (tmin * 1000, reject_tmin * 1000)
        if reject_tmax is None:
            desc += '%i]' % (tmax * 1000)
        else:
            desc += '(%i)%i]' % (tmax * 1000, reject_tmax * 1000)
        return desc

    def get_inv(self, fiff, depth=0.8, **kwargs):
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
        cov = mne.read_cov(self.get('cov'))
        inv = make_inverse_operator(fiff.info, fwd, cov, **inv_kwa)
        return inv

    def expand_template(self, temp, values={}):
        """
        Expands a template so far as subtemplates are neither in
        self.var_values nor in the collection provided through the ``values``
        kwarg

        values : container (implements __contains__)
            values which should not be expanded (in addition to
        """
        temp = self.state.get(temp, temp)

        while True:
            stop = True
            for var in self._fmt_pattern.findall(temp):
                if (var in values) or (var in self.var_values):
                    pass
                else:
                    temp = temp.replace('{%s}' % var, self.state[var])
                    stop = False

            if stop:
                break

        return temp

    def iter_temp(self, temp, constants={}, values={}, exclude={}, prog=False):
        """
        Iterate through all paths conforming to a template given in ``temp``.

        temp : str
            Name of a template in the mne_experiment.templates dictionary, or
            a path template with variables indicated as in ``'{var_name}'``

        """
        # if the name is an existing template, retrieve it
        temp = self.expand_template(temp, values=values)

        # find variables for iteration
        variables = set(self._fmt_pattern.findall(temp))

        for state in self.iter_vars(variables, constants=constants,
                                    values=values, exclude=exclude, prog=prog):
            path = temp.format(**state)
            yield path

    def iter_vars(self, variables=['subject'], constants={}, values={},
                  exclude={}, prog=False):
        """
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
        state_ = self.state.copy()

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
            values = var_values[v]
            for exc in self.exclude.get(v, ()):
                if exc in values:
                    values.remove(exc)
            v_lists.append(values)

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
                yield self.state
                if prog:
                    progm.advance()
        else:
            yield self.state

        self.state.update(state_)

    def label_events(self, ds, experiment, subject):
        ds['T'] = ds.eval('i_start / 1000')
        ds['SOA'] = var(np.ediff1d(ds['T'].x, 0))
        return ds

    def load_edf(self, subject=None, experiment=None):
        src = self.get('edf', subject=subject, experiment=experiment)
        edf = load.eyelink.Edf(src)
        return edf

    def load_events(self, subject=None, experiment=None, edf=True, raw=None):
        """
        Load events from a raw file.

        Loads events from the corresponding raw file, adds the raw to the info
        dict.

        Parameters
        ----------
        subject, experiment, raw : None | str
            Call self.set(...).
        edf : bool
            Loads edf and add it to the info dict.

        """
        self.set(subject=subject, experiment=experiment, raw=raw)
        raw_file = self.get('rawfif')
        proj = self.get('proj')
        if proj:
            proj = self.get('proj_file')
        else:
            proj = None
        ds = load.fiff.events(raw_file, proj=proj)

        raw = ds.info['raw']
        bad_chs = self.bad_channels[(self.state['subject'], self.state['experiment'])]
        raw.info['bads'].extend(bad_chs)

        if subject is None:
            subject = self.state['subject']
        if experiment is None:
            experiment = self.state['experiment']

        self.label_events(ds, experiment, subject)

        # add edf
        if edf:
            edf = self.load_edf()
            edf.add_T_to(ds)
            ds.info['edf'] = edf

        return ds

    def load_evoked(self, stimvar='stim', model='ref%side',
                    epochs=[dict(name='evoked', stim='adj', tmin= -0.1, tmax=0.6)]):
        """
        Load as dataset data created with :meth:`mne_experiment.make_evoked`.
        """
        epochs = self._process_epochs_arg(epochs)
        self.set(model=model)

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

        return ds

    def load_label(self, **kwargs):
        self.set(**kwargs)
        fname = self.get('label_file')
        return self._label_cache[fname]

    def make_evoked(self, stimvar='stim', model='ref%side',
                    epochs=[dict(name='evoked', stim='adj', tmin= -0.1, tmax=0.6)],
                    decim=10, random=('subject',), redo=False):
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
        epochs = self._process_epochs_arg(epochs)
        self.set(model=model)

        stim_epochs = defaultdict(list)
        kwargs = {}
        e_names = []
        for ep in epochs:
            name = ep['name']
            stim = ep['stim']

            e_names.append(name)
            stim_epochs[stim].append(name)

            kw = dict(reject={'mag': 3e-12}, baseline=None, decim=decim,
                      preload=True)
            for k in ('tmin', 'tmax', 'reject_tmin', 'reject_tmax'):
                if k in ep:
                    kw[k] = ep[k]
            kwargs[name] = kw

        fname = self.get('evoked', mkdir=True)
        if not redo and os.path.exists(fname):
            return

        # constants
        sub = self.get('subject')
        proj = self.get('proj')
        model_name = self.get('model')

        ds = self.load_events(proj=proj)
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
        for stim, names in stim_epochs.iteritems():
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

        save.pickle(ds_ev, fname)

    def make_fake_mris(self, subject=None, exclude=None, **kwargs):
        """
        ..warning::
            This is a quick-and-dirty method. Coregistration should be
            supervised using plot.coreg.mri_head_viewer.

        """
        self.set(**kwargs)
        self.set_env()
        kwargs = {}
        if subject:
            if isinstance(subject, str):
                subject = [subject]
            kwargs['values'] = {'subject': subject}
        if exclude:
            if isinstance(exclude, str):
                exclude = [exclude]
            kwargs['exclude'] = {'subjects': exclude}

        for _ in self.iter_vars(['subject'], **kwargs):
            s_to = self.get('subject')
            mri_sdir = self.get('mri_sdir', mrisubject=s_to, match=False)
            if os.path.exists(mri_sdir):
                continue

            s_from = self.get('common_brain')
            raw = self.get('rawfif')
            p = plot.coreg.mri_head_fitter(s_from, raw, s_to)
            p.fit()
            p.save()
            self.set_mri_subject(s_to, s_to)

    def make_proj_for_epochs(self, epochs, projname='ironcross', n_mag=5,
                             save=True, save_plot=True):
        """
        computes the first ``n_mag`` PCA components, plots them, and asks for
        user input (a tuple) on which ones to save.

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
        proj = mne.compute_proj_epochs(epochs, n_grad=0, n_mag=n_mag, n_eeg=0)

        picks = mne.epochs.pick_types(epochs.info)
        sensor = load.fiff.sensor_net(epochs, picks=picks)

        # plot PCA components
        PCA = []
        for p in proj:
            d = p['data']['data'][0]
            name = p['desc'][-5:]
            v = ndvar(d, (sensor,), name=name)
            PCA.append(v)

        p = plot.topo.topomap(PCA, size=1, title=str(epochs.name))
        if save_plot:
            dest = self.get('proj_plot', proj=projname)
            p.figure.savefig(dest)
        if save:
            rm = save
            while not isinstance(rm, tuple):
                rm = input("which components to remove? (tuple / 'x'): ")
                if rm == 'x': raise
            p.close()
            proj = [proj[i] for i in rm]
            dest = self.get('proj_file', proj=projname)
            mne.write_proj(dest, proj)

    def makeplt_coreg(self, variables=['subject', 'experiment'], constants={},
                      values={}):
        """
        Save coregistration plots

        """
        from mayavi import mlab
        self.set(analysis='coreg')
        for _ in self.iter_vars(variables, constants=constants, values=values):
            p = self.plot_coreg()
            fname = self.get('plot_png', name='{subject}_{experiment}', mkdir=True)
            p.save_views(fname)
            mlab.close()

    def parse_dirs(self, subjects=[], mri_subjects={}, parse_subjects=True,
                   parse_mri=True):
        """
        find subject names by looking through the directory
        structure.

        """
        subjects = set(subjects)
        self._mri_subjects = mri_subjects = dict(mri_subjects)

        # find subjects
        if parse_subjects:
            pattern = self.subject_re
            sub_dir = self.get(self._subject_loc)
            if os.path.exists(sub_dir):
                for fname in os.listdir(sub_dir):
                    isdir = os.path.isdir(os.path.join(sub_dir, fname))
                    if isdir and pattern.match(fname):
                        subjects.add(fname)


        # find MRIs
        if parse_mri:
            mri_dir = self.get(self._mri_loc)
            if os.path.exists(mri_dir):
                mris = os.listdir(mri_dir)
                for s in subjects:
                    if s in mri_subjects:
                        continue
                    elif s in mris:
                        mri_subjects[s] = s
                    else:
                        mri_subjects[s] = '{common_brain}'

        self.var_values['subject'] = list(subjects)
        self.var_values['mrisubject'] = list(mri_subjects.values())
        self.var_values['experiment'] = list(self._experiments)
        has_mri = (s for s in subjects if mri_subjects[s] != '{common_brain}')
        self.subjects_has_mri = tuple(has_mri)

    def plot_coreg(self, **kwargs):  # sens=True, mrk=True, fiduc=True, hs=False, hs_mri=True,
        self.set(**kwargs)
        raw = mne.fiff.Raw(self.get('rawfif'))
        return plot.coreg.dev_mri(raw)

    def plot_mrk(self, **kwargs):
        self.set(**kwargs)
        fname = self.get('mrk')
        mf = load.kit.marker_avg_file(fname)
        ax = mf.plot_mpl()
        return ax

    def plot_mrk_fix(self, **kwargs):
        self.set(**kwargs)
        mrk = self.get('mrk')
        raw = self.get('rawfif')
        fig = plot.sensors.mrk_fix(mrk, raw)
        return fig

    def print_tree(self):
        tree = {'.': 'root'}
        for k, v in self.state.iteritems():
            if str(v).startswith('{root}'):
                tree[k] = {'.': v.replace('{root}', '')}
        _etree_expand(tree, self.state)
        nodes = _etree_node_repr(tree, 'root')
        name_len = max(len(n) for n, _ in nodes)
        path_len = max(len(p) for _, p in nodes)
        pad = ' ' * (80 - name_len - path_len)
        print os.linesep.join(n.ljust(name_len) + pad + p.ljust(path_len) for n, p in nodes)

    def pull(self, src_root, names=['rawfif', 'log_sdir'], **kwargs):
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
            tested for 'rawfif' and 'log_sdir'.
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

    def rename(self, old, new, constants={}, v=True, do=True):
        """
        Parameters
        ----------

        old, new : str
            Template names (i.e., the corresponding template needs to be
            present in e.state.

        v : bool
            Verbose mode

        do : bool
            Do actually perform the renaming (use ``v=True, do=False`` to
            check the result without actually performing the operation)

        """
        for old_name in self.iter_temp(old, constants=constants):
            if os.path.exists(old_name):
                new_name = self.get(new)
                if do:
                    os.rename(old_name, new_name)
                if v:
                    print "%r\n  ->%r" % (old_name, new_name)

    def rm(self, temp, constants={}, values={}, exclude={}, **kwargs):
        self.set(**kwargs)
        files = []
        for temp in self.iter_temp(temp, constants=constants, values=values,
                                   exclude=exclude):
            if os.path.exists(temp):
                files.append(temp)
        if files:
            printlist(files)
            if raw_input("Delete (confirm with 'yes')? ") == 'yes':
                for path in files:
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
        else:
            print "No files found for %r" % temp

    def rm_fake_mris(self, confirm=False):
        """
        Remove all fake MRIs and trans files

        """
        rmd = []  # dirs
        rmf = []  # files
        sub = []
        for _ in self.iter_vars(['subject']):
            mri_sdir = self.get('mri_sdir')
            if os.path.exists(mri_sdir):
                if plot.coreg.is_fake_mri(mri_sdir):
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
                self.set_mri_subject(s, self._common_brain)

    def run_mne_analyze(self, subject=None, modal=False):
        mri_dir = self.get('mri_dir')
        if (subject is None) and (self.state['subject'] is None):
            fif_dir = self.get('meg_dir')
            mri_subject = None
        else:
            fif_dir = self.get('raw_sdir', subject=subject)
            mri_subject = self.get('{mrisubject}')

        subp.run_mne_analyze(mri_dir, fif_dir, mri_subject=mri_subject,
                             modal=modal)

    def run_mne_browse_raw(self, subject=None, modal=False):
        if (subject is None) and (self.state['subject'] is None):
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
        Set the state of the experiment.

        Parameters
        ----------
        subjects:
            The corresponding `mrisubject` is also set
        match : bool
            require existence (for variables in self.var_values)
        add : bool
            If the template name does not exist, add a new key.

        """
        if experiment is not None:
            kwargs['experiment'] = experiment
        if subject is not None:
            kwargs['subject'] = subject
            kwargs['mrisubject'] = self._mri_subjects.get(subject, None)

        # test var_value
        for k, v in kwargs.iteritems():
            if match and (k in self.var_values) and not ('*' in v):
                if v not in self.var_values[k]:
                    raise ValueError("Variable %r has not value %r" % (k, v))

        # set state
        for k, v in kwargs.iteritems():
            if add or k in self.state:
                if v is not None:
                    self.state[k] = v
            else:
                raise KeyError("No variable named %r" % k)

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

    def set_mri_subject(self, subject, mri_subject):
        """
        Reassign a subject's MRI and make sure that var_values is
        appropriately updated.

        """
        self._mri_subjects[subject] = mri_subject
        self.var_values['mrisubject'] = list(self._mri_subjects.values())

    def show_in_finder(self, key, **kwargs):
        fname = self.get(key, **kwargs)
        subprocess.call(["open", "-R", fname])

    def split_label(self, src_label, new_name, redo=False, part0='post', part1='ant'):
        """
        new_name : str
            name of the target label (``part0`` and ``part1`` are appended)
        sources : list of str
            names of the source labels

        """
        name0 = new_name + part0
        name1 = new_name + part1
        tgt0 = self.get('label_file', label=name0)
        tgt1 = self.get('label_file', label=name1)
        if (not redo) and os.path.exists(tgt0) and os.path.exists(tgt1):
            return

        label = self.load_label(label=src_label)
        fwd_fname = self.get('fwd')
        lbl0, lbl1 = split_label(label, fwd_fname, name0, name1)
        lbl0.save(tgt0)
        lbl1.save(tgt1)

    def summary(self, templates=['rawfif'], missing='-', link='>', count=True):
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
            if mri_subject == sub:
                if plot.coreg.is_fake_mri(self.get('mri_sdir')):
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
