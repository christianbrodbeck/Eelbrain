'''
mne_experiment is a base class for managing an mne experiment.


Epochs
------

The following aspects need consideration:

 - tstart, tstop of the data
 - a potential link to another epoch (e.g., a disjoint baseline) with which
   trials need to be aligned

This leads to the following name pattern:

"{tstart ms}-{tstop ms}" [ + "-{group name}"]





Created on May 2, 2012

@author: christian
'''

from collections import defaultdict
import fnmatch
import itertools
import os
import re
import shutil
import subprocess

import numpy as np

import mne

from eelbrain import ui
from eelbrain import fmtxt
from eelbrain.utils import subp
from eelbrain import load
from eelbrain import plot
from eelbrain.vessels.data import ndvar
from eelbrain.utils.print_funcs import printlist
from eelbrain.utils.kit import split_label


__all__ = ['mne_experiment']



_kit2fiff_args = {'sfreq':1000, 'lowpass':100, 'highpass':0,
                  'stimthresh':2.5, 'stim':xrange(168, 160, -1)}


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
        for k, v in self.var_values.iteritems():
            if v:
                self.state[k] = v[0]

        # set defaults for any existing templates
        for k in self.state.keys():
            temp = self.state[k]
            for name in self._fmt_pattern.findall(temp):
                if name not in self.state:
                    self.state[name] = '<%s not set>' % name

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
             'trans': os.path.join('{raw_sdir}', '{subject}-trans.fif'),  # mne p. 196

             # eye-tracker
             'edf': os.path.join('{log_sdir}', '*.edf'),

             # mne raw-derivatives analysis
             'proj_file': '{raw}_{proj}-proj.fif',
             'proj_plot': '{raw}_{proj}-proj.pdf',
             'cov': '{raw}_{fwd_an}-cov.fif',
             'fwd': '{raw}_{fwd_an}-fwd.fif',

             # fwd model
             'fid': os.path.join('{mri_sdir}', 'bem', '{mrisubject}-fiducials.fif'),
             'bem': os.path.join('{mri_sdir}', 'bem', '{mrisubject}-5120-bem-sol.fif'),
             'src': os.path.join('{mri_sdir}', 'bem', '{mrisubject}-ico-4-src.fif'),
             'bem_head': os.path.join('{mri_sdir}', 'bem', '{mrisubject}-head.fif'),

             # mne's stc.save() requires stub filename and will add '-?h.stc'
             'evoked_dir': os.path.join('{meg_sdir}', 'evoked'),
             'evoked': os.path.join('{evoked_dir}', '{experiment}_{cell}_{epoch}_{proj}-evoked.fif'),
             'stc_dir': os.path.join('{meg_sdir}', 'stc_{fwd_an}_{stc_an}'),
             'stc': os.path.join('{stc_dir}', '{experiment}_{cell}_{epoch}'),
             'stc_morphed': os.path.join('{stc_dir}', '{experiment}_{cell}_{common_brain}'),
             'label': os.path.join('{mri_sdir}', '{labeldir}', '{hemi}.{analysis}.label'),
             'morphmap': os.path.join('{mri_dir}', 'morph-maps', '{subject}-{common_brain}-morph.fif'),

             'n_key': '{subject}_{experiment}_{epoch}_{cell}_{proj}',

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

    def combine_labels(self, target, sources=[], hemi=['lh', 'rh'], redo=False):
        """
        target : str
            name of the target label
        sources : list of str
            names of the source labels

        """
        msg = "Making Label: %s" % target
        for _ in self.iter_vars(['mrisubject'], values={'hemi': hemi}, prog=msg):
            tgt = self.get('label', analysis=target)
            if redo or not os.path.exists(tgt):
                srcs = [self.get('label', analysis=name) for name in sources]
                label = mne.read_label(srcs.pop(0))
                for path in srcs:
                    label += mne.read_label(path)
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

    def iter_vars(self, variables, constants={}, values={}, exclude={},
                  prog=False):
        """
        variables : list
            variables which should be iterated
        constants : dict(name -> value)
            variables with constant values throughout the iteration
        values : dict(name -> (list of values))
            variables with values to iterate over instead of the corresponding
            `mne_experiment.var_values`
        exclude : dict(name -> (list of values))
            values to exclude from the iteration
        prog : bool | str
            Show a progress dialog; str for dialog title.

        """
        state_ = self.state.copy()

        # set constants
        constants['root'] = self.root
        self.set(**constants)

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
        return ds

    def load_edf(self, subject=None, experiment=None):
        src = self.get('edf', subject=subject, experiment=experiment)
        edf = load.eyelink.Edf(src)
        return edf

    def load_events(self, subject=None, experiment=None,
                    proj=True, edf=True, raw=None):
        """OK 12/7/3

        Loads events from the corresponding raw file, adds the raw to the info
        dict.

        proj : True | False | str
            load a projection file and add it to the raw
        edf : bool
            Loads edf and add it to the info dict.

        """
        self.set(subject=subject, experiment=experiment, raw=raw)
        raw_file = self.get('rawfif')
        if isinstance(proj, str):
            proj = self.get('proj_file', proj=proj)
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

    def makeplt_coreg(self, plotname='{subject}_{experiment}', constants={}):
        """
        view name will be added to plotname

        """
        self.set(analysis='coreg')
        for _ in self.iter_vars(['subject', 'experiment'], constants=constants):
            p = self.plot_coreg()
            p.frontal = True
            p.scene.save(self.get('plot_png', name=plotname + '-F', mkdir=True))
            p.lateral = True
            p.scene.save(self.get('plot_png', name=plotname + '-L'))
            p.top = True
            p.scene.save(self.get('plot_png', name=plotname + '-T'))

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

        fwd = mne.read_forward_solution(self.get('fwd'))
        raw = mne.fiff.Raw(self.get('rawfif'))
        bem = self.get('bem_head')

        return plot.sensors.coreg(raw, fwd, bem=bem)

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

        if ui.ask("Delete?", '\n'.join(sorted(rmd + rmf)), default=False):
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

    def set(self, subject=None, experiment=None, match=False, add=False,
            **kwargs):
        """
        special variables
        -----------------

        subjects:
            The corresponding `mrisubject` is also set

        match : bool
            require existence (for variables in self.var_values)

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

    def set_epoch(self, tstart, tstop, name=None):
        desc = '%i-%i' % (tstart * 1000, tstop * 1000)
        if name:
            desc += '-%s' % name
        self.set(epoch=desc, add=True)

    def set_fwd_an(self, stim, tw, proj):
        temp = '{stim}-{tw}-{proj}'
        fwd_an = temp.format(stim=stim, tw=tw, proj=proj)
        self.set(fwd_an=fwd_an, add=True)

    def set_mri_subject(self, subject, mri_subject):
        """
        Reassign a subject's MRI and make sure that var_values is
        appropriately updated.

        """
        self._mri_subjects[subject] = mri_subject
        self.var_values['mrisubject'] = list(self._mri_subjects.values())

    def set_stc_an(self, blc, method, ori):
        temp = '{blc}-{method}-{ori}'
        stc_an = temp.format(blc=blc, method=method, ori=ori)
        self.set(stc_an=stc_an, add=True)

    def show_in_finder(self, key, **kwargs):
        fname = self.get(key, **kwargs)
        subprocess.call(["open", "-R", fname])

    def split_label(self, src_label, new_name, redo=False, part0='post', part1='ant', hemi=['lh', 'rh']):
        """
        new_name : str
            name of the target label ('post' and 'ant' is appended)
        sources : list of str
            names of the source labels

        """
        msg = "Splitting Label: %s" % src_label
        for _ in self.iter_vars(['mrisubject'], values={'hemi': hemi}, prog=msg):
            name0 = new_name + part0
            name1 = new_name + part1
            tgt0 = self.get('label', analysis=name0)
            tgt1 = self.get('label', analysis=name1)
            if (not redo) and os.path.exists(tgt0) and os.path.exists(tgt1):
                continue

            src = self.get('label', analysis=src_label)
            label = mne.read_label(src)
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
