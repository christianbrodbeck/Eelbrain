'''
mne_experiment is a base class for managing an mne experiment.



Created on May 2, 2012

@author: christian
'''

import fnmatch
import itertools
import os
import re
import shutil

from collections import defaultdict

import mne

from eelbrain import ui
from eelbrain import fmtxt
from eelbrain.utils import subp
from eelbrain import load
from eelbrain import plot
from eelbrain.vessels.data import ndvar


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


class mne_experiment(object):
    def __init__(self, directory=None, 
                 subject=None, experiment=None, analysis=None,
                 kit2fiff_args=_kit2fiff_args):
        """
        directory : str
            the root directory for the experiment (i.e., the directory 
            containing the 'meg' and 'mri' directories) 
        
        fwd : None | dict
            dictionary specifying the forward model parameters
        
        """
        if directory:
            directory = os.path.expanduser(directory)
        else:
            msg = "Please select the meg directory of your experiment"
            directory = ui.ask_dir("Select Directory", msg, True)
        
        self._kit2fiff_args = kit2fiff_args
        
        self._log_path = os.path.join(directory, 'mne-experiment.pickle')
        
        # templates ---
        self.templates = self.get_templates()
        
        # dictionaries ---
        self.edf_use = defaultdict(lambda: ['ESACC', 'EBLINK'])
        self.bad_channels = defaultdict(lambda: ['MEG 065']) # (sub, exp) -> list
        
        
        # find experiment data structure
        self.var_values = {}
        self.state = {'root': directory}
        self.parse_dirs()
        
        mri_dir = self.get('mri_dir')
        lbl_dir = os.path.join(mri_dir, 'fsaverage', 'label', 'aparc')
        if os.path.exists(lbl_dir):
            self.lbl = Labels(lbl_dir)
        
        # store current values
        self.set(subject=subject, experiment=experiment, analysis=analysis)
    
    def get_templates(self):
        # path elements
        root = '{root}'
        sub = '{subject}'
        exp = '{experiment}'
        an = '{analysis}'
        meg_dir = os.path.join(root, 'meg')
        mri_dir = os.path.join(root, 'mri')
        raw_dir = os.path.join(meg_dir, sub, 'raw')
        mne_dir = os.path.join(meg_dir, sub, 'mne')
        log_dir = os.path.join(meg_dir, sub, 'logs', '_'.join((sub, exp)))
        
        t = dict(
                 # basic dir
                 meg_dir = meg_dir, # contains subject-name folders for MEG data
                 mri_dir = mri_dir, # contains subject-name folders for MEG data
                 mri_sdir = os.path.join(mri_dir, sub),
                 raw_sdir = raw_dir,
                 log_sdir = os.path.join(meg_dir, sub, 'logs', '_'.join((sub, exp))),
                 
                 # raw
                 mrk = os.path.join(raw_dir, '_'.join((sub, exp, 'marker.txt'))),
                 elp = os.path.join(raw_dir, sub+'_HS.elp'),
                 hsp = os.path.join(raw_dir, sub+'_HS.hsp'),
                 rawtxt = os.path.join(raw_dir, '_'.join((sub, exp, '*raw.txt'))),
                 rawfif = os.path.join(raw_dir, '_'.join((sub, exp, 'raw.fif'))),
                 trans = os.path.join(raw_dir, '_'.join((sub, exp, 'raw-trans.fif'))), # mne p. 196
                 
                 # eye-tracker
                 edf = os.path.join(log_dir, '*.edf'),
                 
                 # mne analysis
                 projs = os.path.join(mne_dir, '_'.join((sub, exp, an, 'projs.fif'))),
                 
                 # fwd model
                 fwd = os.path.join(mne_dir, '_'.join((sub, exp, an, 'fwd.fif'))),
                 bem = os.path.join(mri_dir, sub, 'bem', sub+'-5120-bem-sol.fif'),
                 src = os.path.join(mri_dir, sub, 'bem', sub+'-ico-4-src.fif'),
                 
                # !! these would invalidate the s_e_* pattern with a third _
                 cov = os.path.join(mne_dir, '_'.join((sub, exp, an)) + '-cov.fif'),
                 
                 # mne's stc.save() requires stub filename and will add '-?h.stc'  
                 stc_tgt = os.path.join(mne_dir, '_'.join((sub, exp, an))),
                 stc = os.path.join(mne_dir, '_'.join((sub, exp, an)) + '.stc'),
                 label = os.path.join(mri_dir, sub, 'label', 'aparc', '%s.label' % an),
                 
                 # EEG
                 vhdr = os.path.join(meg_dir, sub, 'raw_eeg', '_'.join((sub, exp+'.vhdr'))),
                 eegfif = os.path.join(meg_dir, sub, 'raw_eeg', '_'.join((sub, exp, 'raw.fif'))),
                 
                 # BESA
                 besa_triggers = os.path.join(meg_dir, sub, 'besa', '_'.join((sub, exp, an, 'triggers.txt'))),
                 besa_evt = os.path.join(meg_dir, sub, 'besa', '_'.join((sub, exp, an + '.evt'))),
                 )
        
        return t
        
    def __repr__(self):
        args = [repr(self.state['root'])]
        kwargs = []
        
        subject = self.state.get('subject')
        if subject is not None:
            kwargs.append(('subject', repr(subject)))
        
        experiment = self.state.get('experiment')
        if experiment is not None:
            kwargs.append(('experiment', repr(experiment)))
        
        analysis = self.state.get('analysis')
        if analysis is not None:
            kwargs.append(('analysis', repr(analysis)))
        
        args.extend('='.join(pair) for pair in kwargs)
        args = ', '.join(args)
        return "mne_experiment(%s)" % args
    
    def do_kit2fiff(self, do='ask', aligntol=xrange(15, 40, 5), redo=False):
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
        for subject in self._subjects:
            temp = self.get('rawtxt', experiment='*', subject=subject, match=False)
            tdir, tname = os.path.split(temp)
            fnames = fnmatch.filter(os.listdir(tdir), tname)
            for fname in fnames:
                fs, fexp, _ = fname.split('_', 2)
                fifpath = self.get('rawfif', subject=fs, experiment=fexp, match=False)
                if redo or not os.path.exists(fifpath):
                    raw_txt.append((subject, fexp, fname))
        
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
            for subject, experiment, fname in raw_txt:
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
            
            print aligntols
            
            if len(failed) > 0:
                table = fmtxt.Table('l')
                table.cell("Failed")
                table.cells(*failed)
                print table
        else:
            return raw_txt
    
    def get(self, name, subject=None, experiment=None, analysis=None, root=None,
            match=True, mkdir=False):
        """
        Retrieve a path. With match=True, '*' are expanded to match a file, 
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
        temp = self.templates[name]
        self.set(subject=subject, experiment=experiment, analysis=analysis, 
                 match=match)
        fmt = self.state.copy()
        
        if '{subject}' in temp:
            subject = fmt.get('subject')
            if subject is None:
                raise RuntimeError("No subject specified")
            elif name in ['bem', 'cor', 'src', 'mri_sdir', 'label']:
                fmt['subject'] = self._mri_subjects[subject]
        
        if ('{experiment}' in temp) and (fmt.get('experiment') is None):
            raise RuntimeError("No experiment specified")
        
        if ('{analysis}' in temp) and (fmt.get('analysis') is None):
            raise RuntimeError("No analysis specified")
        
        if '{root}' in temp:
            if root is not None:
                if root.endswith(os.path.extsep):
                    root = root[:-1]
                
                fmt['root'] = root
        
        path = temp.format(**fmt)
        
        # assert the presence of the file
        directory, fname = os.path.split(path)
        if match and ('*' in fname):
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
        if name =='trans':
            if not os.path.exists(path):
                ui.show_help(subp.run_mne_analyze)
                a = ui.ask("Launch mne_analyze for Coordinate-Coregistration?", 
                           "The 'trans' file for %r, %r does not exist. Should " 
                           "mne_analyzed be launched to create it?" % 
                           (self.state['subject'], self.state['experiment']),
                           cancel=False, default=True)
                if a:
                    subp.run_mne_analyze(self.get('mri_dir'),
                                         self.get('raw_sdir'), modal=True)
                    if not os.path.exists(path):
                        err = ("Error creating file; %r does not exist" % path)
                        raise IOError(err)
                else:
                    err = ("No trans file for %r, %r" % 
                           (self.state['subject'], self.state['experiment']))
                    raise IOError(err)
        
        path = os.path.expanduser(path)
        return path
    
    def iter_temp(self, name):
        temp = self.templates[name]
        pattern = re.compile('\{(\w+)\}')
        variables = set(pattern.findall(temp)).difference(['root'])
        variables = list(variables)
        for state in self.iter_vars(variables):
            path = temp.format(**state)
            yield path
    
    def iter_path(self, temp, ignore=['root']):
        pattern = re.compile('\{(\w+)\}')
        variables = set(pattern.findall(temp)).difference(ignore)
        variables = list(variables)
        for state in self.iter_vars(variables):
            path = temp.format(**state)
            yield path
    
    def iter_vars(self, variables):
        var_values = tuple(self.var_values[v] for v in variables)
        for v_list in itertools.product(*var_values):
            self.state.update(dict(zip(variables, v_list)))
            yield self.state
    
    def iter_se(self, subject=None, experiment=None, analysis=None):
        """
        iterate through subject and experiment names
        
        """
        subjects = self._subjects if subject is None else [subject]
        experiments = self._experiments if experiment is None else [experiment] 
        for subject in subjects:
            for experiment in experiments:
                self.set(subject, experiment, analysis)
                yield subject, experiment
    
    def label_events(self, ds, experiment, subject):
        return ds
        
    def load_edf(self, subject=None, experiment=None):
        src = self.get('edf', subject=subject, experiment=experiment)
        edf = load.eyelink.Edf(src)
        return edf
    
    def load_events(self, subject=None, experiment=None, proj='fixation', 
                    edf=True):
        """OK 12/7/3
        
        Loads events from the corresponding raw file, adds the raw to the info 
        dict. 
        
        proj : None | name
            load a projection file and ad it to the raw
        edf : bool
            Loads edf and add it to the info dict.
        
        """
        self.set(subject=subject, experiment=experiment)
        
        raw = self.load_raw(proj=proj)
        ds = load.fiff.events(raw)
        
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
    
    def load_raw(self, subject=None, experiment=None, proj='fixation'):
        """OK 12/6/18
        
        proj : None | str
            name of the projections to load
        
        """
        self.set(subject=subject, experiment=experiment)
        
        src = self.get('rawfif')
        raw = mne.fiff.Raw(src)
        bad_chs = self.bad_channels[(self.state['subject'], self.state['experiment'])]
        raw.info['bads'].extend(bad_chs)
        
        if proj:
            proj_file = self.get('projs', analysis=proj)
            proj = mne.read_proj(proj_file)
            raw.info['projs'] += proj[:]
        
        return raw
    
    def make_proj_for_epochs(self, epochs, dest, n_mag=5):
        """
        computes the first ``n_mag`` PCA components, plots them, and asks for 
        user input (a tuple) on which ones to save.
        
        epochs : mne.Epochs
            epochs which should be used for the PCA
        
        dest : str(path)
            path where to save the projections
        
        n_mag : int
            number of components to compute 
            
        """
        proj = mne.proj.compute_proj_epochs(epochs, n_grad=0, n_mag=n_mag, n_eeg=0)
    
        sensor = load.fiff.sensors(epochs)
        
        # plot PCA components
        PCA = []
        for p in proj:
            d = p['data']['data'][0]
            name = p['desc'][-5:]
            v = ndvar(d, (sensor,), name=name)
            PCA.append(v)
        
        title = os.path.basename(dest)
        p = plot.topo.topomap(PCA, size=1, title=title)
        rm = None
        while not isinstance(rm, tuple):
            rm = input("which components to remove? (tuple / 'x'): ")
            if rm == 'x': raise
        proj = [proj[i] for i in rm]
        p.Close()
        mne.write_proj(dest, proj)
    
    def parse_dirs(self):
        """
        find subject and experiment names by looking through directory structure
        
        """
        subjects = self._subjects = set()
        
        meg_dir = self.get('meg_dir')
        if os.path.exists(meg_dir):
            for fname in os.listdir(meg_dir):
                isdir = os.path.isdir(os.path.join(meg_dir, fname))
                isname = not fname.startswith('.')
                raw_sdir = self.get('raw_sdir', subject=fname, match=False)
                hasraw = os.path.exists(raw_sdir)
                if isdir and isname and hasraw:
                    subjects.add(fname)
        
        # find MRIs
        mri_dir = self.get('mri_dir')
        self._mri_subjects = mri_subjects = {}
        if os.path.exists(mri_dir):
            mris = os.listdir(mri_dir)
            for s in subjects:
                if s in mris:
                    mri_subjects[s] = s
                elif 'fsaverage' in mris:
                    mri_subjects[s] = 'fsaverage'
#                else:
#                    err = "If subject has no mri, fsaverage must be provided"
#                    raise IOError(err)
        
        # find experiments
        experiments = self._experiments = set()
        for s in subjects:
            temp_fif = self.get('rawfif', subject=s, experiment='*', match=False)
            temp_txt = self.get('rawtxt', subject=s, experiment='*', match=False)
            
            tdir, fifname = os.path.split(temp_fif)
            _, txtname = os.path.split(temp_txt)
            all_fnames = os.listdir(tdir)
            fnames = fnmatch.filter(all_fnames, fifname)
            fnames.extend(fnmatch.filter(all_fnames, txtname))
            for fname in fnames:
                experiments.add(fname.split('_')[1])
        
        self.var_values['subject'] = list(subjects)
        self.var_values['experiment'] = list(experiments)
    
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
        e = self.__class__(src_root)
        e.push(self._root, names=names, **kwargs)
    
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
    
    def run_mne_analyze(self, subject=None, modal=False):
        mri_dir = self.get('mri_dir')
        if (subject is None) and (self.state['subject'] is None):
            fif_dir = self.get('meg_dir')
        else:
            fif_dir = self.get('raw_sdir', subject=subject)
        
        subp.run_mne_analyze(mri_dir, fif_dir, modal)
    
    def run_mne_browse_raw(self, subject=None, modal=False):
        if (subject is None) and (self.state['subject'] is None):
            fif_dir = self.get('meg_dir')
        else:
            fif_dir = self.get('raw_sdir', subject=subject)
        
        subp.run_mne_browse_raw(fif_dir, modal)
        
    def set(self, subject=None, experiment=None, analysis=None, match=False):
        """
        match : bool
            require existence
        
        """
        if subject is not None:
            if match and not (subject in self._subjects) and not ('*' in subject):
                raise ValueError("No subject named %r" % subject)
            else:
                self.state['subject'] = subject
        
        if experiment is not None:
            if match and not (experiment in self._experiments) and not ('*' in experiment):
                raise ValueError("No experiment named %r" % experiment)
            else:
                self.state['experiment'] = experiment
        
        if analysis is not None:
            self.state['analysis'] = analysis
    
    def summary(self, templates=['rawtxt', 'rawfif', 'fwd'], missing='-', link='>',
                analysis=None, count=True):
        if not isinstance(templates, (list, tuple)):
            templates = [templates]
        
        results = {}
        experiments = set()
        for sub, exp in self.iter_se(analysis=analysis):
            items = []
            
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
        
        table = fmtxt.Table('l' * (2 + len(experiments) + count), title=analysis)
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
            mri_subject = self._mri_subjects.get(subject, '*missing*')
            if mri_subject == subject:
                table.cell('own')
            else:
                table.cell(mri_subject)
            
            for exp in experiments:
                table.cell(results[subject].get(exp, '?'))
        
        return table
    
    def sync(self, template, source=None, dest=None, 
             subject=None, experiment=None, analysis=None, replace=False, v=1):
        """
        copies all files corresponding to a ``template`` *from* the 
        ``source`` experiment, *to* the ``dest`` experiment.
        
        template : str
            name of the template to copy
        replace : bool
            if the file already exists at the destination, replace it
        
        """
        results = {} # {subject -> {experiment -> status}}
        experiments = set()
        for subject, experiment in self.iter_se(subject=subject, experiment=experiment, analysis=analysis):
            src = self.get(template, root=source, match=False)            
            dst = self.get(template, root=dest, match=False)
            
            if '*' in src:
                raise NotImplementedError()
            elif os.path.exists(src):
                if os.path.exists(dst):
                    if replace:
                        shutil.copyfile(src, dst)
                        status = 'replaced'
                    else:
                        status = 'present'
                else:
                    dirname = os.path.dirname(dst)
                    if not os.path.exists(dirname):
                        os.mkdir(dirname)
                    shutil.copyfile(src, dst)
                    status = 'copied'
            elif os.path.exists(dst):
                status = 'src missing'
            else:
                status = 'missing'
            
            results.setdefault(subject, {})[experiment] = status
            experiments.add(experiment)
            
        # create table
        experiments = list(experiments)
        table = fmtxt.Table('l' * (len(experiments) + 1))
        table.cell('subject')
        table.cells(*experiments)
        table.midrule()
        for subject in results:
            table.cell(subject)
            for exp in experiments:
                table.cell(results[subject].get(exp, '?'))
        
        self.last_job = table
        if v:
            return table
                
            
