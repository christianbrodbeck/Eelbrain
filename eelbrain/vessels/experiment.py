'''
mne_experiment is a base class for managing an mne experiment.



Created on May 2, 2012

@author: christian
'''

import cPickle as pickle
import fnmatch
import os
import shutil

import numpy as np
import mne

from eelbrain import ui
from eelbrain import fmtxt
from eelbrain.utils import subp
import data
from eelbrain import load
import process

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
        
        self._edir = directory
        self._log_path = os.path.join(directory, 'mne-experiment.pickle')
        
        # make sure base directories exist 
        self.templates = self.get_templates()
#        path = self.get('meg_dir', root=directory)
#        if not os.path.exists(path):
#            raise IOError("MEG-dir not found: %r" % path)
        
#        path = mri_dir.format(root=directory)
#        if not os.path.exists(path):
#            raise IOError("MRI-dir not found: %r" % path)
        
        
        
        # load config
        for cfg in ['cov', 'epochs']:
            path = self.get('config', analysis=cfg)
            if os.path.exists(path):
                obj = pickle.load(open(path))
            else:
                obj = {}
            
            setattr(self, '_cfg_%s' % cfg, obj)
        
        # find experiment data structure
        self.parse_dirs()
        
        mri_dir = self.get('mri_dir')
        lbl_dir = os.path.join(mri_dir, 'fsaverage', 'label', 'aparc')
        if os.path.exists(lbl_dir):
            self.lbl = Labels(lbl_dir)
        
        # store current values
        self._subject = None
        self._experiment = None
        self._analysis = None
        self._root = directory
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
                 # config
                 config = os.path.join(root, 'cfg_%s.pickled' % an),
                 
                 # basic dir
                 meg_dir = meg_dir, # contains subject-name folders for MEG data
                 mri_dir = mri_dir, # contains subject-name folders for MEG data
                 mri_sdir = os.path.join(mri_dir, sub),
                 raw_sdir = raw_dir,
                 log_sdir = os.path.join(meg_dir, sub, 'logs', '_'.join((sub, exp))),
                 
                 # raw
                 mrk = os.path.join(raw_dir, '_'.join((sub, exp, 'marker.txt'))),
                 elp = os.path.join(raw_dir, '*.elp'),
                 hsp = os.path.join(raw_dir, '*.hsp'),
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
                 cov = os.path.join(raw_dir, '_'.join((sub, exp, an)) + '-cov.fif'),
#                inv = os.path.join(raw_dir, '_'.join((sub, exp, an)) + '-inv.fif'),
                
                 # BESA
                 besa_triggers = os.path.join(meg_dir, sub, 'besa', '_'.join((sub, exp, an, 'triggers.txt'))),
                 besa_edt = os.path.join(meg_dir, sub, 'besa', '_'.join((sub, exp, an + '.edt'))),
                 )
        
        return t
        
    def __repr__(self):
        args = [repr(self._edir)]
        kwargs = []
#        kwargs = [('megdir', repr(self._megdir))]
        if self._subject is not None:
            kwargs.append(('subject', repr(self._subject)))
        if self._experiment is not None:
            kwargs.append(('experiment', repr(self._experiment)))
        if self._analysis is not None:
            kwargs.append(('analysis', repr(self._analysis)))
        args.extend('='.join(pair) for pair in kwargs)
        args = ', '.join(args)
        return "mne_experiment(%s)" % args
    
    def _save_cfg(self, name):
        dest = self.get('config', analysis=name)
        obj = getattr(self, '_cfg_%s' % name)
        pickle.dump(obj, open(dest, 'w'))
    
    def define_cov(self, name, experiment, stim='fixation', stimvar='stim', 
                   tstart=0.2, tstop=0.4, baseline=(None, None), edf=True):
        index = (experiment, name)
        if index in self._cfg_cov:
            raise NotImplementedError()
        
        self._cfg_cov[index] = dict(
                                    stim = stim,
                                    stimvar = stimvar,
                                    tstart = tstart,
                                    tstop = tstop,
                                    baseline = baseline,
                                    edf = edf,
                                    )
        self._save_cfg('cov')
    
    def define_epoch(self, name, experiment, stim='fixation', stimvar='stim',  
                     tstart=-0.2, tstop=0.6, baseline=(None, None), 
                     edf=True, threshold=None):
        index = (experiment, name)
        if index in self._cfg_epochs:
            raise NotImplementedError()
        
        self._cfg_epochs[index] = dict(
                                       stim = stim,
                                       stimvar = stimvar,
                                       tstart = tstart,
                                       tstop = tstop,
                                       baseline = baseline,
                                       edf = edf,
                                       threshold = threshold,
                                       )
        self._save_cfg('epochs')
    
    def do_besa_evts(self, experiment, epoch_name, subject=None, target_root=None, redo=False):
        cfg = self._cfg_epochs[(experiment, epoch_name)]
        kwargs = {k: cfg[k] for k in ('tstart', 'tstop', 'edf', 'threshold', 'stimvar', 'stim')}
        for subject, _ in self.iter_se(subject=subject, experiment=experiment, analysis=epoch_name):
            dest_trg = self.get('besa_triggers', root=target_root)
            dest_edt = self.get('besa_edt', root=target_root)
            if not redo and os.path.exists(dest_edt) and os.path.exists(dest_trg):
                continue
            
            ds = self.load_data(subject, experiment, bad=False, **kwargs)
            ds['T'] = ds['i_start'] / ds.info['samplingrate']
            
            # MEG160 export triggers
            T = ds['T']
            a = np.ones(len(T) + 10) * 2
            a[5:-5] = T.x
            triggers = data.var(a, 'triggers')
            triggers.export(dest_trg)
            
            # BESA events
            evts = data.dataset()
            
            tstart2 = cfg['tstart'] - .1
            tstop2 = cfg['tstop'] + .1
            epoch_len = tstop2 - tstart2
            start = epoch_len * 5 - tstart2
            stop = epoch_len * (5 + len(T))
            evts['Tsec'] = data.var(np.arange(start, stop, epoch_len))
            
            evts['Code'] = data.var(np.ones(len(T)))
            evts['TriNo'] = data.var(ds['eventID'].x)
            evts.export(dest_edt)
    
    def do_cov(self, experiment, cov_name, subject=None, redo=False):
        cfg = self._cfg_cov[(experiment, cov_name)]
        stimvar = cfg['stimvar']
        stim = cfg['stim'] 
        evt_kwargs = {k: cfg[k] for k in ('tstart', 'tstop', 'edf')}
        epoch_kwargs = {k: cfg[k] for k in ('tstart', 'tstop', 'baseline')}
        for subject, _ in self.iter_se(subject=subject, experiment=experiment, analysis=cov_name):
            dest = self.get('cov')
            if not redo and os.path.exists(dest):
                continue
            
            ds = self.load_evts(subject, experiment, bad=False, **evt_kwargs)
            ds = ds.subset(ds[stimvar] == stim)
            epochs = load.fiff.mne_Epochs(ds, **epoch_kwargs)
            cov = mne.cov.compute_covariance(epochs)
            cov.save(dest)
    
    def do_fwd(self, subject=None, exp=None, redo=False, v=1):
        """
        find fifs lacking forward solution
        
        redo : bool
            redo any fwd files already present
        v : 0, 1, 2
            verbosity level: 1) list converted file; 2) show all 
            mne_do_forward_solution output  
        
        """
        missing = []
        for subject, experiment in self.iter_se(subject=subject, experiment=exp):
            rawfif = self.get('rawfif', match=False)
            if os.path.exists(rawfif):
                fwd = self.get('fwd', match=False)
                if redo or (not os.path.exists(fwd)):
                    missing.append((subject, experiment))
        
        table = fmtxt.Table('ll')
        table.cells("Subject", "Experiment")
        table.midrule()
        for subject, experiment in missing:
            self.set(subject=subject, experiment=experiment)
            subp.do_forward_solution(self, overwrite=redo, v=max(0, v-1))
            table.cells(subject, experiment)
        
        self.last_job = table
        if v:
            print table
    
    def do_kit2fiff(self, do='ask', aligntol=xrange(20, 40, 5)):
        """
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
                if not os.path.exists(fifpath):
                    raw_txt.append((subject, fexp, fname))
        
        table = fmtxt.Table('lll')
        table.cells("subject", "experiment", 'file')
        for line in raw_txt:
            table.cells(*line)
        
        print table
        if do == 'ask':
            do = raw_input('convert missing (y)?') in ['y', 'Y', '\n']
        
        if do:
            failed = []
            for subject, experiment, fname in raw_txt:
                self.set(subject=subject, experiment=experiment)
                for at in aligntol:
                    try:
                        subp.kit2fiff(self, aligntol=at, **self._kit2fiff_args)
                    except RuntimeError:
                        if at < max(aligntol):
                            pass
                        else:
                            failed.append(fname)
                    else:
                        break
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
        fmt = {}
        self.set(subject=subject, experiment=experiment, analysis=analysis, 
                 match=match)
        
        if '{subject}' in temp:
            if self._subject is None:
                raise RuntimeError("No subject specified")
            else:
                subject = self._subject
            
            if name in ['bem', 'cor', 'src', 'mri_sdir']:
                subject = self._mri_subjects[subject]
            
            fmt['subject'] = subject
        
        if '{experiment}' in temp:
            if self._experiment is None:
                raise RuntimeError("No experiment specified")
            else:
                experiment = self._experiment
            
            fmt['experiment'] = experiment
        
        if '{analysis}' in temp:
            if self._analysis is None:
                raise RuntimeError("No analysis specified")
            else:
                analysis = self._analysis
            
            fmt['analysis'] = analysis
        
        if '{root}' in temp:
            if root is None:
                root = self._edir
            
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
                a = ui.ask("Launch mne_analyze for Coordinate-Coregistration?", 
                           "The 'trans' file for %r, %r does not exist. Should " 
                           "mne_analyzed be launched to create it?" % 
                           (self._subject, self._experiment),
                           cancel=False, default=True)
                if a:
                    subp.run_mne_analyze(self.get('mri_dir'),
                                         self.get('raw_sdir'), modal=True)
                    if not os.path.exists(path):
                        err = ("Error creating file; %r does not exist" % path)
                        raise IOError(err)
                else:
                    err = ("No trans file for %r, %r" % 
                           (self._subject, self._experiment))
                    raise IOError(err)
        
        path = os.path.expanduser(path)
        return path
    
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
    
    def load_data(self, subject, experiment, 
                  edf=True, bad=False,
                  tstart=-0.1, tstop=0.6, baseline=(None, 0), downsample=4,
                  stim='fixation', stimvar='stim',  
                  threshold=None):
        """
        Threshold
            1.25e-11
        bad : bool
            keep bad trials
        
        """
        ds = self.load_evts(subject, experiment, 
                            edf=edf, bad=bad, tstart=tstart, tstop=tstop)
        ds = ds.subset(ds[stimvar] == stim)
        
        load.fiff.add_epochs(ds, tstart=tstart, tstop=tstop, baseline=baseline, 
                         downsample=downsample)
        
        # threshold rejection
        if threshold:
            
            process.mark_by_threshold(ds, threshold=threshold)
            if not bad:
                ds = ds.subset('accept')
        
        return ds
    
    def load_evts(self, subject, experiment, 
                  edf=True, bad=True, tstart=-0.1, tstop=0.6):
        """
        adding labels:
        
        - ID -> label 
        
            * dict
            * dict with list key?
        
        - labels that span a trial
        
            * list with trial start IDs
            
            
            
        #.  trial start ID list
        #.  trial 
        
        """
        fifraw = self.get('rawfif', subject, experiment)
        ds = load.fiff.events(fifraw)
        if edf:
            edf_file = self.get('edf', subject, experiment)
            edf = load.eyelink.Edf(edf_file)
            edf.add_by_Id(ds, tstart=tstart, tstop=tstop)
        
        self.label_events(ds, experiment, subject)
        
        if not bad:
            ds = ds.subset('accept')
        
        return ds
    
    def load_mne_raw(self, subject=None, experiment=None):
        src = self.get('rawfif', subject=subject, experiment=experiment)
        raw = mne.fiff.Raw(src)
        return raw
    
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
    
    def pull(self, src_root, names=['rawfif', 'log_sdir'], overwrite=False):
        """
        Copies all items matching a template from another root to the current
        root.
        
        src_root : str(path)
            root of the source experiment
        names : list of str
            list of template names to copy.
            tested for 'rawfif' and 'log_sdir'. 
            Should work for any template with an exact match; '*' is not 
            implemented and will raise an error. 
        overwrite : bool
            if the item already exists in the current experiment, replace it 
         
        """
        if isinstance(names, basestring):
            names = [names]
        
        e = self.__class__(src_root)
        for name in names:
            if '{experiment}' in self.templates[name]:
                exp = None
            else:
                exp = 'NULL'
            
            for sub, exp in e.iter_se(experiment=exp):
                src = e.get(name)
                if '*' in src:
                    raise NotImplementedError("Can't fnmatch here yet")
                
                dst = self.get(name, subject=sub, experiment=exp, match=False, mkdir=True)
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
    
    def run_mne_analyze(self, subject=None, modal=False):
        mri_dir = self.get('mri_dir')
        if (subject is None) and (self._subject is None):
            fif_dir = self.get('meg_dir')
        else:
            fif_dir = self.get('raw_sdir', subject=subject)
        
        subp.run_mne_analyze(mri_dir, fif_dir, modal)
    
    def run_mne_browse_raw(self, subject=None, modal=False):
        if (subject is None) and (self._subject is None):
            fif_dir = self.get('meg_dir')
        else:
            fif_dir = self.get('raw_sdir', subject=subject)
        
        subp.run_mne_browse_raw(fif_dir, modal)
        
    def rm_cov(self, experiment, cov_name):
        index = (experiment, cov_name)
        del self._cfg_cov[index]
        
        for _ in self.iter_se(experiment=experiment, analysis=cov_name):
            path = self.get('cov')
            if os.path.exists(path):
                os.remove(path)
        
        self._save_cfg('cov')
    
    def rm_epoch(self, experiment, name):
        index = (experiment, name)
        del self._cfg_epochs[index]
        self._save_cfg('cov')
    
    def set(self, subject=None, experiment=None, analysis=None, match=False):
        """
        match : bool
            require existence
        
        """
        if subject is not None:
            if match and not (subject in self._subjects) and not ('*' in subject):
                raise ValueError("No subject named %r" % subject)
            else:
                self._subject = subject
        
        if experiment is not None:
            if match and not (experiment in self._experiments) and not ('*' in experiment):
                raise ValueError("No experiment named %r" % experiment)
            else:
                self._experiment = experiment
        
        if analysis is not None:
            self._analysis = analysis
    
    def summary(self, templates=['rawtxt', 'rawfif', 'fwd'], missing='-', link='>',
                analysis=None):
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
        
        table = fmtxt.Table('l' * (2 + len(experiments)), title=analysis)
        table.cells('Subject', 'MRI')
        experiments = list(experiments)
        table.cells(*experiments)
        table.midrule()
        
        for subject in sorted(results):
            table.cell(subject)
            mri_subject = self._mri_subjects.get('subject', '*missing*')
            if mri_subject == subject:
                table.cell('own')
            else:
                table.cell(mri_subject)
            
            for exp in experiments:
                table.cell(results[subject].get(exp, '?'))
        
        return table
    
    def summary_cov(self):
        files = {}
        table = fmtxt.Table('l' * (1 + len(self._cfg_cov)))
        table.cell('subject')
        for (exp, name), cov in self._cfg_cov.iteritems():
            table.cell('%r-%r' % (exp, name))
            for sub, _ in self.iter_se(experiment=exp, analysis=name):
                path = self.get('cov')
                exists = os.path.exists(path)
                files.setdefault(sub, []).append(exists)
        
        for s in files:
            table.cell(s)
            for v in files[s]:
                if v: 
                    table.cell('X')
                else:
                    table.cell('')
        
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
                
            
