'''
Defines some basic example datasets that are used in testing.
'''
import os

import numpy as np

import mne
from mne import minimum_norm as mn

from . import load
from ._colorspaces import eeg_info
from ._data_obj import Dataset, Factor, Var, NDVar, Sensor, UTS
from ._design import permute


def get_loftus_masson_1994():
    "Dataset used for illustration purposes by Loftus and Masson (1994)"
    ds = Dataset()
    ds['subject'] = Factor(range(1, 11), tile=3, random=True)
    ds['exposure'] = Var([1, 2, 5], repeat=10)
    ds['n_recalled'] = Var([10, 6, 11, 22, 16, 15, 1, 12, 9, 8,
                            13, 8, 14, 23, 18, 17, 1, 15, 12, 9,
                            13, 8, 14, 25, 20, 17, 4, 17, 12, 12])
    return ds


def _mne_source_space(subject, src_tag, subjects_dir):
    """Load mne source space"""
    src_file = os.path.join(subjects_dir, subject, 'bem',
                            '%s-%s-src.fif' % (subject, src_tag))
    src = src_tag[:3]
    if os.path.exists(src_file):
        return mne.read_source_spaces(src_file, False)
    elif src == 'ico':
        return mne.setup_source_space(subject, src_file, 'ico4',
                                      subjects_dir=subjects_dir, add_dist=True)
    elif src == 'vol':
        mri_file = os.path.join(subjects_dir, subject, 'mri', 'orig.mgz')
        bem_file = os.path.join(subjects_dir, subject, 'bem',
                                'sample-5120-5120-5120-bem-sol.fif')
        return mne.setup_volume_source_space(subject, src_file, pos=10.,
                                             mri=mri_file, bem=bem_file,
                                             mindist=0., exclude=0.,
                                             subjects_dir=subjects_dir)
    else:
        raise ValueError("src_tag=%s" % repr(src_tag))


def get_mne_sample(tmin=-0.1, tmax=0.4, baseline=(None, 0), sns=False,
                   src=None, sub="modality=='A'", fixed=False, snr=2,
                   method='dSPM', rm=False, stc=False):
    """Load events and epochs from the MNE sample data

    Parameters
    ----------
    tmin, tmax baseline :
        Epoch parameters.
    sns : bool
        Add sensor space data as NDVar as ``ds['sns']`` (default ``False``).
    src : False | 'ico' | 'vol'
        Add source space data as NDVar as ``ds['src']`` (default ``False``).
    sub : str | list | None
        Expresion for subset of events to load. For a very small dataset use e.g.
        ``[0,1]``.
    fixed : bool
        MNE inverse parameter.
    snr : scalar
        MNE inverse parameter.
    method : str
        MNE inverse parameter.
    rm : bool
        Pretend to be a repeated measures dataset (adds 'subject' variable).
    stc : bool
        Add mne SourceEstimate for source space data as ``ds['stc']`` (default
        ``False``).

    Returns
    -------
    ds : Dataset
        Dataset with epochs from the MNE sample dataset in ``ds['epochs']``.
    """
    data_dir = mne.datasets.sample.data_path()
    meg_dir = os.path.join(data_dir, 'MEG', 'sample')
    raw_file = os.path.join(meg_dir, 'sample_audvis_filt-0-40_raw.fif')
    event_file = os.path.join(meg_dir, 'sample_audvis_filt-0-40-eve.fif')
    subjects_dir = os.path.join(data_dir, 'subjects')
    subject = 'sample'
    label_path = os.path.join(subjects_dir, subject, 'label', '%s.label')

    if not os.path.exists(event_file):
        raw = mne.io.Raw(raw_file)
        events = mne.find_events(raw, stim_channel='STI 014')
        mne.write_events(event_file, events)
    ds = load.fiff.events(raw_file, events=event_file)
    ds.index()
    ds.info['subjects_dir'] = subjects_dir
    ds.info['subject'] = subject
    ds.info['label'] = label_path

    # get the trigger variable form the dataset for eaier access
    trigger = ds['trigger']

    # use trigger to add various labels to the dataset
    ds['condition'] = Factor(trigger, labels={1:'LA', 2:'RA', 3:'LV', 4:'RV',
                                              5:'smiley', 32:'button'})
    ds['side'] = Factor(trigger, labels={1: 'L', 2:'R', 3:'L', 4:'R',
                                         5:'None', 32:'None'})
    ds['modality'] = Factor(trigger, labels={1: 'A', 2:'A', 3:'V', 4:'V',
                                             5:'None', 32:'None'})

    if rm:
        ds = ds.sub('trigger < 5')
        ds = ds.equalize_counts('side % modality')
        subject_f = ds.eval('side % modality').enumerate_cells()
        ds['subject'] = subject_f.as_factor('s%r', random=True)

    if sub:
        ds = ds.sub(sub)

    load.fiff.add_mne_epochs(ds, tmin, tmax, baseline)
    if sns:
        ds['sns'] = load.fiff.epochs_ndvar(ds['epochs'], data='mag',
                                           sysname='neuromag306mag')

    if not src:
        return ds
    elif src == 'ico':
        src_tag = 'ico-4'
    elif src == 'vol':
        src_tag = 'vol-10'
    else:
        raise ValueError("src = %r" % src)
    epochs = ds['epochs']

    # get inverse operator
    inv_file = os.path.join(meg_dir, 'sample_eelbrain_%s-inv.fif' % src_tag)
    if os.path.exists(inv_file):
        inv = mne.minimum_norm.read_inverse_operator(inv_file)
    else:
        fwd_file = os.path.join(meg_dir, 'sample-%s-fwd.fif' % src_tag)
        bem_dir = os.path.join(subjects_dir, subject, 'bem')
        bem_file = os.path.join(bem_dir, 'sample-5120-5120-5120-bem-sol.fif')
        trans_file = os.path.join(meg_dir, 'sample_audvis_raw-trans.fif')

        if os.path.exists(fwd_file):
            fwd = mne.read_forward_solution(fwd_file)
        else:
            src_ = _mne_source_space(subject, src_tag, subjects_dir)
            fwd = mne.make_forward_solution(epochs.info, trans_file, src_,
                                            bem_file, fwd_file)

        cov_file = os.path.join(meg_dir, 'sample_audvis-cov.fif')
        cov = mne.read_cov(cov_file)
        inv = mn.make_inverse_operator(epochs.info, fwd, cov, None, None,
                                       fixed)
        mne.minimum_norm.write_inverse_operator(inv_file, inv)
    ds.info['inv'] = inv

    stcs = mn.apply_inverse_epochs(epochs, inv, 1. / (snr ** 2), method)
    ds['src'] = load.fiff.stc_ndvar(stcs, subject, src_tag, subjects_dir,
                                    method, fixed)
    if stc:
        ds['stc'] = stcs

    return ds


def get_uts(utsnd=False, seed=0):
    """Create a sample Dataset with 60 cases and random data.

    Parameters
    ----------
    utsnd : bool
        Add a sensor by time NDVar (called 'utsnd').
    seed : None | int
        If not None, call ``numpy.random.seed(seed)`` to ensure replicability.

    Returns
    -------
    ds : Dataset
        Datasets with data from random distributions.
    """
    if seed is not None:
        np.random.seed(seed)

    ds = Dataset()

    # add a model
    ds['A'] = Factor(['a0', 'a1'], repeat=30)
    ds['B'] = Factor(['b0', 'b1'], repeat=15, tile=2)
    ds['rm'] = Factor(('R%.2i' % i for i in xrange(15)), tile=4, random=True)
    ds['ind'] = Factor(('R%.2i' % i for i in xrange(60)), random=True)

    # add dependent variables
    rm_var = np.tile(np.random.normal(size=15), 4)
    y = np.hstack((np.random.normal(size=45), np.random.normal(1, size=15)))
    y += rm_var
    ds['Y'] = Var(y)
    ybin = np.random.randint(0, 2, size=60)
    ds['YBin'] = Factor(ybin, labels={0:'c1', 1:'c2'})
    ycat = np.random.randint(0, 3, size=60)
    ds['YCat'] = Factor(ycat, labels={0:'c1', 1:'c2', 2:'c3'})

    # add a uts NDVar
    time = UTS(-.2, .01, 100)
    y = np.random.normal(0, .5, (60, len(time)))
    y += rm_var[:, None]
    y[:15, 20:60] += np.hanning(40) * 1  # interaction
    y[:30, 50:80] += np.hanning(30) * 1  # main effect
    ds['uts'] = NDVar(y, dims=('case', time))

    # add sensor NDVar
    if utsnd:
        locs = np.array([[-1.0, 0.0, 0.0],
                         [ 0.0, 1.0, 0.0],
                         [ 1.0, 0.0, 0.0],
                         [ 0.0, -1.0, 0.0],
                         [ 0.0, 0.0, 1.0]])
        sensor = Sensor(locs, sysname='test_sens')
        sensor.set_connectivity(connect_dist=1.75)

        y = np.random.normal(0, 1, (60, 5, len(time)))
        y += rm_var[:, None, None]
        # add interaction
        win = np.hanning(50)
        y[:15, 0, 50:] += win * 3
        y[:15, 1, 50:] += win * 2
        y[:15, 4, 50:] += win
        # add main effect
        y[30:, 2, 25:75] += win * 2.5
        y[30:, 3, 25:75] += win * 1.5
        y[30:, 4, 25:75] += win
        # add spectral effect
        freq = 15.0  # >= 2
        x = np.sin(time.times * freq * 2 * np.pi)
        for i in xrange(30):
            shift = np.random.randint(0, 100 / freq)
            y[i, 2, 25:75] += 1.1 * win * x[shift: 50+shift]
            y[i, 3, 25:75] += 1.5 * win * x[shift: 50+shift]
            y[i, 4, 25:75] += 0.5 * win * x[shift: 50+shift]

        dims = ('case', sensor, time)
        ds['utsnd'] = NDVar(y, dims, eeg_info())

    return ds


def get_uv(seed=0):
    """Dataset with random univariate data

    Parameters
    ----------
    seed : None | int
        Seed the numpy random state before generating random data.
    """
    if seed is not None:
        np.random.seed(seed)

    ds = permute([('A', ('a1', 'a2')),
                  ('B', ('b1', 'b2')),
                  ('rm', ['s%03i' % i for i in xrange(20)])])
    ds['rm'].random = True
    ds['intvar'] = Var(np.random.randint(5, 15, 80))
    ds['intvar'][:20] += 3
    ds['fltvar'] = Var(np.random.normal(0, 1, 80))
    ds['fltvar'][:40] += 1.
    ds['fltvar2'] = Var(np.random.normal(0, 1, 80))
    ds['fltvar2'][40:] += ds['fltvar'][40:].x
    ds['index'] = Var(np.repeat([True, False], 40))
    return ds
