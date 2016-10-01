# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Test MneExperiment using mne-python sample data"""
import imp
from os.path import join, realpath

from nose.tools import eq_

from eelbrain import *

from ..._utils.testing import TempDir, assert_dataobj_equal, requires_mne_sample_data


@requires_mne_sample_data
def test_sample():
    set_log_level('warning', 'mne')

    # import from file:  http://stackoverflow.com/a/67692/166700
    e_path = realpath(join(__file__, '..', '..', '..', '..', 'examples',
                           'experiment', 'sample_experiment.py'))
    e_module = imp.load_source('sample_experiment', e_path)

    tempdir = TempDir()
    datasets.setup_samples_experiment(tempdir, 3, 2)

    root = join(tempdir, 'SampleExperiment')
    e = e_module.SampleExperiment(root)
    e.set(raw='clm')

    eq_(e.get('subject'), 'R0000')
    eq_(e.get('subject', subject='R0002'), 'R0002')

    e.set('R0001')
    e.make_bad_channels(['MEG 0331'], redo=True)

    sds = []
    for _ in e:
        e.make_rej(auto=2.5e-12)
        sds.append(e.load_evoked())

    ds = e.load_evoked('all')
    assert_dataobj_equal(combine(sds), ds)
