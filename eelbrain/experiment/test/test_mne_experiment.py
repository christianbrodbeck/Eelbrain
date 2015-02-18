# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from nose.tools import eq_
from eelbrain.experiment import MneExperiment


def test_mne_experiment_templates():
    "Test MneExperiment template formatting"
    e = MneExperiment('root', False)

    eq_(e.get('src-kind'), 'clm bestreg free-3-dSPM')
    e.set_inv('fixed')
    eq_(e.get('src-kind'), 'clm bestreg fixed-3-dSPM')
    e.set(cov='noreg')
    eq_(e.get('src-kind'), 'clm noreg fixed-3-dSPM')
    e.set(raw='0-40')
    eq_(e.get('src-kind'), '0-40 noreg fixed-3-dSPM')
