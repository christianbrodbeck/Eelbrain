# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from nose.tools import eq_, assert_raises
from eelbrain.experiment import MneExperiment


def test_mne_experiment_templates():
    "Test MneExperiment template formatting"
    e = MneExperiment('', False)

    # Don't create dirs without root
    assert_raises(IOError, e.get, 'raw-file', mkdir=True)

    # compounds
    eq_(e.get('src-kind'), 'clm bestreg free-3-dSPM')
    e.set_inv('fixed')
    eq_(e.get('src-kind'), 'clm bestreg fixed-3-dSPM')
    e.set(cov='noreg')
    eq_(e.get('src-kind'), 'clm noreg fixed-3-dSPM')
    e.set(raw='0-40')
    eq_(e.get('src-kind'), '0-40 noreg fixed-3-dSPM')

    # inv
    e.set_inv('free', 3, 'dSPM', .8, True)
    eq_(e.get('inv'), 'free-3-dSPM-0.8-pick_normal')
    eq_(e._params['make_inv_kw'], {'loose': 1})
    eq_(e._params['apply_inv_kw'], {'method': 'dSPM', 'lambda2': 1. / 3**2})
    e.set_inv('fixed', 2, 'MNE', pick_normal=True)
    eq_(e.get('inv'), 'fixed-2-MNE-pick_normal')
    eq_(e._params['make_inv_kw'], {'fixed': True, 'loose': None})
    eq_(e._params['apply_inv_kw'], {'method': 'MNE', 'lambda2': 1. / 2**2,
                                    'pick_normal': True})
    e.set_inv(0.5, 3, 'sLORETA')
    eq_(e.get('inv'), 'loose.5-3-sLORETA')
    eq_(e._params['make_inv_kw'], {'loose': 0.5})
    eq_(e._params['apply_inv_kw'], {'method': 'sLORETA', 'lambda2': 1. / 3**2})
