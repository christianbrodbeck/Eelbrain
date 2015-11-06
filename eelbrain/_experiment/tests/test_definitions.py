# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from nose.tools import eq_

from eelbrain._experiment.definitions import (find_epoch_vars, find_epochs_vars,
    find_test_vars)


def test_find_epoch_vars():
    eq_(find_epoch_vars({'sel': "myvar == 'x'"}), ('myvar',))
    eq_(find_epoch_vars({'post_baseline_trigger_shift': "myvar"}), ('myvar',))

    epochs = {'a': {'sel': "vara == 'a'"},
              'b': {'sel': "logical_and(varb == 'b', varc == 'c')"},
              'sec': {'sel_epoch': 'a', 'sel': "svar == 's'"},
              'super': {'sub_epochs': ('a', 'b')}}
    eq_(find_epochs_vars(epochs), {'a': ('vara',),
                                   'b': ('logical_and', 'varb', 'varc'),
                                   'sec': ('vara', 'svar'),
                                   'super': ('vara', 'logical_and', 'varb', 'varc')})

def test_find_test_vars():
    eq_(find_test_vars({'kind': 'anova', 'model': "a % b % c"}),
        ('a', 'b', 'c'))
