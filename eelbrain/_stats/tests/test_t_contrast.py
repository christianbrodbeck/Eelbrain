# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from eelbrain._stats import t_contrast

from nose.tools import eq_, assert_raises
import numpy as np
from numpy.testing import assert_array_equal


def test_t_contrast():
    y = np.arange(9.).reshape((3, 3))
    indexes = {'a': 0, 'b': 1, 'c': 2}

    contrast = "+sum(a>c, b>c)"
    contrast_ = t_contrast._parse_t_contrast(contrast)
    eq_(contrast_, ('func', '+', np.sum, [('comp', None, 'a', 'c'),
                                                   ('comp', None, 'b', 'c')]))

    contrast = "+sum(a>*, b>*)"
    contrast_ = t_contrast._parse_t_contrast(contrast)
    eq_(contrast_, ('func', '+', np.sum, [('comp', None, 'a', '*'),
                                                   ('comp', None, 'b', '*')]))
    _, cells = t_contrast._t_contrast_rel_properties(contrast_)
    pc, mc = t_contrast._t_contrast_rel_expand_cells(cells, ('a', 'b', 'c'))
    data = t_contrast._t_contrast_rel_data(y, indexes, pc, mc)
    assert_array_equal(data['a'], np.arange(3.))
    assert_array_equal(data['*'], y.mean(0))

    assert_raises(ValueError, t_contrast._t_contrast_rel_expand_cells, cells,
                  ('a|c', 'b|c', 'c|c'))
