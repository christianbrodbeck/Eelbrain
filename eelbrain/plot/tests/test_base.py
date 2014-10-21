from nose.tools import assert_raises, eq_

from eelbrain.plot import _base
from eelbrain.plot._base import Layout


class InfoObj:
    "Dummy object to stand in for objects with an info dictionary"
    def __init__(self, **info):
        self.info = info


def assert_layout_ok(*args, **kwargs):
    error = None
    l = Layout(*args, **kwargs)
    if l.nrow * l.ncol < l.nax:
        error = ("%i rows * %i cols = %i < %i (nax). args=%%r, kwargs=%%r"
                 % (l.nrow, l.ncol, l.nrow * l.ncol, l.nax))

    if error:
        raise AssertionError(error % (args, kwargs))


def test_layout():
    "Test the Layout class"
    for nax in xrange(1, 100):
        assert_layout_ok(nax, 1.5, 2, True, w=5)
        assert_layout_ok(nax, 1.5, 2, True, h=5)
        assert_layout_ok(nax, 1.5, 2, True, axw=5)
        assert_layout_ok(nax, 1.5, 2, True, axh=5)
        assert_layout_ok(nax, 1.5, 2, True, axw=5, w=20)
        assert_layout_ok(nax, 1.5, 2, True, axw=5, h=20)
        assert_layout_ok(nax, 1.5, 2, True, axh=5, w=20)
        assert_layout_ok(nax, 1.5, 2, True, axh=5, h=20)

    # single axes larger than figure
    assert_raises(ValueError, Layout, 2, 1.5, 2, True, h=5, axh=6)
    assert_raises(ValueError, Layout, 2, 1.5, 2, True, w=5, axw=6)


def test_vlims():
    "Test vlim determination"
    # symmetric
    sym_cmap = 'polar'
    v1 = InfoObj(meas='m', cmap=sym_cmap, vmax=2)
    key1 = ('m', sym_cmap)

    lims = _base.find_fig_vlims([[v1]])
    eq_(lims[key1], (-2, 2))
    lims = _base.find_fig_vlims([[v1]], False, 1)
    eq_(lims[key1], (-1, 1))
    lims = _base.find_fig_vlims([[v1]], False, 1, 0)
    eq_(lims[key1], (-1, 1))

    # zero-based
    zero_cmap = 'sig'
    v2 = InfoObj(meas='m', cmap=zero_cmap, vmax=2)
    key2 = ('m', zero_cmap)

    lims = _base.find_fig_vlims([[v2]])
    eq_(lims[key2], (0, 2))
    lims = _base.find_fig_vlims([[v2]], False, 1)
    eq_(lims[key2], (0, 1))
    lims = _base.find_fig_vlims([[v2]], False, 1, -1)
    eq_(lims[key2], (0, 1))
