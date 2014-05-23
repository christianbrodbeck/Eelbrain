# -*- coding: utf-8 -*-
from nose.tools import assert_equal

from eelbrain import fmtxt
from eelbrain.fmtxt import html


def test_symbol():
    "Test fmtxt.symbol()"
    s = fmtxt.symbol('t', 21)
    assert_equal(str(s), 't(21)')
    assert_equal(html(s), 't<sub>21</sub>')


def test_texstr():
    t = fmtxt.FMText('test')
    print t
