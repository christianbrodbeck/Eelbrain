# -*- coding: utf-8 -*-
from nose.tools import eq_
import os
import shutil
import tempfile

from eelbrain import fmtxt
from eelbrain.fmtxt import html, tex
from eelbrain import datasets, plot


def test_fmtext():
    "Tet FMText base class"
    t = fmtxt.FMText('test')
    print t

    tuc = fmtxt.FMText(u'FMText with unicode: \x80abc')
    print str(tuc)

    ts = fmtxt.FMText((t, tuc, u'unicode: \x80abc'))

    print str(ts)
    print html(ts)
    print tex(ts)


def test_report():
    "Test fmtxt.Report class"
    tempdir = tempfile.mkdtemp()
    report = fmtxt.Report("Test Report")

    section = report.add_section(u'unicode: \xe2 abc')
    ds = datasets.get_uv()
    p = plot.Barplot('fltvar', 'A', sub="B=='b1'", ds=ds)
    image = p.image()
    section.add_figure("test", image)

    report.sign()

    # report output
    print report
    dst = os.path.join(tempdir, 'report.html')
    report.save_html(dst)

    # clean up
    shutil.rmtree(tempdir)


def test_eq():
    "Test equation factory"
    s = fmtxt.eq('t', 0.1234)
    eq_(str(s), "t = 0.12")
    eq_(html(s), "t = 0.12")
    eq_(tex(s), "$t = 0.12$")

    s = fmtxt.eq('t', 0.1234, 18)
    eq_(str(s), "t(18) = 0.12")
    eq_(html(s), "t<sub>18</sub> = 0.12")
    eq_(tex(s), "$t_{18} = 0.12$")

    s = fmtxt.peq(0.1299)
    eq_(str(s), "p = .130")
    eq_(html(s), "p = .130")
    eq_(tex(s), "$p = .130$")

    s = fmtxt.peq(0.0009)
    eq_(str(s), "p < .001")
    eq_(html(s), "p < .001")
    eq_(tex(s), "$p < .001$")


def test_symbol():
    "Test fmtxt.symbol()"
    s = fmtxt.symbol('t', 21)
    eq_(str(s), 't(21)')
    eq_(html(s), 't<sub>21</sub>')
    eq_(tex(s), '$t_{21}$')
