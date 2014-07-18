# -*- coding: utf-8 -*-
from nose.tools import assert_equal
import os
import shutil
import tempfile

from eelbrain import fmtxt
from eelbrain.fmtxt import html, tex
from eelbrain import datasets, plot


def test_symbol():
    "Test fmtxt.symbol()"
    s = fmtxt.symbol('t', 21)
    assert_equal(str(s), 't(21)')
    assert_equal(html(s), 't<sub>21</sub>')


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
    p = plot.uv.barplot('fltvar', 'A', sub="B=='b1'", ds=ds)
    image = p.image()
    section.add_figure("test", image)

    report.sign()

    # report output
    print report
    dst = os.path.join(tempdir, 'report.html')
    report.save_html(dst)

    # clean up
    shutil.rmtree(tempdir)
