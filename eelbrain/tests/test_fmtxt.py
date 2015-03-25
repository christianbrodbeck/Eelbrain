# -*- coding: utf-8 -*-
from nose.tools import eq_
import os
import shutil
import tempfile

from eelbrain import fmtxt
from eelbrain._utils.testing import TempDir
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
    p = plot.Barplot('fltvar', 'A', sub="B=='b1'", ds=ds, show=False)
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


def test_table():
    table = fmtxt.Table('ll')
    table.cells('A', 'B')
    table.midrule()
    table.cells('a1', 'b1', 'a2', 'b2')
    eq_(str(table), 'A    B \n-------\na1   b1\na2   b2')
    eq_(table.get_html(), u'<figure><table rules="none" cellpadding="2" '
                          u'frame="hsides" border="1"><tr>\n'
                          u' <td>A</td>\n <td>B</td>\n</tr>\n<tr>\n'
                          u' <td>a1</td>\n <td>b1</td>\n</tr>\n<tr>\n'
                          u' <td>a2</td>\n <td>b2</td>\n</tr></table></figure>')
    eq_(table.get_rtf(), '\\trowd\n\\cellx0000\n\\cellx1000\n\\row\n'
                         'A\\intbl\\cell\nB\\intbl\\cell\n\\row\n'
                         'a1\\intbl\\cell\nb1\\intbl\\cell\n\\row\n'
                         'a2\\intbl\\cell\nb2\\intbl\\cell\n\\row')
    eq_(table.get_tex(), '\\begin{center}\n\\begin{tabular}{ll}\n\\toprule\n'
                         'A & B \\\\\n\\midrule\n'
                         'a1 & b1 \\\\\na2 & b2 \\\\\n'
                         '\\bottomrule\n\\end{tabular}\n\\end{center}')

    # saving
    tempdir = TempDir()
    # HTML
    path = os.path.join(tempdir, 'test.html')
    table.save_html(path)
    eq_(open(path).read(),  '<!DOCTYPE html>\n<html>\n<head>\n'
                            '    <title>Untitled</title>\n</head>\n\n'
                            '<body>\n\n<figure>'
                            '<table rules="none" cellpadding="2" frame="hsides" '
                            'border="1"><tr>\n'
                            ' <td>A</td>\n <td>B</td>\n</tr>\n<tr>\n'
                            ' <td>a1</td>\n <td>b1</td>\n</tr>\n<tr>\n'
                            ' <td>a2</td>\n <td>b2</td>\n</tr>'
                            '</table></figure>\n\n</body>\n</html>\n')
    # rtf
    path = os.path.join(tempdir, 'test.rtf')
    table.save_rtf(path)
    eq_(open(path).read(), '{\\rtf1\\ansi\\deff0\n\n'
                           '\\trowd\n\\cellx0000\n\\cellx1000\n\\row\n'
                           'A\\intbl\\cell\nB\\intbl\\cell\n\\row\n'
                           'a1\\intbl\\cell\nb1\\intbl\\cell\n\\row\n'
                           'a2\\intbl\\cell\nb2\\intbl\\cell\n\\row\n}')
    # TeX
    path = os.path.join(tempdir, 'test.tex')
    table.save_tex(path)
    eq_(open(path).read(), '\\begin{center}\n\\begin{tabular}{ll}\n\\toprule\n'
                           'A & B \\\\\n\\midrule\n'
                           'a1 & b1 \\\\\na2 & b2 \\\\\n'
                           '\\bottomrule\n\\end{tabular}\n\\end{center}')
    # txt
    path = os.path.join(tempdir, 'test.txt')
    table.save_txt(path)
    eq_(open(path).read(), 'A    B \n-------\na1   b1\na2   b2')


def test_symbol():
    "Test fmtxt.symbol()"
    s = fmtxt.symbol('t', 21)
    eq_(str(s), 't(21)')
    eq_(html(s), 't<sub>21</sub>')
    eq_(tex(s), '$t_{21}$')
