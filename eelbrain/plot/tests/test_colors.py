# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from nose.tools import assert_equal, assert_raises

from eelbrain import plot


def test_generate_colors():
    "Test functions for assigning colors to cells"

    cells_1 = ('A', 'B')
    cells_2 = ('a', 'b', 'c')

    colors = plot.colors_for_oneway(cells_1)
    assert_equal(len(colors), len(cells_1))

    colors = plot.colors_for_twoway(cells_1, cells_2)
    assert_equal(len(colors), len(cells_1) * len(cells_2))

    assert_raises(NotImplementedError, plot.colors_for_twoway, ('A', 'B', 'C'),
                  cells_2)


def test_plot_colors():
    "Test plotting color schemes"
    plot.configure_backend(False, False)

    cells_1 = ('A', 'B')
    cells_2 = ('a', 'b', 'c')

    colors = plot.colors_for_oneway(cells_1)
    p = plot.ColorList(colors)
    p.close()

    colors = plot.colors_for_twoway(cells_1, cells_2)
    p = plot.ColorList(colors)
    p.close()
