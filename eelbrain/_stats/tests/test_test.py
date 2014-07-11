# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>

from eelbrain import datasets, test


def test_ttest():
    "Test test.ttest"
    ds = datasets.get_uv()

    print test.ttest('fltvar', ds=ds)
    print test.ttest('fltvar', 'A', ds=ds)
    print test.ttest('fltvar', 'A%B', ds=ds)
    print test.ttest('fltvar', 'A', match='rm', ds=ds)
