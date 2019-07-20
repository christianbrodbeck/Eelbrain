# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from eelbrain import datasets, load
from eelbrain._utils import IS_OSX
from eelbrain.testing import assert_dataobj_equal, file_path


def test_pickle():
    ds = datasets.get_uts()

    decimal = None if IS_OSX else 15
    ds_2 = load.unpickle(file_path('uts-py2.pickle'))
    assert_dataobj_equal(ds_2, ds, decimal)
    ds_3 = load.unpickle(file_path('uts-py3.pickle'))
    assert_dataobj_equal(ds_3, ds, decimal)
