# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from pathlib import Path

import numpy as np
from numpy.testing import assert_array_equal

from eelbrain import Dataset, datasets, load
from eelbrain.testing import TempDir, assert_dataobj_equal, assert_dataset_equal, file_path


def test_r_tsv_io():
    "Test reading output of write.table"
    ds = load.tsv(file_path('r-write.table.txt'))
    assert_array_equal(ds['row'], ['1', '2'])
    assert_array_equal(ds['bin'], [0, 0])


def test_tsv_io():
    """Test tsv I/O"""
    tempdir = TempDir()
    names = ['A', 'B', 'rm', 'intvar', 'fltvar', 'fltvar2', 'index']

    # get Dataset
    ds = datasets.get_uv()
    ds['fltvar'][5:10] = np.nan
    ds[:4, 'rm'] = ''

    # save and load
    dst = Path(tempdir) / 'ds.txt'
    ds.save_txt(dst)
    ds1 = load.tsv(dst)
    assert_dataset_equal(ds1, ds, decimal=10)
    ds1 = load.tsv(dst, skiprows=1, names=names)
    assert_dataset_equal(ds1, ds, decimal=10)

    # guess data types with missing
    intvar2 = ds['intvar'].as_factor()
    intvar2[10:] = ''
    ds_intvar = Dataset((intvar2,))
    ds_intvar.save_txt(dst)
    ds_intvar1 = load.tsv(dst, empty='nan')
    assert_dataobj_equal(ds_intvar1['intvar', :10], ds['intvar', :10])
    assert_array_equal(ds_intvar1['intvar', 10:], np.nan)
