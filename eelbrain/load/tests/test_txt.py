# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from nose.tools import assert_raises
import os
import shutil
import tempfile

import numpy as np

from eelbrain import datasets, load

from ...tests.test_data import assert_dataset_equal


def test_tsv_io():
    """Test tsv I/O"""
    names = ['A', 'B', 'rm', 'intvar', 'fltvar', 'index']

    # get Dataset
    ds = datasets.get_uv()
    ds['fltvar'][5:10] = np.nan
    ds[:4, 'rm'] = ''

    # save and load
    tempdir = tempfile.mkdtemp()
    try:
        dst = os.path.join(tempdir, 'ds.txt')

        ds.save_txt(dst)
        ds1 = load.tsv(dst)
        ds2 = load.tsv(dst, skiprows=1, names=names)
    finally:
        shutil.rmtree(tempdir)

    assert_dataset_equal(ds1, ds, "TSV write/read test failed", 10)
    assert_dataset_equal(ds2, ds, "TSV write/read test failed", 10)
