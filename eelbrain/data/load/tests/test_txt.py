# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import os
import shutil
import tempfile

import numpy as np

from eelbrain.data import datasets, load

from ...tests.test_data import assert_dataset_equal


def test_tsv_io():
    """Test tsv I/O"""
    # get Dataset
    ds = datasets.get_uv()
    ds['fltvar'][5:10] = np.nan

    # save and load
    tempdir = tempfile.mkdtemp()
    try:
        dst = os.path.join(tempdir, 'ds.txt')
        ds.save_txt(dst)
        ds_1 = load.txt.tsv(dst)
    finally:
        shutil.rmtree(tempdir)

    assert_dataset_equal(ds, ds_1, "TSV write/read test failed", 10)
