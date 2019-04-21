import os
import itertools
from nose.tools import assert_in, assert_raises, assert_true, \
    assert_equal
from eelbrain.testing import TempDir

from ..mrat import DatasetSTCLoader


def _create_fake_files(tmpdir):
    """Write empty stc files to tmpdir to be detected
    by the DatasetSTCLoader"""
    subjects = ("R0000", "A9999")
    factor1 = ("level-a", "level-b")
    factor2 = ("noun", "verb")
    folder_combos = itertools.product(factor1, factor2)
    file_combos = itertools.product(subjects, factor1, factor2)
    folders = ["_".join(f) for f in folder_combos]
    files = ["_".join(f) + "-lh.stc" for f in file_combos]
    for f in folders:
        folder_path = os.path.join(tmpdir, f)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        files_to_write = filter(lambda x: f in x, files)
        for fw in files_to_write:
            file_path = os.path.join(folder_path, fw)
            open(file_path, "a").close()


def test_loader():
    tmp = TempDir()
    _create_fake_files(tmp)
    loader = DatasetSTCLoader(tmp)
    levels = loader.levels
    # check detected level names
    assert_in("level-a", levels[0])
    assert_in("level-b", levels[0])
    assert_in("noun", levels[1])
    assert_in("verb", levels[1])
    assert_in("R0000", loader.subjects)
    assert_raises(ValueError, loader.set_factor_names, ["only-one-factor"])
    loader.set_factor_names(["factor1", "factor2"])
    assert_equal(loader.design_shape, "2 x 2")
    ds = loader.make_dataset(load_stcs=False)
    assert_in("factor1", ds)
    assert_in("factor2", ds)
    assert_in("subject", ds)
    assert_true(ds["subject"].random)
