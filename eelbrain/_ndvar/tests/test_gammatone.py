"""Requires test data at root/test_data: gammatone.pickle, gammatone-left.pickle"""
import pytest

from eelbrain import load, gammatone_bank
from eelbrain.testing import assert_dataobj_equal
from eelbrain.testing.data import TEST_DATA_DIRECTORY, generate_sound


def test_gammatone_bank():
    target_path = TEST_DATA_DIRECTORY / 'gammatone.pickle'
    if not target_path.exists():
        pytest.skip(f"Test data missing: {target_path}")
    sound = generate_sound()

    target = load.unpickle(target_path)
    gt = gammatone_bank(sound, 20, 2000, 32, 1 / 100)
    assert_dataobj_equal(gt, target)

    target_left = load.unpickle(TEST_DATA_DIRECTORY / 'gammatone-left.pickle')
    gt = gammatone_bank(sound, 20, 2000, 32, 1 / 100, location='left')
    assert_dataobj_equal(gt, target_left)
