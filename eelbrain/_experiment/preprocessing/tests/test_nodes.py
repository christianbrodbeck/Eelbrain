# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import mne
import numpy as np
import pandas as pd
import pytest
from mne_bids import BIDSPath

from eelbrain._experiment.preprocessing.nodes import RawHeadPositionDerivative, RawSourceInput, mean_head_position


def test_read_raw_applies_bids_channels(tmp_path):
    "RawSourceInput._apply_bids_channels applies channels.tsv channel metadata"
    bids_path = BIDSPath(
        root=tmp_path,
        subject='01',
        task='test',
        datatype='meg',
        suffix='meg',
        extension='.fif',
    )
    bids_path.fpath.parent.mkdir(parents=True)
    info = mne.create_info(['MEG 001', 'EOG 001'], 100, ['mag', 'misc'])
    data = np.zeros((2, 100))
    raw = mne.io.RawArray(data, info, verbose='error')
    raw.save(bids_path.fpath, overwrite=True, verbose='error')

    channels_path = bids_path.copy().update(suffix='channels', extension='.tsv').fpath
    channels = pd.DataFrame({
        'name': ['MEG 001', 'EOG 001'],
        'type': ['MEGMAG', 'EOG'],
        'units': ['T', 'V'],
        'status': ['good', 'bad'],
    })
    channels.to_csv(channels_path, sep='\t', index=False)

    raw_read = RawSourceInput._read_raw(bids_path.fpath, preload=False)
    RawSourceInput._apply_bids_channels(bids_path, raw_read)

    assert raw_read.get_channel_types(picks=['EOG 001']) == ['eog']
    assert raw_read.info['bads'] == []


def generate_head_positions(n: int) -> np.ndarray:
    """Synthetic (n, 10) head positions in MaxFilter format"""
    out = np.empty((n, 10))
    out[:, 0] = 42.0 + np.arange(n) * 0.01  # t
    out[:, 1:4] = np.column_stack([np.linspace(.010, .014, n), np.linspace(-.005, -.001, n), np.full(n, .002)])  # q1, q2, q3
    out[:, 4:7] = np.column_stack([np.linspace(.001, .003, n), np.full(n, -.002), np.linspace(.050, .053, n)])  # tx, ty, tz
    out[:, 7:] = np.column_stack([np.full(n, .99), np.full(n, .001), np.full(n, .01)])  # gof, err, v
    return out


@pytest.mark.parametrize('n', [1, 5])
def test_head_position_roundtrip(tmp_path, n):
    """RawHeadPositionDerivative save/load preserves the MaxFilter (n, 10) format"""
    node = RawHeadPositionDerivative('raw-input@raw')
    path = tmp_path / 'test.pos'
    positions = generate_head_positions(n)
    node.save(None, path, positions)
    loaded = node.load(None, path)
    assert loaded.shape == (n, 10)
    assert np.allclose(loaded, positions, atol=1e-5)
    # _check_pos requires strictly ascending times, which the 3-decimal .pos time format must not collapse
    assert (np.diff(loaded[:, 0]) > 0).all()


def test_head_position_none_roundtrip(tmp_path):
    """RawHeadPositionDerivative encodes None as an empty file"""
    node = RawHeadPositionDerivative('raw-input@raw')
    path = tmp_path / 'test.pos'
    node.save(None, path, None)
    assert path.stat().st_size == 0
    assert node.load(None, path) is None


def test_mean_head_position():
    """mean_head_position returns the Fréchet mean, or None without movement"""
    positions = generate_head_positions(5)[:, 1:7]
    assert mean_head_position(positions[:1]) is None
    assert mean_head_position(np.tile(positions[:1], (5, 1))) is None

    # samples with movement yield a proper transform
    trans = mean_head_position(positions)
    assert trans['from'] == mne.io.constants.FIFF.FIFFV_COORD_DEVICE
    assert trans['to'] == mne.io.constants.FIFF.FIFFV_COORD_HEAD
    assert np.allclose(trans['trans'][:3, 3], positions[:, 3:].mean(0))
    # the rotation is a proper rotation, which an element-wise mean of rotation matrices would not be
    rot = trans['trans'][:3, :3]
    assert np.allclose(rot @ rot.T, np.eye(3))
