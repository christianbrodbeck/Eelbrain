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
    # a rebuild writes over the previous artifact in place, so None has to replace earlier positions
    node.save(None, path, generate_head_positions(5))
    node.save(None, path, None)
    assert node.load(None, path) is None


def test_mean_head_position():
    """mean_head_position weights samples by the time they were held, or returns None without movement"""
    info = mne.create_info(['MEG 0111'], 100., 'mag')
    raws = [mne.io.RawArray(np.zeros((1, 1000)), info, first_samp=4200, verbose=False) for _ in range(2)]  # 10 s each, starting at t=42 s
    positions = generate_head_positions(5)  # t = 42.00, 42.01, ..., 42.04
    static = positions[:1]
    assert mean_head_position(raws[:1], [static]) is None
    assert mean_head_position(raws, [static, static]) is None

    # a tracked recording with movement and a static recording
    trans = mean_head_position(raws, [positions, static])
    assert trans['from'] == mne.io.constants.FIFF.FIFFV_COORD_DEVICE
    assert trans['to'] == mne.io.constants.FIFF.FIFFV_COORD_HEAD
    # each sample counts until the next sample or the end of its recording: 4 x 10 ms, then 9.96 s for the last tracked sample and 10 s for the static one
    weights = np.array([.01, .01, .01, .01, 9.96, 10.])
    samples = np.vstack([positions, static])
    assert np.allclose(trans['trans'][:3, 3], (weights / weights.sum()) @ samples[:, 4:7], atol=1e-6)
    # the rotation is a proper rotation, which an element-wise mean of rotation matrices would not be
    rot = trans['trans'][:3, :3]
    assert np.allclose(rot @ rot.T, np.eye(3))

    # BAD segments are excluded: with the static recording marked bad, only the tracked recording contributes
    raws[1].set_annotations(mne.Annotations(0., 10., 'BAD_all'))  # onset relative to the start of the data
    trans_bad = mean_head_position(raws, [positions, static])
    trans_tracked = mean_head_position(raws[:1], [positions])
    assert np.allclose(trans_bad['trans'], trans_tracked['trans'])
