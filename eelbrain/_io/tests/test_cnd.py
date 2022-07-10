# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from pathlib import Path

import pytest

from eelbrain import load


def test_cnd_dilibach():
    "Data from https://cnsp-workshop.github.io/website/resources.html"
    root = Path('~/Data/CND/diliBach').expanduser()
    if not root.exists():
        pytest.skip("CND diliBach dataset not found")
    # Stimuli
    ds = load.cnd(root / 'dataCND' / 'dataStim.mat')
    assert ds.n_cases == 30
    assert ds[0, 'Speech_Envelope_Vectors'].shape == (10172,)
    # Data
    ds = load.cnd(root / 'dataCND' / 'dataSub1.mat')
    assert ds.n_cases == 30
    assert ds[0, 'eeg'].shape == (82395, 64)
    assert ds[0, 'extChan'].shape == (82395, 2)
