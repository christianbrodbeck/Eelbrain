"""Test mne interaction"""
import mne

from eelbrain.lab import datasets, testnd


def test_source_estimate():
    "Test SourceSpace dimension"
    ds = datasets.get_mne_sample(src=True)

    # source space clustering
    res = testnd.ttest_ind('src', 'side', ds=ds, samples=0, tstart=0.05,
                           mintime=0.02, minsource=10)
