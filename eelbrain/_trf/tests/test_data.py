# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from itertools import chain

from numpy.testing import assert_array_equal

from eelbrain import datasets, epoch_impulse_predictor
from eelbrain._trf.shared import RevCorrData


def test_segmentation():
    ds = datasets.get_uts()
    n_times = len(ds['uts'].time)
    ds['imp'] = epoch_impulse_predictor('uts', ds=ds)
    ds['imp_a'] = epoch_impulse_predictor('uts', "A == 'a1'", ds=ds)

    data = RevCorrData('uts', 'imp', 'l1', True, ds)
    assert_array_equal(data.segments, [[i * n_times, (i + 1) * n_times] for i in range(60)])
    data.initialize_cross_validation(6)
    assert len(data.cv_segments) == 6
    data.initialize_cross_validation(6, 'A', ds)
    allsegments = sorted(map(tuple, data.segments))
    # test that all segments are used
    for segments, train, test in data.cv_segments:
        assert sorted(tuple(i) for i in chain(train, test)) == allsegments
    # test that cells are used equally
    for index in data.cv_indexes:
        assert ds[index, 'imp_a'].sum() == 5  # 30 (number of a1) / 6
