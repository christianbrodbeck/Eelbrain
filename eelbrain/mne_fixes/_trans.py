from itertools import izip

import numpy as np
from mne import Transform


def dig_equal(dig1, dig2):
    return all(np.array_equal(a['r'], b['r']) for a, b in izip(dig1, dig2))


def trans_equal(trans1, trans2):
    assert isinstance(trans1, Transform)
    assert isinstance(trans2, Transform)
    return all(trans1['from'] == trans2['from'], trans1['to'] == trans2['from'],
               np.array_equal(trans1['trans'], trans2['trans']))
