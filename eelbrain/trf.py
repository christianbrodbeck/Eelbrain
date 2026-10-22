# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Components for implementing TRF estimators

The data layout (:class:`DeconvolutionData`), cross-validation scheme
(:class:`Splits`) and lag-window conventions (:func:`lag_window`,
:func:`lag_matrix`) that :func:`eelbrain.boosting` is built on, shared with the
`NCRF <https://github.com/proloyd/neuro-currentRF>`_ package.
"""
import os

from ._config import CONFIG
from ._trf.shared import DeconvolutionData, RowBlock, SegmentedArray, Split, Splits, lag_matrix, lag_window, package_values, split_data

__all__ = ['DeconvolutionData', 'RowBlock', 'SegmentedArray', 'Split', 'Splits', 'lag_matrix', 'lag_window', 'package_values', 'split_data', 'n_workers', 'init_worker']


def n_workers() -> int:
    "Number of worker processes configured for multiprocessing (see :func:`eelbrain.configure`; 0 when multiprocessing is disabled)"
    return CONFIG['n_workers']


def init_worker() -> None:
    "Apply Eelbrain's configuration to a worker process (the scheduling priority, see :func:`eelbrain.configure`); call from the worker"
    if CONFIG['nice']:
        os.nice(CONFIG['nice'])
