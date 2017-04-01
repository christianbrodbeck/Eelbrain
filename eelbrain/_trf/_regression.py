# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Literal, Union, Tuple, Sequence

import mne
import numpy as np
import sklearn.metrics

from .._data_obj import Case, Dataset, Dimension, NDVar, Var, CategorialArg, NDVarArg, combine, dataobj_repr
from .._utils import LazyProperty, PickleableDataClass, tqdm, user_activity
from .shared import PredictorData, DeconvolutionData, Split, Splits, merge_segments


@dataclass(eq=False)
class TimeShiftRegressionResult(PickleableDataClass):
    # basic parameters
    y: str
    x: Union[str, Tuple[str]]
    tstart: Union[float, Tuple[float, ...]]
    tstop: Union[float, Tuple[float, ...]]
    # results
    h: Union[NDVar, Tuple[NDVar, ...]]
    r2: Union[float, NDVar] = None
    i_test: int = None
    partition_results: List[TimeShiftRegressionResult] = None

    def __post_init__(self):
        self._h_is_list = not isinstance(self.x, str)

    @LazyProperty
    def trf_ds(self):
        rows = []
        for res in self.partition_results:
            hs = res.h if self._h_is_list else [res.h]
            rows.append([res.i_test, *hs])
        xs = self.x if self._h_is_list else [self.x]
        return Dataset.from_caselist(['i_test', *xs], rows)


@dataclass(eq=False)
class SplitResult:
    split: Split
    h: np.ndarray
    r2: np.ndarray


class TimeShiftRegression:

    tstart: float = None
    tstop: float = None
    _split_results: list = None
    _y_pred = None

    def __init__(self, data: DeconvolutionData):
        self.data = data

    @user_activity
    def fit(
            self,
            tstart: Union[float, Sequence[float]],
            tstop: Union[float, Sequence[float]],
            alpha: float = 0.1,
            fit_intercept: bool = True,
    ):
        self.data._check_data()
        assert not self.data.splits.n_validate, 'Validation not implemented'
        if self.data.splits.n_test:
            if not self.data.has_case:
                raise NotImplementedError
        else:
            raise NotImplementedError

        # prepare estimator
        tstep = self.data.time.tstep
        h_n_samples = int(round((tstop - tstart) / tstep))
        imin = int(round(tstart / tstep))
        n_y, n_times = self.data.y.shape
        n_x = len(self.data.x)
        istop = imin + h_n_samples
        imax = istop - 1

        if self.data.has_case:
            n_times = len(self.data.time)
            x_data = self.data.x.reshape((n_x, -1, n_times)).swapaxes(0, 2)
            y_data = self.data.y.reshape((n_y, -1, n_times)).swapaxes(0, 2)
        else:
            x_data = self.data.x.T
            y_data = self.data.y.T

        estimator = mne.decoding.ReceptiveField(imin, imax, 1, estimator=alpha, fit_intercept=fit_intercept)
        if self.data.splits.n_test:
            y_pred = np.empty_like(y_data)
        else:
            y_pred = None
        split_results = []
        for split in tqdm(self.data.splits.splits, "Fitting models", leave=False):
            y_train = y_data[:, split.train_set]
            x_train = x_data[:, split.train_set]
            estimator.fit(x_train, y_train)
            # predict test set
            y_pred[:, split.test_set] = estimator.predict(x_data[:, split.test_set])
            y_test = y_data[:, split.test_set].reshape((-1, n_y))
            y_pred_test = y_pred[:, split.test_set].reshape((-1, n_y))
            r2 = sklearn.metrics.r2_score(y_test, y_pred_test, multioutput='raw_values')
            # package result
            res = SplitResult(split, estimator.coef_, r2)
            split_results.append(res)

        self.tstart = tstart
        self.tstop = tstop
        self._split_results = split_results
        self._y_pred = y_pred.swapaxes(0, 2).reshape((n_y, -1))

    def evaluate_fit(
            self,
            i_test: int = None,
            partition_results: bool = False,
    ):
        if i_test is not None:
            split_results = [res for res in self._split_results if res.split.i_test == i_test]
            assert len(split_results) == 1
            split_result = split_results[0]
            h = self.data.package_kernel(split_result.h, self.tstart)
            r2 = self.data.package_value(split_result.r2, 'R2')
            return TimeShiftRegressionResult(self.data.y_name, self.data.x_name, self.tstart, self.tstop, h, r2, i_test)

        if partition_results:
            partition_results_ = [self.evaluate_fit(split.i_test) for split in self.data.splits.splits]
        else:
            partition_results_ = None
        # average h
        h = np.zeros_like(self._split_results[0].h)
        for res in self._split_results:
            h += res.h
        h /= len(self._split_results)
        h = self.data.package_kernel(h, self.tstart)
        # overall r2
        r2 = sklearn.metrics.r2_score(self.data.y.T, self._y_pred.T, multioutput='raw_values')
        r2 = self.data.package_value(r2, 'R2')
        return TimeShiftRegressionResult(self.data.y_name, self.data.x_name, self.tstart, self.tstop, h, r2, partition_results=partition_results_)

    @LazyProperty
    def cross_prediction(self):
        return self.data.package_y_like(self._y_pred, self.data.y_name)
