# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from typing import List
from warnings import catch_warnings, filterwarnings

import numpy as np
from scipy.linalg import norm
from scipy.stats import spearmanr

from ._boosting_opt import l1, l2
from .shared import RevCorrData


class Evaluator:
    vector = False
    attr = NotImplemented
    name = NotImplemented
    meas = None

    def __init__(self, data: RevCorrData):
        n = len(data.y)
        if self.vector:
            if not data.vector_dim:
                raise ValueError(f"{self.__class__.__name__}: Vector evaluator for non-vector data")
            n //= len(data.vector_dim)
        self.x = np.empty(n, np.float64)

    def add_y(
            self,
            i: int,  # y index (row in data.y)
            y: np.ndarray,  # actual data
            y_pred: np.ndarray,  # data predicted by model
    ):
        raise NotImplementedError

    def __repr__(self):
        return f"<{self.__class__.__name__} evaluator>"


class L1(Evaluator):
    attr = 'residual'
    name = 'L1 residuals'

    def add_y(self, i, y, y_pred):
        index = np.array(((0, len(y)),), np.int64)
        self.x[i] = l1(y - y_pred, index)


class L2(Evaluator):
    attr = 'residual'
    name = 'L2 residuals'

    def add_y(self, i, y, y_pred):
        index = np.array(((0, len(y)),), np.int64)
        self.x[i] = l2(y - y_pred, index)


class Correlation(Evaluator):
    attr = 'r'
    name = 'Correlation'
    meas = 'r'

    def add_y(self, i, y, y_pred):
        with catch_warnings():
            filterwarnings('ignore', "invalid value encountered", RuntimeWarning)
            r = np.corrcoef(y, y_pred)[0, 1]
        self.x[i] = 0 if np.isnan(r) else r


class RankCorrelation(Evaluator):
    attr = 'r_rank'
    name = 'Rank correlation'
    meas = 'r'

    def add_y(self, i, y, y_pred):
        with catch_warnings():
            filterwarnings('ignore', "invalid value encountered", RuntimeWarning)
            r = spearmanr(y, y_pred)[0]
        self.x[i] = 0 if np.isnan(r) else r


class VectorL1(Evaluator):
    vector = True
    attr = 'residual'
    name = 'Vector l1 residuals'

    def add_y(self, i, y, y_pred):
        y_pred_error = norm(y - y_pred, axis=0)
        self.x[i] = y_pred_error.mean(-1)


class VectorL2(Evaluator):
    vector = True
    attr = 'residual'
    name = 'Vector l2 residuals'

    def add_y(self, i, y, y_pred):
        dist = y - y_pred
        dist **= 2
        self.x[i] = y.sum() / y.shape[1]


class VectorCorrelation(Evaluator):
    vector = True
    attr = 'r'
    name = 'Vector correlation'
    meas = 'r'

    def __init__(self, data: RevCorrData):
        self.y_scale = data.scale_data
        Evaluator.__init__(self, data)

    def add_y(self, i, y, y_pred):
        self.x[i] = self._r(y, y_pred)

    def _r(self, y, y_pred):
        # shape (space, time)
        y_norm = norm(y, axis=0)
        y_pred_norm = norm(y_pred, axis=0)
        # rescale y_pred
        y_pred_scale = (y_pred_norm ** 2).mean() ** 0.5
        if y_pred_scale == 0:
            return 0
        y_pred_l2 = y_pred / y_pred_scale
        # rescale y
        y_scale = (y_norm ** 2).mean() ** 0.5
        if y_scale == 0:
            return 0
        elif self.y_scale == 'l2':
            y_l2 = y
        else:
            y_l2 = y / y_scale
        # l2 correlation
        return np.multiply(y_l2, y_pred_l2, out=y_pred_l2).sum(0).mean(0)


class VectorCorrelationL1(VectorCorrelation):
    vector = True
    attr = 'r_l1'
    name = 'Vector l1-correlation'
    meas = 'r'

    def _r(self, y, y_pred):
        # shape (space, time)
        y_norm = norm(y, axis=0)
        y_pred_norm = norm(y_pred, axis=0)
        # rescale y_pred
        y_pred_scale = y_pred_norm.mean()
        if y_pred_scale == 0:
            return 0
        y_pred_l1 = y_pred / y_pred_scale
        # rescale y
        y_scale = y_norm.mean()
        if y_scale == 0:
            return 0
        elif self.y_scale == 'l1':
            y_l1 = y
        else:
            y_l1 = y / y_scale
        # l1 correlation
        # --------------
        # E|X| = 1 --> E√XX = 1
        yy = np.multiply(y_l1, y_pred_l1, out=y_pred_l1).sum(0)
        sign = np.sign(yy)
        np.abs(yy, out=yy)
        yy **= 0.5
        yy *= sign
        return yy.mean()


evaluators = {
    'l1': L1,
    'l2': L2,
    'r': Correlation,
    'r_rank': RankCorrelation,
    'vec-l1': VectorL1,
    'vec-l2': VectorL2,
    'vec-corr': VectorCorrelation,
    'vec-corr-l1': VectorCorrelationL1,
}


def get_evaluator(
        key: str,
        data: RevCorrData,
) -> Evaluator:
    return evaluators[key](data)


def get_evaluators(
        keys: List[str],
        data: RevCorrData,
) -> (List[Evaluator], List[Evaluator], List[Evaluator]):
    evaluators = [get_evaluator(key, data) for key in keys]
    # split into scalar and vector evaluators
    evaluators_s = [e for e in evaluators if not e.vector]
    evaluators_v = [e for e in evaluators if e.vector]
    return evaluators, evaluators_s, evaluators_v
