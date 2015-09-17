# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Statistical Parametric Mapping"""
from operator import mul

import numpy as np
from scipy.linalg import inv

from .._data_obj import (Dataset, Factor, Var, NDVar, asmodel, asndvar,
                         DimensionMismatchError)
from . import opt
from .testnd import ttest_1samp


class LM(object):
    """Fixed effects model for SPM"""
    def __init__(self, y, model, ds=None, coding='dummy', subject=None):
        y = asndvar(y, ds=ds)
        n_cases = len(y)
        model = asmodel(model, None, ds, n_cases)
        p = model._parametrize(coding)
        n_coeff = p.x.shape[1]
        coeffs_flat = np.empty((n_coeff, reduce(mul, y.shape[1:])))
        y_flat = y.x.reshape((n_cases, -1))
        opt.lm_betas(y_flat, p.x, p.projector, coeffs_flat)

        if subject is not None and not isinstance(subject, basestring):
            raise TypeError("subject needs to be None or string, got %s"
                            % repr(subject))

        self.model = model
        self._coeffs_flat = coeffs_flat
        self._p = p
        self._dims = y.dims
        self._shape = y.shape[1:]
        self.subject = subject
        self.n_cases = len(y)

    def _coefficient(self, term):
        """Regression coefficient for a given term"""
        if term in self._p.column_names:
            index = self._p.column_names.index(term)
        elif term in self._p.terms:
            index = self._p.terms[term]
            if index.stop - index.start > 1:
                raise NotImplementedError("Term has more than one column")
        else:
            raise KeyError("Unknown term: %s" % repr(term))
        return self._coeffs_flat[index].reshape((1,) + self._shape)

    def _n_columns(self):
        return {term: s.stop - s.start for term, s in self._p.terms.iteritems()}


class RandomLM(object):
    """Random effects model for SPM"""
    def __init__(self, lms):
        lm0 = lms[0]
        other_lms = lms[1:]

        # dims
        dims = lm0._dims
        if any(lm._dims != dims for lm in other_lms):
            raise DimensionMismatchError("Not all LM instances have same dimensions")

        # terms
        self._n_columns = lm0._n_columns()
        for lm in other_lms:
            if lm._n_columns() != self._n_columns:
                raise ValueError("Model for %s and %s don't match"
                                 % (lm0.subject, lm.subject))
        self.column_names = lm0._p.column_names

        # unique subject labels
        name_i = 0
        subjects = [lm.subject for lm in lms]
        str_names = filter(None, subjects)
        if len(set(str_names)) < len(str_names):
            raise ValueError("Duplicate subject names in %s" % str_names)
        new_name = 'S000'
        for i in xrange(len(subjects)):
            if not subjects[i]:
                while new_name in subjects:
                    name_i += 1
                    new_name = 'S%3i' % name_i
                subjects[i] = new_name

        self._lms = lms
        self.dims = dims

    def _single_column_coefficient(self, term):
        return NDVar(np.concatenate([lm._coefficient(term) for lm in self._lms]),
                     self.dims)

    def column_ttest(self, term, return_data=False, popmean=0, *args, **kwargs):
        """Perform a one-sample t-test on a single model column

        Parameters
        ----------
        term : str
            Name of the term to test.
        return_data : bool
            Return the individual subjects' coefficients along with test
            results.
        popmean : scalar
            Value to compare Y against (default is 0).
        tail : 0 | 1 | -1
            Which tail of the t-distribution to consider:
            0: both (two-tailed);
            1: upper tail (one-tailed);
            -1: lower tail (one-tailed).
        samples : None | int
            Number of samples for permutation cluster test. For None, no
            clusters are formed. Use 0 to compute clusters without performing
            any permutations.
        pmin : None | scalar (0 < pmin < 1)
            Threshold for forming clusters:  use a t-value equivalent to an
            uncorrected p-value.
        tmin : None | scalar
            Threshold for forming clusters.
        tfce : bool
            Use threshold-free cluster enhancement (Smith & Nichols, 2009).
            Default is False.
        tstart, tstop : None | scalar
            Restrict time window for permutation cluster test.
        mintime : scalar
            Minimum duration for clusters (in seconds).
        minsource : int
            Minimum number of sources per cluster.


        Notes
        -----
        Performs a one-sample t-test on coefficient estimates from all subjects
        to test the hypothesis that the coefficient is different from popmean
        in the population.
        """
        ds = Dataset(('coeff', self._single_column_coefficient(term)))
        res = ttest_1samp('coeff', popmean, None, None, ds, *args, **kwargs)
        if return_data:
            ds['subject'] = Factor([lm.subject for lm in self._lms])
            ds['n'] = Var([lm.n_cases for lm in self._lms])
            return res, ds
        else:
            return res

    def _column_ttests(self, *args, **kwargs):
        "precompute all tests"
        out = {}
        for term in self.column_names:
            out[term] = self.column_ttest(term, *args, **kwargs)
        return out
