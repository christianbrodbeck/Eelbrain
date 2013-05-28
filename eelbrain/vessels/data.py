'''
Data Representation
===================

Data is stored in three main vessels:

:class:`factor`:
    stores categorical data
:class:`var`:
    stores numeric data
:class:`ndvar`:
    stores numerical data where each cell contains an array if data (e.g., EEG
    or MEG data)


managed by

    * dataset



'''

from __future__ import division

import collections
import itertools
import cPickle as pickle
import operator
import os
from warnings import warn

import numpy as np
from numpy import dot
import scipy.stats
from scipy.linalg import inv

from eelbrain import fmtxt
from eelbrain import ui
from eelbrain.utils import LazyProperty
from dimensions import find_time_point




preferences = dict(fullrepr=False,  # whether to display full arrays/dicts in __repr__ methods
                   repr_len=5,  # length of repr
                   dataset_str_n_cases=500,
                   var_repr_n_cases=100,
                   factor_repr_n_cases=100,
                   var_repr_fmt='%.3g',
                   factor_repr_use_labels=True,
                   short_repr=True,  # "A % B" vs "interaction(A, B)"
                   )


class DimensionMismatchError(Exception):
    pass



def _effect_eye(n):
    """
    Returns effect coding for n categories. E.g.::

        >>> _effect_eye(4)
        array([[ 1,  0,  0],
               [ 0,  1,  0],
               [ 0,  0,  1],
               [-1, -1, -1]])

    """
    X = np.empty((n, n - 1), dtype=np.int8)
    X[:n - 1] = np.eye(n - 1, dtype=np.int8)
    X[n - 1] = -1
    return X


def _effect_interaction(a, b):
    k = a.shape[1]
    out = [a[:, i, None] * b for i in range(k)]
    return np.hstack(out)



def cellname(cell, delim=' '):
    """
    Returns a consistent ``str`` representation for cells.

    * for factor cells: the cell (str)
    * for interaction cell: delim.join(cell).

    """
    if isinstance(cell, str):
        return cell
    elif isinstance(cell, (list, tuple)):
        return delim.join(cell)
    else:
        return str(cell)


def rank(A, tol=1e-8):
    """
    Rank of a matrix, from
    http://mail.scipy.org/pipermail/numpy-discussion/2008-February/031218.html

    """
    s = np.linalg.svd(A, compute_uv=0)
    return np.sum(np.where(s > tol, 1, 0))


def isbalanced(X):
    """
    returns True if X is balanced, False otherwise.

    X : categorial
        categorial model (factor or interaction)

    """
    if ismodel(X):
        return all(isbalanced(e) for e in X.effects)
    else:
        ns = (np.sum(X == c) for c in X.cells)
        return len(np.unique(ns)) <= 1

def iscategorial(Y):
    "factors as well as interactions are categorial"
    if isfactor(Y):
        return True
    elif isinteraction(Y):
        return Y.is_categorial
    else:
        return False

def isdataobject(Y):
    dataob = ["model", "var", "ndvar", "factor", "interaction", "nonbasic",
              "nested", "list"]
    return hasattr(Y, '_stype_') and  Y._stype_ in dataob

def isdataset(Y):
    return hasattr(Y, '_stype_') and Y._stype_ == 'dataset'

def iseffect(Y):
    effectnames = ["factor", "var", "interaction", "nonbasic", "nested"]
    return hasattr(Y, '_stype_') and  Y._stype_ in effectnames

def isdatalist(Y):
    return isinstance(Y, datalist)

def isfactor(Y):
    return hasattr(Y, '_stype_') and Y._stype_ == "factor"

def isinteraction(Y):
    return hasattr(Y, '_stype_') and Y._stype_ == "interaction"

def ismodel(X):
    return hasattr(X, '_stype_') and X._stype_ == "model"

def isnested(Y):
    return hasattr(Y, '_stype_') and Y._stype_ == "nested"

def isnestedin(item, item2):
    "returns True if item is nested in item2, False otherwise"
    if hasattr(item, 'nestedin'):
        return item.nestedin and (item2 in find_factors(item.nestedin))
    else:
        return False

def isndvar(Y):
    return hasattr(Y, '_stype_') and Y._stype_ == "ndvar"

def isscalar(Y):
    "var, ndvar"
    return hasattr(Y, '_stype_') and Y._stype_ in ["ndvar", "var"]

def isuv(Y):
    "univariate (var, factor)"
    return hasattr(Y, '_stype_') and Y._stype_ in ["factor", "var"]

def isvar(Y):
    return hasattr(Y, '_stype_') and Y._stype_ == "var"


def hasrandom(Y):
    """True if Y is or contains a random effect, False otherwise"""
    if isfactor(Y):
        return Y.random
    elif isinteraction(Y):
        for e in Y.base:
            if isfactor(e) and e.random:
                return True
    elif ismodel(Y):
        return any(map(hasrandom, Y.effects))
    return False


def asmodel(X, sub=None, ds=None):
    if isinstance(X, str):
        if ds is None:
            err = ("Model was specified as string, but no dataset was "
                   "specified")
            raise TypeError(err)
        X = ds.eval(X)

    if ismodel(X):
        pass
    elif isinstance(X, (list, tuple)):
        X = model(*X)
    else:
        X = model(X)

    if sub is not None:
        return X[sub]
    else:
        return X

def asvar(Y, sub=None, ds=None):
    if isinstance(Y, str):
        if ds is None:
            err = ("Var was specified as string, but no dataset was specified")
            raise TypeError(err)
        Y = ds.eval(Y)

    if isvar(Y):
        pass
    else:
        Y = var(Y)

    if sub is not None:
        return Y[sub]
    else:
        return Y

def asfactor(Y, sub=None, ds=None):
    if isinstance(Y, str):
        if ds is None:
            err = ("Factor was specified as string, but no dataset was "
                   "specified")
            raise TypeError(err)
        Y = ds.eval(Y)

    if isfactor(Y):
        pass
    elif isvar(Y):
        Y = Y.as_factor()
    else:
        Y = factor(Y)

    if sub is not None:
        return Y[sub]
    else:
        return Y

def asndvar(Y, sub=None, ds=None):
    if isinstance(Y, str):
        if ds is None:
            err = ("Ndvar was specified as string, but no dataset was "
                   "specified")
            raise TypeError(err)
        Y = ds.eval(Y)

    if not isndvar(Y):
        raise TypeError("ndvar required")

    if sub is not None:
        return Y[sub]
    else:
        return Y

def ascategorial(Y, sub=None, ds=None):
    if isinstance(Y, str):
        if ds is None:
            err = ("Parameter was specified as string, but no dataset was "
                   "specified")
            raise TypeError(err)
        Y = ds.eval(Y)

    if iscategorial(Y):
        pass
    else:
        Y = asfactor(Y)

    if sub is not None:
        return Y[sub]
    else:
        return Y


def _empty_like(obj, n=None, name=None):
    "Create an empty object of the same type as obj"
    n = n or len(obj)
    name = name or obj.name
    if isfactor(obj):
        return factor([''], rep=n, name=name)
    elif isvar(obj):
        return var(np.empty(n) * np.NaN, name=name)
    elif isndvar(obj):
        shape = (n,) + obj.shape[1:]
        return ndvar(np.empty(shape) * np.NaN, dims=obj.dims, name=name)
    elif isdatalist(obj):
        return datalist([None] * n, name=name)
    else:
        err = "Type not supported: %s" % type(obj)


def cell_label(cell, delim=' '):
    if isinstance(cell, tuple):
        return delim.join(cell)
    else:
        return cell


def align(d1, d2, out='data', i1='index', i2='index'):
    """
    Aligns two data-objects d1 and d2 based on two index variables, i1 and i2.

    Before aligning, d1 and d2 describe the same cases, but their order does
    not correspond. Align uses the indexes (i1 and i2) to match each case in
    d2 to a case in d1 (i.e., d1 is used as the basis for the case order).
    Cases that are not present in both d1 and d2 are dropped.


    Parameters
    ----------

    d1, d2 : data-object
        Two data objects which are to be aligned
    i1, i2 : str | array-like (dtype=int)
        Indexes for cases in d1 and d2.
        If d1 and d2 are datasets, i1 and i2 can be keys for variables in d1 and
        d2. If d1 an d2 are other data objects, i1 and i2 have to be actual indices
        (array-like)
    out : 'data' | 'index'
        **'data'**: returns the two aligned data objects. **'index'**: returns two
        indices index1 and index2 which can be used to align the datasets with
        ``ds1[index1]; ds2[index2]``.


    Examples
    --------

    see examples/datasets/align.py

    """
    i1 = asvar(i1, ds=d1)
    i2 = asvar(i2, ds=d2)

    if len(i1) > len(i1.values):
        raise ValueError('Non-unique index in i1 for %r' % d1.name)
    if len(i2) > len(i2.values):
        raise ValueError('Non-unique index in i2 for %r' % d2.name)

    idx1 = []
    idx2 = []
    for i, idx in enumerate(i1):
        if idx in i2:
            idx1.append(i)
            where2 = i2.index(idx)[0]
            idx2.append(where2)

    if out == 'data':
        return d1[idx1], d2[idx2]
    elif out == 'index':
        return idx1, idx2
    else:
        raise ValueError("Invalid value for out parameter: %r" % out)


def align1(d, idx, d_idx='index', out='data'):
    """
    Align a data object to an index

    Parameters
    ----------
    d : data object, n_cases = n1
        Data object with cases that should be aligned to idx.
    idx : index array, len = n2
        index to which d should be aligned.
    d_idx : str | index array, len = n1
        Indices of cases in d. If d is a dataset, d_idx can be a name in d.
    out : 'data' | 'index' | 'bool'
        Return a subset of d, an array of numerical indices into d, or a
        boolean array into d.
    """
    idx = asvar(idx)
    d_idx = asvar(d_idx, ds=d)

    where = np.in1d(d_idx, idx, True)
    if out == 'bool':
        return where
    elif out == 'index':
        return np.nonzero(where)
    elif out == 'data':
        return d[where]
    else:
        ValueError("Invalid value for out parameter: %r" % out)


def combine(items):
    """
    combine a list of items of the same type into one item

    Parameters
    ----------
    items : collection
        Collection (:py:class:`list`, :py:class:`tuple`, ...) of data objects
        of a single type (dataset, var, factor, ndvar or datalist).

    Notes
    -----
    For vars and factors, the :attr:`name` attribute of the combined item is
    the name of the first item.

    For datasets, missing variables are filled in with empty values ('', NaN).

    """
    item0 = items[0]
    if isdataset(item0):
        # find all keys and data types
        keys = item0.keys()
        sample = dict(item0)
        for item in items:
            for key in item.keys():
                if key not in keys:
                    keys.append(key)
                    sample[key] = item[key]
        # create new dataset
        out = dataset()
        for key in keys:
            pieces = [ds[key] if key in ds else
                      _empty_like(sample[key], ds.n_cases) for ds in items]
            out[key] = combine(pieces)
        return out
    elif isvar(item0):
        x = np.hstack(i.x for i in items)
        return var(x, name=item0.name)
    elif isfactor(item0):
        if all(f._labels == item0._labels for f in items[1:]):
            x = np.hstack(f.x for f in items)
            kwargs = item0._child_kwargs()
        else:
            x = sum((i.as_labels() for i in items), [])
            kwargs = dict(name=item0.name,
                          random=item0.random)
        return factor(x, **kwargs)
    elif isndvar(item0):
        dims = item0.dims
        has_case = np.array([v.has_case for v in items])
        if all(has_case):
            x = np.concatenate([v.x for v in items], axis=0)
        elif all(has_case == False):
            x = np.array([v.x for v in items])
            dims = ('case',) + dims
        else:
            err = ("Some items have a 'case' dimension, others do not")
            raise DimensionMismatchError(err)
        return ndvar(x, dims=dims, name=item0.name, properties=item0.properties)
    elif isinstance(item0, datalist):
        return sum(items[1:], item0)
    else:
        err = ("Objects of type %s can not be combined." % type(item0))
        raise TypeError(err)



def find_factors(obj):
    "returns a list of all factors contained in obj"
    if isuv(obj):
        return effect_list([obj])
    elif ismodel(obj):
        f = set()
        for e in obj.effects:
            f.update(find_factors(e))
        return effect_list(f)
    elif isnested(obj):
        return find_factors(obj.effect)
    elif isinteraction(obj):
        return obj.base
    else:  # nonbasic_effect
        try:
            return effect_list(obj.factors)
        except:
            raise TypeError("%r has no factors" % obj)


class effect_list(list):
    def __repr__(self):
        return 'effect_list((%s))' % ', '.join(self.names())

    def __contains__(self, item):
        for f in self:
            if (len(f) == len(item)) and np.all(item == f):
                return True
        return False

    def index(self, item):
        for i, f in enumerate(self):
            if (len(f) == len(item)) and np.all(item == f):
                return i
        raise ValueError("factor %r not in effect_list" % item.name)

    def names(self):
        return [e.name if isuv(e) else repr(e) for e in self]



class var(object):
    """
    Container for scalar data.

    While :py:class:`var` objects support a few basic operations in a
    :py:mod:`numpy`-like fashion (``+``, ``-``, ``*``, ``/``, ``//``), their
    :py:attr:`var.x` attribute provides access to the corresponding
    :py:class:`numpy.array` which can be used for anything more complicated.
    :py:attr:`var.x` can be read and modified, but should not be replaced.

    """
    _stype_ = "var"
    def __init__(self, x, name=None):
        """
        Parameters
        ----------
        x : array-like
            Data; is converted with ``np.asarray(x)``
        name : str | None
            Name of the variable

        """
        x = np.asarray(x)
        if x.ndim > 1:
            raise ValueError("Use ndvar class for data with more than one dimension")
        self.__setstate__((x, name))

    def __setstate__(self, state):
        x, name = state
        # raw
        self.name = name
        self.x = x
        # constants
        self._n_cases = len(x)
        self.df = 1
        self.random = False

    def __getstate__(self):
        return (self.x, self.name)

    def __repr__(self, full=False):
        n_cases = preferences['var_repr_n_cases']

        if self.x.dtype == bool:
            fmt = '%r'
        else:
            fmt = preferences['var_repr_fmt']

        if full or len(self.x) <= n_cases:
            x = [fmt % v for v in self.x]
        else:
            x = [fmt % v for v in self.x[:n_cases]]
            x.append('<... N=%s>' % len(self.x))

        args = ['[%s]' % ', '.join(x)]
        if self.name is not None:
            args.append('name=%r' % self.name)

        return "var(%s)" % ', '.join(args)

    def __str__(self):
        return self.__repr__(True)

    # container ---
    def __len__(self):
        return self._n_cases

    def __getitem__(self, index):
        "if factor: return new variable with mean values per factor category"
        if isfactor(index):
            f = index
            x = []
            for v in np.unique(f.x):
                x.append(np.mean(self.x[f == v]))
            return var(x, self.name)
        elif isvar(index):
            index = index.x

        x = self.x[index]
        if np.iterable(x):
            return var(x, self.name)
        else:
            return x

    def __setitem__(self, index, value):
        self.x[index] = value

    def __contains__(self, value):
        return value in self.x

    # numeric ---
    def __add__(self, other):
        if isdataobject(other):
            # ??? should var + var return sum or model?
            return model(self, other)
        else:
            x = self.x + other
            if np.isscalar(other):
                name = '%s+%s' % (self.name, other)
            else:
                name = self.name

            return var(x, name=name)

    def __sub__(self, other):
        "subtract: values are assumed to be ordered. Otherwise use .sub method."
        if np.isscalar(other):
            return var(self.x - other,
                       name='%s-%s' % (self.name, other))
        else:
            assert len(other.x) == len(self.x)
            x = self.x - other.x
            n1, n2 = self.name, other.name
            if n1 == n2:
                name = n1
            else:
                name = "%s-%s" % (n1, n2)
            return var(x, name)

    def __mul__(self, other):
        if iscategorial(other):
            return model(self, other, self % other)
        elif isvar(other):
            x = self.x * other.x
            name = '%s*%s' % (self.name, other.name)
        else:  #  np.isscalar(other)
            x = self.x * other
            other_name = str(other)
            if len(other_name) < 12:
                name = '%s*%s' % (self.name, other_name)
            else:
                name = self.name

        return var(x, name=name)

    def __floordiv__(self, other):
        if isvar(other):
            x = self.x // other.x
            name = '%s//%s' % (self.name, other.name)
        elif np.isscalar(other):
            x = self.x // other
            name = '%s//%s' % (self.name, other)
        else:
            x = self.x // other
            name = '%s//%s' % (self.name, '?')
        return var(x, name=name)

    def __mod__(self, other):
        if  ismodel(other):
            return model(self) % other
        elif isdataobject(other):
            return interaction((self, other))
        elif isvar(other):
            other = other.x
            other_name = other.name
        else:
            other_name = str(other)[:10]

        name = '{name}%{other}'
        name = name.format(name=self.name, other=other_name)
        return var(self.x % other, name=name)

    def __lt__(self, y):
        return self.x < y

    def __le__(self, y):
        return self.x <= y

    def __eq__(self, y):
        return self.x == y

    def __ne__(self, y):
        return self.x != y

    def __gt__(self, y):
        return self.x > y

    def __ge__(self, y):
        return self.x >= y

    def __truediv__(self, other):
        return self.__div__(other)

    def __div__(self, other):
        """
        type of other:
        scalar:
            returns var divided by other
        factor:
            returns a separate slope for each level of the factor; needed for
            ANCOVA

        """
        if np.isscalar(other):
            return var(self.x / other,
                       name='%s/%s' % (self.name, other))
        elif isvar(other):
            return var(self.x / other.x,
                       name='%s/%s' % (self.name, other.name))
        else:
            categories = other
            if not hasattr(categories, 'as_dummy_complete'):
                raise NotImplementedError
            dummy_factor = categories.as_dummy_complete
            codes = dummy_factor * self.as_effects
            # center
            means = codes.sum(0) / dummy_factor.sum(0)
            codes -= dummy_factor * means
            # create effect
            name = '%s per %s' % (self.name, categories.name)
            labels = categories.dummy_complete_labels
            out = nonbasic_effect(codes, [self, categories], name,
                                  beta_labels=labels)
            return out

    @property
    def as_effects(self):
        "for effect initialization"
        return self.centered()[:, None]

    def as_factor(self, name=None, labels='%r'):
        """
        convert the var into a factor

        :arg name: if None, it will copy the var name
        :arg labels: dictionary maping values->labels, or format string for
            converting values into labels

        """
        if name is None:
            name = self.name

        if type(labels) is not dict:
            fmt = labels
            labels = {}
            for value in np.unique(self.x):
                labels[value] = fmt % value

        f = factor(self.x, name=name, labels=labels)
        return f

    def centered(self):
        return self.x - self.x.mean()

    def copy(self, name='{name}'):
        "returns a deep copy of itself"
        x = self.x.copy()
        name = name.format(name=self.name)
        return var(x, name=name)

    def compress(self, X, func=np.mean, name='{name}'):
        """
        X: factor or interaction; returns a compressed var object with one
        value for each cell in X.

        """
        if len(X) != len(self):
            err = "Length mismatch: %i (var) != %i (X)" % (len(self), len(X))
            raise ValueError(err)

        x = []
        for cell in X.cells:
            x_cell = self.x[X == cell]
            if len(x_cell) > 0:
                x.append(func(x_cell))

        x = np.array(x)
        name = name.format(name=self.name)
        out = var(x, name=name)
        return out

    @property
    def beta_labels(self):
        return [self.name]

    def diff(self, X, v1, v2, match):
        """
        subtracts X==v2 from X==v1; sorts values in ascending order according
        to match

        """
        raise NotImplementedError
        # FIXME: use celltable
        assert isfactor(X)
        I1 = (X == v1);         I2 = (X == v2)
        Y1 = self[I1];          Y2 = self[I2]
        m1 = match[I1];         m2 = match[I2]
        s1 = np.argsort(m1);    s2 = np.argsort(m2)
        y = Y1[s1] - Y2[s2]
        name = "{n}({x1}-{x2})".format(n=self.name,
                                       x1=X.cells[v1],
                                       x2=X.cells[v2])
        return var(y, name)

    def index(self, *values):
        """
        v.index(*values) returns an array of indices of where v has one of
        *values.

        """
        idx = []
        for v in values:
            where = np.where(self == v)[0]
            idx.extend(where)
        return sorted(idx)

    def isany(self, *values):
        return np.any([self.x == v for v in values], axis=0)

    def isin(self, values):
        return np.any([self.x == v for v in values], axis=0)

    def isnot(self, *values):
        return np.all([self.x != v for v in values], axis=0)

    def mean(self):
        return self.x.mean()

    def repeat(self, repeats, name='{name}'):
        "Analogous to :py:func:`numpy.repeat`"
        return var(self.x.repeat(repeats), name=name.format(name=self.name))

    @property
    def values(self):
        return np.unique(self.x)



class _effect_(object):
    # numeric ---
    def __add__(self, other):
        return model(self) + other

    def __mul__(self, other):
        return model(self, other, self % other)

    def __mod__(self, other):
        return interaction((self, other))

    def index(self, cell):
        "``e.index(cell)`` returns an array of indices where e equals cell"
        return np.nonzero(self == cell)[0]



class factor(_effect_):
    """
    Container for categorial data.

    """
    _stype_ = "factor"
    def __init__(self, x, name=None, random=False, rep=1, tile=1, labels={}):
        """
        Parameters
        ----------
        x : iterator
            Sequence of factor values (see also the ``labels`` kwarg).
        name : str
            Name of the factor.
        random : bool
            Treat factor as random factor (for ANOVA; default is False).
        rep : int
            Repeat each element in ``x`` ``rep`` many times.
        tile : int
            Repeat x as a whole ``tile`` many times.
        labels : dict or None
            If provided, these labels are used to replace values in x when
            constructing the labels dictionary. All labels for values of
            x not in ``labels`` are constructed using ``str(value)``.


        Examples
        --------
        The most obvious way to initialize a factor is a list of strings::

            >>> factor(['in', 'in', 'in', 'out', 'out', 'out'])
            factor(['in', 'in', 'in', 'out', 'out', 'out'])

        The same can be achieved with a list of integers plus a labels dict::

            >>> factor([1, 1, 1, 0, 0, 0], labels={1: 'in', 0: 'out'})
            factor(['in', 'in', 'in', 'out', 'out', 'out'])

        Since the factor initialization simply iterates over the ``x``
        argument, a factor with one-character codes can also be initialized
        with a single string::

            >>> factor('iiiooo')
            factor(['i', 'i', 'i', 'o', 'o', 'o'])

        """
        state = {'name': name, 'random': random}
        labels_ = state['labels'] = {}  # {code -> label}

        try:
            n_cases = len(x)
        except TypeError:  # for generators:
            x = tuple(x)
            n_cases = len(x)

        # convert x to codes
        codes = {}  # {label -> code}
        x_ = np.empty(n_cases, dtype=np.uint16)
        for i, value in enumerate(x):
            label = labels.get(value, value)
            if not isinstance(label, unicode):
                label = str(label)
            if label in codes:
                code = codes.get(label)
            else:  # new code
                code = max(labels_) + 1 if labels_ else 0
                labels_[code] = label
                codes[label] = code

            x_[i] = code

        if rep > 1:
            x_ = x_.repeat(rep)

        if tile > 1:
            x_ = np.tile(x_, tile)

        state['x'] = x_
        self.__setstate__(state)

    def __setstate__(self, state):
        self.x = x = state['x']
        self.name = state['name']
        self.random = state['random']
        self._labels = labels = state['labels']
        self._codes = {lbl: code for code, lbl in labels.iteritems()}
        self._n_cases = len(x)

    def __getstate__(self):
        state = {'x': self.x,
                 'name': self.name,
                 'random': self.random,
                 'labels': self._labels}
        return state

    def __repr__(self, full=False):
        use_labels = preferences['factor_repr_use_labels']
        n_cases = preferences['factor_repr_n_cases']

        if use_labels:
            values = self.as_labels()
        else:
            values = self.x.tolist()

        if full or len(self.x) <= n_cases:
            x = repr(values)
        else:
            x = [repr(v) for v in values[:n_cases]]
            x.append('<... N=%s>' % len(self.x))
            x = '[' + ', '.join(x) + ']'

        args = [x]

        if self.name is not None:
            args.append('name=%r' % self.name)

        if self.random:
            args.append('random=True')

        if not use_labels:
            args.append('labels=%s' % self._labels)

        return 'factor(%s)' % ', '.join(args)

    def __str__(self):
        return self.__repr__(True)

    # container ---
    def __len__(self):
        return self._n_cases

    def __getitem__(self, index):
        """
        sub needs to be int or an array of bools of shape(self.x)
        this method is valid for factors and nonbasic effects

        """
        if isvar(index):
            index = index.x

        x = self.x[index]
        if np.iterable(x):
            return factor(x, **self._child_kwargs())
        else:
            return self._labels[x]

    def __setitem__(self, index, x):
        # convert x to code
        if isinstance(x, basestring):
            code = self._get_code(x)
        elif np.iterable(x):
            code = np.empty(len(x), dtype=np.uint16)
            for i, v in enumerate(x):
                code[i] = self._get_code(v)

        # assign
        self.x[index] = code

        # obliterate redundant labels
        codes_in_use = set(np.unique(self.x))
        rm = set(self._labels) - codes_in_use
        for code in rm:
            label = self._labels.pop(code)
            del self._codes[label]

    def _get_code(self, label):
        "add the label if it does not exists and return its code"
        try:
            return self._codes[label]
        except KeyError:
            code = 0
            while code in self._labels:
                code += 1

            if code >= 65535:
                raise ValueError("Too many categories in this factor.")

            self._labels[code] = label
            self._codes[label] = code
            return code

    def __iter__(self):
        return (self._labels[i] for i in self.x)

    def __contains__(self, value):
        try:
            code = self._codes[value]
        except KeyError:
            return False
        return code in self.x

    # numeric ---
    def __eq__(self, other):
        return self.x == self._encode_(other)

    def __ne__(self, other):
        return self.x != self._encode_(other)

    def _encode_(self, Y):
        if isinstance(Y, basestring):
            return self._codes.get(Y, -1)
        else:
            out = np.empty(len(Y), dtype=self.x.dtype)
            for i, v in enumerate(Y):
                out[i] = self._codes.get(v, -1)
            return out

    def __call__(self, other):
        """
        Create a nested effect. A factor A is nested in another factor B if
        each level of A only occurs together with one level of B.

        """
        return nested_effect(self, other)

    def _child_kwargs(self, name='{name}'):
        kwargs = dict(labels=self._labels,
                      name=name.format(name=self.name),
                      random=self.random)
        return kwargs

    def _interpret_y(self, Y, create=False):
        """
        Parameters
        ----------
        Y : str | list of str
            String(s) to be converted to code values.

        Returns
        -------
        codes : int | list of int
            List of values (codes) corresponding to the categories.

        """
        if isinstance(Y, basestring):
            if Y in self._codes:
                return self._codes[Y]
            elif create:
                code = 0
                while code in self._labels:
                    code += 1
                if code >= 65535:
                    raise ValueError("Too many categories in this factor.")
                self._labels[code] = Y
                self._codes[Y] = code
                return code
            else:
                return 65535  # code for values not present in the factor
        elif np.iterable(Y):
            out = np.empty(len(Y), dtype=np.uint16)
            for i, y in enumerate(Y):
                out[i] = self._interpret_y(y, create=create)
            return out
        elif Y in self._labels:
            return Y
        else:
            raise ValueError("unknown cell: %r" % Y)

    @property
    def as_dummy(self):  # x_dummy_coded
        shape = (self._n_cases, self.df)
        codes = np.empty(shape, dtype=np.int8)
        for i, cell in enumerate(self.cells[:-1]):
            codes[:, i] = (self == cell)

        return codes

    @property
    def as_dummy_complete(self):
        x = self.x[:, None]
        categories = np.unique(x)
        codes = np.hstack([x == cat for cat in categories])
        return codes.astype(np.int8)

    @property
    def as_effects(self):  # x_deviation_coded
        shape = (self._n_cases, self.df)
        codes = np.empty(shape, dtype=np.int8)
        for i, cell in enumerate(self.cells[:-1]):
            codes[:, i] = (self == cell)

        contrast = (self == self.cells[-1])
        codes -= contrast[:, None]
        return codes

    def as_labels(self):
        return [self._labels[v] for v in self.x]

    @property
    def beta_labels(self):
        cells = self.cells
        txt = '{0}=={1}'
        return [txt.format(cells[i], cells[-1]) for i in range(len(cells) - 1)]

    @property
    def cells(self):
        return sorted(self._labels.values())

    def compress(self, X, name='{name}'):
        """
        Summarize the factor by collapsing within cells in `X`.

        Raises an error if there are cells that contain more than one value.

        Parameters
        ----------
        X : categorial
            A categorial model defining cells to collapse.

        Returns
        -------
        f : factor
            A copy of self with only one value for each cell in X

        """
        if len(X) != len(self):
            err = "Length mismatch: %i (var) != %i (X)" % (len(self), len(X))
            raise ValueError(err)

        x = []
        for cell in X.cells:
            idx = (X == cell)
            if np.sum(idx):
                x_i = np.unique(self.x[idx])
                if len(x_i) > 1:
                    err = ("ambiguous cell: factor %r has multiple values for "
                           "cell %r" % (self.name, cell))
                    raise ValueError(err)
                else:
                    x.append(x_i[0])

        x = np.array(x)
        name = name.format(name=self.name)
        out = factor(x, name=name, labels=self._labels, random=self.random)
        return out

    def copy(self, name='{name}', rep=1, tile=1):
        "returns a deep copy of itself"
        f = factor(self.x.copy(), rep=rep, tile=tile,
                   **self._child_kwargs(name))
        return f

    @property
    def df(self):
        return max(0, len(self._labels) - 1)

    def get_index_to_match(self, other):
        """
        Assuming that ``other`` is a shuffled version of self, this method
        returns ``index`` to transform from the order of self to the order of
        ``other``. To guarantee exact matching, each value can only occur once
        in self.

        Example::

            >>> index = factor1.get_index_to_match(factor2)
            >>> all(factor1[index] == factor2)
            True

        """
        assert self._labels == other._labels
        index = []
        for v in other.x:
            where = np.where(self.x == v)[0]
            if len(where) == 1:
                index.append(where[0])
            else:
                msg = "%r contains several cases of %r" % (self, v)
                raise ValueError(msg)
        return np.array(index)

    def isany(self, *values):
        """
        Returns a boolean array that is True where the factor matches
        one of the ``values``

        Example::

            >>> a = factor('aabbcc')
            >>> b.isany('b', 'c')
            array([False, False,  True,  True,  True,  True], dtype=bool)

        """
        return self.isin(values)

    def isin(self, values):
        is_v = [self.x == self._codes.get(v, np.nan) for v in values]
        return np.any(is_v, 0)

    def isnot(self, *values):
        """
        returns a boolean array that is True where the data does not equal any
        of the values

        """
        is_not_v = [self.x != self._codes.get(v, np.nan) for v in values]
        if is_not_v:
            return np.all(is_not_v, axis=0)
        else:
            return np.ones(len(self), dtype=bool)

    def table_categories(self):
        "returns a table containing information about categories"
        table = fmtxt.Table('rll')
        table.title(self.name)
        for title in ['i', 'Label', 'n']:
            table.cell(title)
        table.midrule()
        for code, label in self._labels.iteritems():
            table.cell(code)
            table.cell(label)
            table.cell(np.sum(self.x == code))
        return table

    def project(self, target, name='{name}'):
        """
        Project the factor onto an index array ``target``

        Example::

            >>> f = V.data.factor('abc')
            >>> f.as_labels()
            ['a', 'b', 'c']
            >>> fp = f.project([1,2,1,2,0,0])
            >>> fp.as_labels()
            ['b', 'c', 'b', 'c', 'a', 'a']

        """
        if isvar(target):
            target = target.x
        x = self.x[target]
        return factor(x, **self._child_kwargs(name))

    def repeat(self, repeats, name='{name}'):
        "Repeat elements of a factor (analogous to :py:func:`numpy.repeat`)"
        return factor(self.x.repeat(repeats), **self._child_kwargs(name))



class ndvar(object):
    "Container for n-dimensional data."
    _stype_ = "ndvar"
    def __init__(self, x, dims=('case',), properties=None, name=None):
        """
        Parameters
        ----------

        x : array
            The data
        dims : tuple
            The dimensions characterizing the axes of the data. If present,
            ``'case'`` should be provided as a :py:class:`str`, and should
            always occupy the first position.
        properties : dict
            A dictionary with data properties.


        Notes
        -----

        ``x`` and ``dims`` are stored without copying. A shallow
        copy of ``properties`` is stored. Make sure the relevant objects
        are not modified externally later.


        Examples
        --------

        Importing 600 epochs of data for 80 time points:

            >>> time = var('time', range(-.2, .6, .01))
            >>> dims = ('case', time)
            >>> data.shape
            (600, 80)
            >>> Y = ndvar(data, dims=dims)

        """
        # check data shape
        dims = tuple(dims)
        ndim = len(dims)
        if ndim != x.ndim:
            err = ("Unequal number of dimensions (data: %i, dims: %i)" %
                   (x.ndim, ndim))
            raise DimensionMismatchError(err)

        # check dimensions
        d0 = dims[0]
        if isinstance(d0, basestring):
            if d0 == 'case':
                has_case = True
            else:
                err = ("String dimension needs to be 'case' (got %r)" % d0)
                raise ValueError(err)
        else:
            has_case = False

        for dim, n in zip(dims, x.shape)[has_case:]:
            if isinstance(dim, basestring):
                err = ("Invalid dimension: %r in %r. First dimension can be "
                       "'case', other dimensions need to be array-like" %
                       (dim, dims))
                raise TypeError(err)
            n_dim = len(dim)
            if n_dim != n:
                err = ("Dimension %r length mismatch: %i in data, "
                       "%i in dimension %r" % (dim.name, n, n_dim, dim.name))
                raise DimensionMismatchError(err)

        state = {'dims': dims,
                 'x': x,
                 'name': name}

        # store attributes
        if properties is None:
            state['properties'] = {}
        else:
            state['properties'] = properties.copy()

        self.__setstate__(state)

    def __setstate__(self, state):
        self.dims = dims = state['dims']
        self.has_case = (dims[0] == 'case')
        self._truedims = truedims = dims[self.has_case:]

        # dimnames
        self.dimnames = tuple(dim.name for dim in truedims)
        if self.has_case:
            self.dimnames = ('case',) + self.dimnames

        self.x = x = state['x']
        self.name = state['name']
        self.properties = state['properties']
        # derived
        self.ndim = len(dims)
        self._len = len(x)
        self._dim_2_ax = dict(zip(self.dimnames, xrange(self.ndim)))
        # attr
        for dim in truedims:
            if hasattr(self, dim.name):
                err = ("invalid dimension name: %r (already present as ndvar"
                       " attr)" % dim.name)
                raise ValueError(err)
            else:
                setattr(self, dim.name, dim)

    def __getstate__(self):
        state = {'dims': self.dims,
                 'x': self.x,
                 'name': self.name,
                 'properties': self.properties}
        return state

    # numeric ---
    def _align(self, other):
        "align data from 2 ndvars"
        if isvar(other):
            return self.dims, self.x, self._ialign(other)
        elif isndvar(other):
            i_self = list(self.dimnames)
            dims = list(self.dims)
            i_other = []

            for dim in i_self:
                if dim in other.dimnames:
                    i_other.append(dim)
                else:
                    i_other.append(None)

            for dim in other.dimnames:
                if dim in i_self:
                    pass
                else:
                    i_self.append(None)
                    i_other.append(dim)
                    dims.append(other.get_dim(dim))

            x_self = self.get_data(i_self)
            x_other = other.get_data(i_other)
            return dims, x_self, x_other
        else:
            raise TypeError

    def _ialign(self, other):
        "align for self-modifying operations (+=, ...)"
        if isvar(other):
            assert self.has_case
            n = len(other)
            shape = (n,) + (1,) * (self.x.ndim - 1)
            return other.x.reshape(shape)
        elif isndvar(other):
            assert all(dim in self.dimnames for dim in other.dimnames)
            i_other = []
            for dim in self.dimnames:
                if dim in other.dimnames:
                    i_other.append(dim)
                else:
                    i_other.append(None)
            return other.get_data(i_other)
        else:
            raise TypeError

    def __add__(self, other):
        if isscalar(other):
            dims, x_self, x_other = self._align(other)
            x = x_self + x_other
            name = '%s+%s' % (self.name, other.name)
        elif np.isscalar(other):
            x = self.x + other
            dims = self.dims
            name = '%s+%s' % (self.name, str(other))
        else:
            raise ValueError("can't add %r" % other)
        return ndvar(x, dims=dims, name=name, properties=self.properties)

    def __iadd__(self, other):
        self.x += self._ialign(other)
        return self

    def __sub__(self, other):  # TODO: use dims
        if isscalar(other):
            dims, x_self, x_other = self._align(other)
            x = x_self - x_other
            name = '%s-%s' % (self.name, other.name)
        elif np.isscalar(other):
            x = self.x - other
            dims = self.dims
            name = '%s-%s' % (self.name, str(other))
        else:
            raise ValueError("can't subtract %r" % other)
        return ndvar(x, dims=dims, name=name, properties=self.properties)

    def __isub__(self, other):
        self.x -= self._ialign(other)
        return self

    def __rsub__(self, other):
        x = other - self.x
        return ndvar(x, self.dims, self.properties, name=self.name)

    # container ---
    def __getitem__(self, index):
        if isvar(index):
            index = index.x

        if np.iterable(index) or isinstance(index, slice):
            x = self.x[index]
            if x.shape[1:] != self.x.shape[1:]:
                raise NotImplementedError("Use subdata method when dims are affected")
            return ndvar(x, dims=self.dims, name=self.name, properties=self.properties)
        else:
            index = int(index)
            x = self.x[index]
            dims = self.dims[1:]
            name = '%s_%i' % (self.name, index)
            return ndvar(x, dims=dims, name=name, properties=self.properties)

    def __len__(self):
        return self._len

    def __repr__(self):
        rep = '<ndvar %(name)r: %(dims)s>'
        if self.has_case:
            dims = [(self._len, 'case')]
        else:
            dims = []
        dims.extend([(len(dim), dim.name) for dim in self._truedims])

        dims = ' X '.join('%i (%r)' % fmt for fmt in dims)
        args = dict(name=self.name, dims=dims)
        return rep % args

    def assert_dims(self, dims):
        if self.dimnames != dims:
            err = "Dimensions of %r do not match %r" % (self, dims)
            raise DimensionMismatchError(err)

    def compress(self, X, func=np.mean, name='{name}'):
        """
        Return an ndvar with one case for each cell in ``X``.

        Parameters
        ----------

        X : categorial
            Categorial whose cells are used to compress the ndvar.
        func : function with axis argument
            Function that is used to create a summary of the cases falling
            into each cell of X. The function needs to accept the data as
            first argument and ``axis`` as keyword-argument. Default is
            ``numpy.mean``.
        name : str
            Name for the resulting ndvar. ``'{name}'`` is formatted to the
            current ndvar's ``.name``.

        """
        if not self.has_case:
            raise DimensionMismatchError("%r has no case dimension" % self)
        if len(X) != len(self):
            err = "Length mismatch: %i (var) != %i (X)" % (len(self), len(X))
            raise ValueError(err)

        x = []
        for cell in X.cells:
            idx = (X == cell)
            if np.sum(idx):
                x_cell = self.x[idx]
                x.append(func(x_cell, axis=0))

        # update properties for summary
        properties = self.properties.copy()
        for key in self.properties:
            if key.startswith('summary_') and (key != 'summary_func'):
                properties[key[8:]] = properties.pop(key)

        x = np.array(x)
        name = name.format(name=self.name)
        out = ndvar(x, self.dims, properties=properties, name=name)
        return out

    def copy(self, name='{name}'):
        "returns a deep copy of itself"
        x = self.x.copy()
        name = name.format(name=self.name)
        properties = self.properties.copy()
        return self.__class__(x, dims=self.dims, name=name,
                              properties=properties)

    def get_axis(self, dim):
        return self._dim_2_ax[dim]

    def get_data(self, dims):
        """
        returns the data with a specific ordering of dimension as indicated in
        ``dims``.

        Parameters
        ----------
        dims : sequence of str and None
            List of dimension names. The array that is returned will have axes
            in this order. None can be used to increase the insert a dimension
            with size 1.

        """
        if set(dims).difference([None]) != set(self.dimnames):
            err = "Requested dimensions %r from %r" % (dims, self)
            raise DimensionMismatchError(err)

        dimnames = list(self.dimnames)
        x = self.x

        index = []
        dim_seq = []
        for dim in dims:
            if dim is None:
                index.append(None)
            else:
                index.append(slice(None))
                dim_seq.append(dim)

        for i_tgt, dim in enumerate(dim_seq):
            i_src = dimnames.index(dim)
            if i_tgt != i_src:
                x = x.swapaxes(i_src, i_tgt)
                dimnames[i_src], dimnames[i_tgt] = dimnames[i_tgt], dimnames[i_src]

        return x[index]

    def get_dim(self, name):
        "Returns the dimension var named ``name``"
        if self.has_dim(name):
            i = self._dim_2_ax[name]
            return self.dims[i]
        elif name == 'epoch':
            return var(np.arange(len(self)), 'epoch')
        else:
            msg = "%r has no dimension named %r" % (self, name)
            raise DimensionMismatchError(msg)

    def get_dims(self, names):
        "Returns a tuple with the requested dimension vars"
        return tuple(self.get_dim(name) for name in names)

    def has_dim(self, name):
        return name in self._dim_2_ax

    def repeat(self, repeats, dim='case', name='{name}'):
        """
        Analogous to :py:func:`numpy.repeat`

        """
        ax = self.get_axis(dim)
        x = self.x.repeat(repeats, axis=ax)

        repdim = self.dims[ax]
        if not isinstance(repdim, str):
            repdim = repdim.repeat(repeats)

        dims = self.dims[:ax] + (repdim,) + self.dims[ax + 1:]
        properties = self.properties.copy()
        name = name.format(name=self.name)
        return ndvar(x, dims, properties=properties, name=name)

    def summary(self, *dims, **regions):
        r"""
        Returns a new ndvar with specified dimensions collapsed.

        .. warning::
            Data is collapsed over the different dimensions in turn using the
            provided function with an axis argument. For certain functions
            this is not equivalent to collapsing over several axes concurrently
            (e.g., np.var).

        dimension:
            A whole dimension is specified as string argument. This
            dimension is collapsed over the whole range.
        range:
            A range within a dimension is specified through a keyword-argument.
            Only the data in the specified range is included. Use like the
            :py:meth:`.subdata` method.


        **additional kwargs:**

        func : callable
            Function used to collapse the data. Needs to accept an "axis"
            kwarg (default: np.mean)
        name : str
            Name for the new ndvar. Default: "{func}({name})".


        Examples
        --------

        Assuming UTS is a normal time series. Get the average in a time
        window::

            >>> Y = UTS.summary(time=(.1, .2))

        Get the peak in a time window::

            >>> Y = UTS.summary(time=(.1, .2), func=np.max)

        Assuming MEG is an ndvar with dimensions time and sensor. Get the
        average across sensors 5, 6, and 8 in a time window::

            >>> ROI = [5, 6, 8]
            >>> Y = MEG.summary(sensor=ROI, time=(.1, .2))

        Get the peak in the same data:

            >>> ROI = [5, 6, 8]
            >>> Y = MEG.summary(sensor=ROI, time=(.1, .2), func=np.max)

        Get the RMS over all sensors

            >>> MEG_RMS = MEG.summary('sensor', func=statfuncs.RMS)

        """
        func = regions.pop('func', self.properties.get('summary_func', np.mean))
        name = regions.pop('name', '{func}({name})')
        name = name.format(func=func.__name__, name=self.name)
        if len(dims) + len(regions) == 0:
            dims = ('case',)

        if regions:
            dims = list(dims)
            dims.extend(dim for dim in regions if not np.isscalar(regions[dim]))
            data = self.subdata(**regions)
            return data.summary(*dims, func=func, name=name)
        else:
            x = self.x
            axes = [self._dim_2_ax[dim] for dim in np.unique(dims)]
            dims = list(self.dims)
            for axis in sorted(axes, reverse=True):
                x = func(x, axis=axis)
                dims.pop(axis)

            # update properties for summary
            properties = self.properties.copy()
            for key in self.properties:
                if key.startswith('summary_') and (key != 'summary_func'):
                    properties[key[8:]] = properties.pop(key)

            if len(dims) == 0:
                return x
            elif dims == ['case']:
                return var(x, name=name)
            else:
                return ndvar(x, dims=dims, name=name, properties=properties)

    def subdata(self, **kwargs):
        """
        returns an ndvar object with a subset of the current ndvar's data.
        The slice is specified using kwargs, with dimensions as keywords and
        indexes as values, e.g.::

            >>> Y.subdata(time = 1)

        returns a slice for time point 1 (second). For dimensions whose values
        change monotonically, a tuple can be used to specify a window::

            >>> Y.subdata(time = (.2, .6))

        returns a slice containing all values for times .2 seconds to .6
        seconds.

        """
        properties = self.properties.copy()
        dims = list(self.dims)
        index = [slice(None)] * len(dims)

        for name, args in kwargs.iteritems():
            try:
                dimax = self._dim_2_ax[name]
                dim = self.dims[dimax]
            except KeyError:
                err = ("Segment does not contain %r dimension." % name)
                raise DimensionMismatchError(err)

            if hasattr(dim, 'dimindex'):
                args = dim.dimindex(args)

            if np.isscalar(args):
                if name == 'sensor':
                    i, value = args, args
                else:
                    i, value = find_time_point(dim, args)
                index[dimax] = i
                dims[dimax] = None
                properties[name] = value
            elif isinstance(args, tuple) and len(args) == 2:
                start, end = args
                if start is None:
                    i0 = None
                else:
                    i0, _ = find_time_point(dim, start)

                if end is None:
                    i1 = None
                else:
                    i1, _ = find_time_point(dim, end)

                s = slice(i0, i1)
                dims[dimax] = dim[s]
                index[dimax] = s
            else:
                index[dimax] = args
                if name == 'sensor':
                    dims[dimax] = dim.get_subnet(args)
                else:
                    dims[dimax] = dim[args]
                properties[name] = args

        # create subdata object
        x = self.x[index]
        dims = tuple(dim for dim in dims if dim is not None)
        return ndvar(x, dims=dims, name=self.name, properties=properties)



class datalist(list):
    """
    :py:class:`list` subclass for including lists in in a dataset.

    The subclass adds certain methods that makes indexing behavior more
    similar to numpy and other data objects. The following methods are
    modified:

    __getitem__:
        Add support for list and array indexes.
    __init__:
        Store a name attribute.

    """
    _stype_ = 'list'
    def __init__(self, items=None, name=None):
        self.name = name
        if items:
            super(datalist, self).__init__(items)
        else:
            super(datalist, self).__init__()

    def __getitem__(self, index):
        if isinstance(index, (int, slice)):
            return list.__getitem__(self, index)

        index = np.array(index)
        if issubclass(index.dtype.type, np.bool_):
            N = len(self)
            assert len(index) == N
            return datalist(self[i] for i in xrange(N) if index[i])
        elif issubclass(index.dtype.type, np.integer):
            return datalist(self[i] for i in index)
        else:
            err = ("Unsupported type of index for datalist: %r" % index)
            raise TypeError(err)

    def compress(self, X, merge='mean'):
        """
        X: factor or interaction; returns a compressed factor with one value
        for each cell in X.

        merge : str
            How to merge entries.
            ``'mean'``: use sum(Y[1:], Y[0])

        """
        if len(X) != len(self):
            err = "Length mismatch: %i (var) != %i (X)" % (len(self), len(X))
            raise ValueError(err)

        x = datalist()
        for cell in X.cells:
            x_cell = self[X == cell]
            n = len(x_cell)
            if n == 1:
                x.append(x_cell)
            elif n > 1:
                if merge == 'mean':
                    xc = reduce(lambda x, y: x + y, x_cell)
                    xc /= n
                else:
                    raise ValueError("Invalid value for merge: %r" % merge)
                x.append(xc)

        return x




class dataset(collections.OrderedDict):
    """
    A dataset is a dictionary that represents a data table.

    Attributes
    ----------
    n_cases : None | int
        The number of cases in the dataset (corresponding to the number of
        rows in the table representation). None if no variables have been
        added.
    n_items : int
        The number of items (variables) in the dataset (corresponding to the
        number of columns in the table representation).

    Methods
    -------
    add:
        Add a variable to the dataset.
    as_table:
        Create a data table representation of the dataset.
    compress:
        For a given categorial description, reduce the number of cases in the
        dataset per cell to one (e.g., average by subject and condition).
    copy:
        Create a shallow copy (i.e., return a new dataset containing the same
        data objects).
    export:
        Save the dataset in different formats.
    eval:
        Evaluate an expression in the dataset's namespace.
    index:
        Add an index to the dataset (to keep track of cases when selecting
        subsets).
    itercases:
        Iterate through each case as a dictionary.
    subset:
        Create a dataset form a subset of cases (slightly faster than normal
        indexing).

    See Also
    --------
    collections.OrderedDict: superclass

    Notes
    -----
    A dataset represents a data table as a {variable_name: value_list}
    dictionary. Each variable corresponds to a column, and each index in the
    value list corresponds to a row, or case.

    The dataset class inherits most of its behavior from its superclass
    :py:class:`collections.OrderedDict`.
    Dictionary keys are enforced to be :py:class:`str` objects and should
    preferably correspond to the variable names.
    An exception is the dataset's length, which reflects the number of cases
    in the dataset (i.e., the number of rows; the number of items can be
    retrieved as  :py:attr:`dataset.n_items`).


    **Accessing Data:**

    Standard indexing with *strings* is used to access the contained var and
    factor objects. Nesting is possible:

    - ``ds['var1']`` --> ``var1``.
    - ``ds['var1',]`` --> ``[var1]``.
    - ``ds['var1', 'var2']`` --> ``[var1, var2]``

    When indexing numerically, the first index defines cases (rows):

    - ``ds[1]`` --> row 1
    - ``ds[1:5]`` == ``ds[1,2,3,4]`` --> rows 1 through 4
    - ``ds[1, 5, 6, 9]`` == ``ds[[1, 5, 6, 9]]`` --> rows 1, 5, 6 and 9

    The second index accesses columns, so case indexing can be combined with
    column indexing:

     - ``ds[:4, :2]`` --> first 4 rows,

    The ``.get_case()`` method or iteration over the dataset
    retrieve individual cases/rows as {name: value} dictionaries.


    **Naming:**

    While var and factor objects themselves need not be named, they need
    to be named when added to a dataset. This can be done by a) adding a
    name when initializing the dataset::

        >>> ds = dataset(('v1', var1), ('v2', var2))

    or b) by adding the var or factor witha key::

        >>> ds['v3'] = var3

    If a var/factor that is added to a dataset does not have a name, the new
    key is automatically assigned to the var/factor's ``.name`` attribute.

    """
    _stype_ = "dataset"
    def __init__(self, *items, **kwargs):
        """
        Datasets can be initialize with data-objects, or with
        ('name', data-object) tuples.::

            >>> ds = dataset(var1, var2)
            >>> ds = dataset(('v1', var1), ('v2', var2))

        The dataset stores the input items themselves, without making a copy().


        Parameters
        ----------

        name : str
            name describing the dataset
        info : dict
            info dictionary, can contain arbitrary entries and can be accessad
            as ``.info`` attribute after initialization.

        """
        args = []
        for item in items:
            if isdataobject(item):
                if item.name:
                    args.append((item.name, item))
                else:
                    err = ("items need to be named in a dataset; use "
                            "dataset(('name', item), ...), or ds = dataset(); "
                            "ds['name'] = item")
                    raise ValueError(err)
            else:
                name, v = item
                if not v.name:
                    v.name = name
                args.append(item)

        self.n_cases = None
        super(dataset, self).__init__(args)
        self.__setstate__(kwargs)

    def __setstate__(self, kwargs):
        self.name = kwargs.get('name', None)
        self.info = kwargs.get('info', {})

    def __reduce__(self):
        args = tuple(self.items())
        kwargs = {'name': self.name, 'info': self.info}
        return self.__class__, args, kwargs

    def __getitem__(self, index):
        """
        possible::

            >>> ds[9]        (int) -> case
            >>> ds[9:12]     (slice) -> subset with those cases
            >>> ds[[9, 10, 11]]     (list) -> subset with those cases
            >>> ds['MEG1']  (strings) -> var
            >>> ds['MEG1', 'MEG2']  (list of strings) -> list of vars; can be nested!

        """
        if isinstance(index, (int, slice)):
            return self.subset(index)

        if isinstance(index, basestring):
            return super(dataset, self).__getitem__(index)

        if not np.iterable(index):
            raise KeyError("Invalid index for dataset: %r" % index)

        if all(isinstance(item, basestring) for item in index):
            return [self[item] for item in index]

        if isinstance(index, tuple):
            i0 = index[0]
            if isinstance(i0, basestring):
                return self[i0][index[1:]]

            if len(index) != 2:
                raise KeyError("Invalid index for dataset: %r" % index)
            i1 = index[1]
            if isinstance(i1, basestring):
                return self[i1][i0]
            elif np.iterable(i1) and all(isinstance(item, basestring) for item
                                         in i1):
                keys = i1
            else:
                keys = datalist(self.keys())[i1]
                if isinstance(keys, basestring):
                    return self[i1][i0]

            subds = dataset(*((k, self[k][i0]) for k in keys))
            return subds

        return self.subset(index)

    def __repr__(self):
        class_name = self.__class__.__name__
        if self.n_cases is None:
            items = []
            if self.name:
                items.append('name=%r' % self.name)
            if self.info:
                info = repr(self.info)
                if len(info) > 60:
                    info = '<...>'
                items.append('info=%s' % info)
            return '%s(%s)' % (class_name, ', '.join(items))

        rep_tmp = "<%(class_name)s %(name)s%(N)s{%(items)s}>"
        fmt = {'class_name': class_name}
        fmt['name'] = '%r ' % self.name if self.name else ''
        fmt['N'] = 'n_cases=%i ' % self.n_cases
        items = []
        for key in self:
            v = self[key]
            if isinstance(v, var):
                lbl = 'V'
            elif isinstance(v, factor):
                lbl = 'F'
            elif isinstance(v, ndvar):
                lbl = 'Vnd'
            else:
                lbl = type(v).__name__

            if getattr(v, 'name', key) == key:
                item = '%r:%s' % (key, lbl)
            else:
                item = '%r:<%s %r>' % (key, lbl, v.name)

            items.append(item)

        fmt['items'] = ', '.join(items)
        return rep_tmp % fmt

    def __setitem__(self, name, item, overwrite=True):
        if not isinstance(name, str):
            raise TypeError("dataset indexes need to be strings")
        else:
            # test if name already exists
            if (not overwrite) and (name in self):
                raise KeyError("dataset already contains variable of name %r" % name)

            # coerce item to data-object
            if isdataobject(item) or isinstance(object, datalist):
                if not item.name:
                    item.name = name
            elif isinstance(item, (list, tuple)):
                item = datalist(item, name=name)
            else:
                pass

            # make sure the item has the right length
            if isndvar(item) and not item.has_case:
                N = 0
            else:
                try:
                    N = len(item)
                except:
                    if hasattr(item, '_data'):  # mne epochs
                        N = len(item._data)
                    else:
                        raise

            if self.n_cases is None:
                self.n_cases = N
            elif self.n_cases != N:
                msg = ("Can not assign item to dataset. The item`s length "
                       "(%i) is different from the number of cases in the "
                       "dataset (%i)." % (N, self.n_cases))
                raise ValueError(msg)

            super(dataset, self).__setitem__(name, item)

    def __str__(self):
        return unicode(self).encode('utf-8')

    def __unicode__(self):
        if sum(map(isuv, self.values())) == 0:
            return self.__repr__()

        maxn = preferences['dataset_str_n_cases']
        txt = unicode(self.as_table(cases=maxn, fmt='%.5g', midrule=True))
        if self.n_cases > maxn:
            note = "... (use .as_table() method to see the whole dataset)"
            txt = os.linesep.join((txt, note))
        return txt

    def _check_n_cases(self, X, empty_ok=True):
        """Check that an input argument has the appropriate length.

        Also raise an error if empty_ok is False and the dataset is empty.
        """
        if self.n_cases is None:
            if empty_ok == True:
                return
            else:
                err = ("Dataset is empty.")
                raise RuntimeError(err)

        n = len(X)
        if self.n_cases != n:
            name = getattr(X, 'name', "the argument")
            err = ("The dataset has a different length (%i) than %s "
                   "(%i)" % (self.n_cases, name, n))
            raise ValueError(err)

    def add(self, item, replace=False):
        """
        ``ds.add(item)`` -> ``ds[item.name] = item``

        unless the dataset already contains a variable named item.name, in
        which case a KeyError is raised. In order to replace existing
        variables, set ``replace`` to True::

            >>> ds.add(item, True)

        """
        if not isdataobject(item):
            raise ValueError("Not a valid data-object: %r" % item)
        elif (item.name in self) and not replace:
            raise KeyError("Dataset already contains variable named %r" % item.name)
        else:
            self[item.name] = item

    def as_table(self, cases=0, fmt='%.6g', f_fmt='%s', match=None, sort=False,
                 header=True, midrule=False, count=False):
        r"""
        returns a fmtxt.Table containing all vars and factors in the dataset
        (ndvars are skipped). Can be used for exporting in different formats
        such as csv.

        Parameters
        ----------

        cases : int
            number of cases to include (0 includes all; negative number works
            like negative indexing)
        count : bool
            Add an initial column containing the case number
        fmt : str
            format string for numerical variables. Hast to be a valid
            `format string,
            <http://docs.python.org/library/stdtypes.html#string-formatting>`
        f_fmt : str
            format string for factors (None -> code; e.g. `'%s'`)
        match : factor
            create repeated-measurement table
        header : bool
            Include the varibale names as a header row
        midrule : bool
            print a midrule after table header
        sort : bool
            Sort the columns alphabetically

        """
        if cases < 1:
            cases = self.n_cases + cases
            if cases < 0:
                raise ValueError("Can't get table for fewer than 0 cases")
        else:
            cases = min(cases, self.n_cases)

        keys = [k for k, v in self.iteritems() if isuv(v)]
        if sort:
            keys = sorted(keys)

        values = [self[key] for key in keys]

        table = fmtxt.Table('l' * (len(keys) + count))

        if header:
            if count:
                table.cell('#')
            for name in keys:
                table.cell(name)

            if midrule:
                table.midrule()

        for i in xrange(cases):
            if count:
                table.cell(i)

            for v in values:
                if isfactor(v):
                    if f_fmt is None:
                        table.cell(v.x[i], fmt='%i')
                    else:
                        table.cell(f_fmt % v[i])
                elif isvar:
                    table.cell(v[i], fmt=fmt)

        return table

    def export(self, fn=None, fmt='%.10g', header=True, sort=False):
        """
        Writes the dataset to a file. The file extesion is used to determine
        the format:

        - '.txt' or '.tsv':  tsv
        - '.tex':  as TeX table
        - '.pickled':  use pickle.dump
        - a filename with any other extension is exported as tsv

        Text and tex export use the :py:meth:`.as_table` method. You can use
        :py:meth:`.as_table` directly for more control over the output.


        Parameters
        ----------

        fn : str(path) | None
            target file name (if ``None`` is supplied, a save file dialog is
            displayed). the extesion is used to determine the format (see
            above)
        fmt : format str
            format for scalar values
        header : bool
            write the variables' names in the first line
        sort : bool
            Sort variables alphabetically according to their name

        """
        if not isinstance(fn, basestring):
            fn = ui.ask_saveas(ext=[('txt', "Tab-separated values"),
                                    ('tex', "Tex table"),
                                    ('pickled', "Pickle")])
            if fn:
                print 'saving %r' % fn
            else:
                return

        ext = os.path.splitext(fn)[1][1:]
        if ext == 'pickled':
            pickle.dump(self, open(fn, 'w'))
        else:
            table = self.as_table(fmt=fmt, header=header, sort=sort)
            if ext in ['txt', 'tsv']:
                table.save_tsv(fn, fmt=fmt)
            elif ext == 'tex':
                table.save_tex(fn)
            else:
                table.save_tsv(fn, fmt=fmt)

    def eval(self, expression):
        """
        Evaluate an expression involving items stored in the dataset.

        ``ds.eval(expression)`` is equivalent to ``eval(expression, ds)``.

        Examples
        --------
        In a dataset containing factors 'A' and 'B'::

            >>> ds.eval('A % B')
            A % B

        """
        return eval(expression, self)

    def get_case(self, i):
        "returns the i'th case as a dictionary"
        return dict((k, v[i]) for k, v in self.iteritems())

    def get_subsets_by(self, X, exclude=[], name='{name}[{cell}]'):
        """
        splits the dataset by the cells of a factor and
        returns as dictionary of subsets.

        """
        if isinstance(X, basestring):
            X = self[X]

        out = {}
        for cell in X.cells:
            if cell not in exclude:
                setname = name.format(name=self.name, cell=cell)
                index = (X == cell)
                out[cell] = self.subset(index, setname)
        return out

    def compress(self, X, drop_empty=True, name='{name}', count='n', drop_bad=False):
        """
        Return a dataset with one case for each cell in X.

        drop_empty : True
            Drops empty cells in X from the dataset. This is currently the only
            option.

        count : str
            Add a variable with this name to the new dataset, containing the
            number of cases in each cell in X.

        drop_bad : bool
            Drop bad items: silently drop any items for which compression
            raises an error. This concerns primarily factors with non-unique
            values for cells in X (if drop_bad is False, an error is raised
            when such a factor is encountered)

        """
        if not drop_empty:
            raise NotImplementedError('drop_empty = False')

        self._check_n_cases(X, empty_ok=False)

        ds = dataset(name=name.format(name=self.name))

        if count:
            x = filter(None, (np.sum(X == cell) for cell in X.cells))
            ds[count] = var(x)

        for k, v in self.iteritems():
            try:
                if hasattr(v, 'compress'):
                    ds[k] = v.compress(X)
                else:
                    from mne import Epochs
                    if isinstance(v, Epochs):
                        evokeds = []
                        for cell in X.cells:
                            idx = (X == cell)
                            evoked = v[idx].average()
                            evokeds.append(evoked)
                        ds[k] = evokeds
                    else:
                        err = ("Unsupported value type: %s" % type(v))
                        raise TypeError(err)
            except:
                if drop_bad:
                    pass
                else:
                    raise

        return ds

    def copy(self):
        "ds.copy() returns an shallow copy of ds"
        ds = dataset(name=self.name, info=self.info.copy())
        ds.update(self)
        return ds

    def index(self, name='index'):
        """
        Add an index to the dataset (i.e., `range(n_cases)`), e.g. for later
        alignment.

        name : str
            Name of the new index variable.

        """
        self[name] = var(np.arange(self.n_cases))

    def itercases(self, start=None, stop=None):
        "iterate through cases (each case represented as a dict)"
        if start is None:
            start = 0

        if stop is None:
            stop = self.n_cases
        elif stop < 0:
            stop = self.n_cases - stop

        for i in xrange(start, stop):
            yield self.get_case(i)

    @property
    def N(self):
        warn("dataset.N s deprecated; use dataset.n_cases instead",
             DeprecationWarning)
        return self.n_cases

    @property
    def n_items(self):
        return super(dataset, self).__len__()

    def repeat(self, n, name='{name}'):
        """
        Analogous to :py:func:`numpy.repeat`. Returns a new dataset with each
        row repeated ``n`` times.

        """
        ds = dataset(name=name.format(name=self.name))
        for k, v in self.iteritems():
            ds[k] = v.repeat(n)
        return ds

    @property
    def shape(self):
        return (self.n_cases, self.n_items)

    def subset(self, index, name='{name}'):
        """
        Returns a dataset containing only the subset of cases selected by
        `index`.

        index : array | str
            index selecting a subset of epochs. Can be an valid numpy index or
            a string (the name of a variable in dataset, or any expression to
            be evaluated in the dataset's namespace).
        name : str
            name for the new dataset

        Notes
        -----
        Keep in mind that index is passed on to numpy objects, which means
        that advanced indexing always returns a copy of the data, whereas
        basic slicing (using slices) returns a view.

        """
        if isinstance(index, int):
            index = slice(index, index + 1)
        elif isinstance(index, str):
            index = self.eval(index)

        name = name.format(name=self.name)
        info = self.info.copy()

        if isvar(index):
            index = index.x

        ds = dataset(name=name, info=info)
        for k, v in self.iteritems():
            ds[k] = v[index]

        return ds



class interaction(_effect_):
    """
    Represents an interaction effect.


    Attributes
    ----------

    factors :
        List of all factors (i.e. nonbasic effects are broken up into
        factors).
    base :
        All effects.

    """
    _stype_ = "interaction"
    def __init__(self, base):
        """
        Usually not initialized directly but through operations on
        factors/vars.

        Parameters
        ----------
        base : list
            List of data-objects that form the basis of the interaction.

        """
        # FIXME: interaction does not update when component factors update
        self.base = effect_list()
        self.is_categorial = True
        self.nestedin = set()

        for b in base:
            if isuv(b):
                self.base.append(b.copy()),
                if isvar(b):
                    if self.is_categorial:
                        self.is_categorial = False
                    else:
                        raise TypeError("No Interaction between two var objects")
            elif isinteraction(b):
                if (not b.is_categorial) and (not self.is_categorial):
                    raise TypeError("No Interaction between two var objects")
                else:
                    self.base.extend(b.base)
                    self.is_categorial = (self.is_categorial and b.is_categorial)

            elif b._stype_ == "nonbasic":  # TODO: nested effects
                raise NotImplementedError("interaction of non-basic effects")
#    from _regresor_.__mod__ (?)
#        if any([type(e)==nonbasic_effect for e in [self, other]]):
#            multcodes = _inter
#            name = ':'.join([self.name, other.name])
#            factors = self.factors + other.factors
#            nestedin = self._nestedin + other._nestedin
#            return nonbasic_effect(multcodes, factors, name, nestedin=nestedin)
#        else:
                self.base.append(b)
                self.nestedin.update(b.nestedin)
            else:
                raise TypeError("Invalid type for interaction: %r" % type(b))

        self._n_cases = N = len(self.base[0])
        if not all([len(f) == N for f in self.base[1:]]):
            err = ("Interactions only between effects with the same number of "
                   "cases")
            raise ValueError(err)

        self.base_names = [str(f.name) for f in self.base]
        self.name = ' x '.join(self.base_names)
        self.random = False
        self.df = reduce(operator.mul, [f.df for f in self.base])

        # determine cells:
        factors = effect_list(filter(isfactor, self.base))
        self.cells = list(itertools.product(*(f.cells for f in factors)))

        # effect coding
        codelist = [f.as_effects for f in self.base]
        codes = reduce(_effect_interaction, codelist)
        self.as_effects = codes
        self.beta_labels = ['?'] * codes.shape[1]  # TODO:

    def __repr__(self):
        names = [str(f.name) for f in self.base]
        if preferences['short_repr']:
            return ' % '.join(names)
        else:
            return "interaction({n})".format(n=', '.join(names))

    # container ---
    def __len__(self):
        return self._n_cases

    def __getitem__(self, index):
        if isvar(index):
            index = index.x

        out = tuple(f[index] for f in self.base)

        if np.iterable(index):
            return interaction(out)
        else:
            return out

    def __contains__(self, item):
        return self.base.__contains__(item)

    def __iter__(self):
        for i in xrange(len(self)):
            yield tuple(b[i] for b in self.base)

    # numeric ---
    def __eq__(self, other):
        X = tuple((f == cell) for f, cell in zip (self.base, other))
        return np.all(X, axis=0)

    def __ne__(self, other):
        X = tuple((f != cell) for f, cell in zip (self.base, other))
        return np.any(X, axis=0)

    def as_factor(self):
        name = self.name.replace(' ', '')
        x = self.as_labels()
        return factor(x, name)

    def as_cells(self):
        [case for case in self]

    def as_labels(self, delim=' '):
        return map(delim.join, self)



class diff(object):
    """
    helper to create difference values for correlation.

    """
    def __init__(self, X, c1, c2, match, sub=None):
        """
        X: factor providing categories
        c1: category 1
        c2: category 2
        match: factor matching values between categories

        """
        raise NotImplementedError
        # FIXME: use celltable
        sub = X.isany(c1, c2)
#        ct = celltable
#        ...
        i1 = X.code_for_label(c1)
        i2 = X.code_for_label(c2)
        self.I1 = X == i1;                self.I2 = X == i2

        if sub is not None:
            self.I1 = self.I1 * sub
            self.I2 = self.I2 * sub

        m1 = match.x[self.I1];          m2 = match.x[self.I2]
        self.s1 = np.argsort(m1);       self.s2 = np.argsort(m2)
        assert np.all(np.unique(m1) == np.unique(m2))
        self.name = "{n}({x1}-{x2})".format(n='{0}',
                                            x1=X.cells[i1],
                                            x2=X.cells[i2])

    def subtract(self, Y):
        ""
        assert type(Y) is var
#        if self.sub is not None:
#            Y = Y[self.sub]
        Y1 = Y[self.I1]
        Y2 = Y[self.I2]
        y = Y1[self.s1] - Y2[self.s2]
        name = self.name.format(Y.name)
        # name = Y.name + '_DIFF'
        return var(y, name)

    def extract(self, Y):
        ""
        y1 = Y[self.I1].x[self.s1]
        y2 = Y[self.I2].x[self.s2]
        assert np.all(y1 == y2), Y.name
        if type(Y) is factor:
            return factor(y1, Y.name, random=Y.random, labels=Y.cells,
                          sort=False)
        else:
            return var(y1, Y.name)

    @property
    def N(self):
        return np.sum(self.I1)


def var_from_dict(base, values, name=None, default=0):
    """
    Create a var object by mapping ``base`` to ``values``.

    Parameters
    ----------
    base : sequence
        Sequence to be mapped to the new var.
    values : dict
        Mapping from values in base to values in the new var.
    name : None | str
        Name for the new var.
    default : scalar
        Default value to supply for entries in ``base`` that are not in
        ``values``.

    Examples
    --------
    >>> base = factor('aabbcde')
    >>> var_from_dict(base, {'a': 5, 'e': 8}, default=0)
    var([5, 5, 0, 0, 0, 0, 8])

    """
    Y = var([values.get(b, default) for b in base], name=name)
    return Y


def var_from_apply(source_var, function, apply_to_array=True):
    if apply_to_array:
        x = function(source_var.x)
    else:
        x = np.array([function(val) for val in source_var.x])
    name = "%s(%s)" % (function.__name__, source_var.name)
    return var(x, name=name)



def box_cox_transform(X, p, name=True):
    """
    :returns: a variable with the Box-Cox transform applied to X. With p==0,
        this is the log of X; otherwise (X**p - 1) / p

    :arg var X: Source variable
    :arg float p: Parameter for Box-Cox transform

    """
    if isvar(X):
        if name is True:
            name = "Box-Cox(%s)" % X.name
        X = X.x
    else:
        if name is True:
            name = "Box-Cox(x)"

    if p == 0:
        y = np.log(X)
    else:
        y = (X ** p - 1) / p

    return var(y, name=name)



def split(Y, n=2, name='{name}_{split}'):
    """
    Returns a factor splitting Y in n categories with equal number of cases
    (e.g. n=2 for a median split)

    Y : array-like
        variable to split
    n : int
        number of categories
    name : str |

    """
    if isinstance(Y, var):
        y = Y.x

    d = 100. / n
    percentile = np.arange(d, 100., d)
    values = [scipy.stats.scoreatpercentile(y, p) for p in percentile]
    x = np.zeros(len(y))
    for v in values:
        x += y > v

    fmt = {'name': Y.name}
    if n == 2:
        fmt['split'] = "mediansplit"
    else:
        fmt['split'] = "split%i" % n

    name = name.format(fmt)
    return factor(x, name=name)



class nested_effect(object):
    _stype_ = "nested"
    def __init__(self, effect, nestedin):
        if not iscategorial(nestedin):
            raise TypeError("Effects can only be nested in categorial base")

        self.effect = effect
        self.nestedin = nestedin
        self._n_cases = len(effect)

        if isfactor(self.effect):
            e_name = self.effect.name
        else:
            e_name = '(%s)' % self.effect
        self.name = "%s(%s)" % (e_name, nestedin.name)

        if len(nestedin) != self._n_cases:
            err = ("Unequal lengths: effect %r len=%i, nestedin %r len=%i" %
                   (e_name, len(effect), nestedin.name, len(nestedin)))
            raise ValueError(err)

    def __repr__(self):
        return self.name

    def __len__(self):
        return self._n_cases

    @property
    def df(self):
        return len(self.effect.cells) - len(self.nestedin.cells)

    @property
    def as_effects(self):
        "create effect codes"
        codes = np.zeros((self._n_cases, self.df))
        ix = 0
        iy = 0
        for cell in self.nestedin.cells:
            n_idx = (self.nestedin == cell)
            n = n_idx.sum()
            codes[iy:iy + n, ix:ix + n - 1] = _effect_eye(n)
            iy += n
            ix += n - 1

        return codes

#        nesting_base = self.nestedin.as_effects
#        value_map = map(tuple, nesting_base.tolist())
#        codelist = []
#        for v in np.unique(value_map):
#            nest_indexes = np.where([v1 == v for v1 in value_map])[0]
#
#            e_local_values = self.effect.x[nest_indexes]
#            e_unique_local_values = np.unique(e_local_values)
#
#            n = len(e_unique_local_values)
#            nest_codes = _effect_eye(n)
#
#            v_codes = np.zeros((self.effect._n_cases, n - 1), dtype=int)
#
#            i1 = set(nest_indexes)
#            for v_self, v_code in zip(e_unique_local_values, nest_codes):
#                i2 = set(np.where(self.effect.x == v_self)[0])
#                i = list(i1.intersection(i2))
#                v_codes[i] = v_code
#
#            codelist.append(v_codes)
#
#        effect_codes = np.hstack(codelist)
#        return effect_codes



class nonbasic_effect(object):
    _stype_ = "nonbasic"
    def __init__(self, effect_codes, factors, name, nestedin=[],
                 beta_labels=None):
        self.nestedin = nestedin
        self.name = name
        self.random = False
        self.as_effects = effect_codes
        self._n_cases, self.df = effect_codes.shape
        self.factors = factors
        self.beta_labels = beta_labels

    def __repr__(self):
        txt = "<nonbasic_effect: {n}>"
        return txt.format(n=self.name)

    # container ---
    def __len__(self):
        return self._n_cases



class model(object):
    """
    stores a list of effects which constitute a model for an ANOVA.

    a model's data is exhausted by its. .effects list; all the rest are
    @properties.

    Accessing effects:
     - as list in model.effects
     - with name as model[name]

    """
    _stype_ = "model"
    def __init__(self, *x):
        """

        Parameters : factors | effects | models
            factors and secondary effects contained in the model.
            Can also contain models, in which case all the models' effects
            will be added.

        """
        if len(x) == 0:
            raise ValueError("model needs to be initialized with effects")

        # try to find effects in input
        self.effects = effects = effect_list()
        self._n_cases = n_cases = len(x[0])
        for e in x:
            # check that all effects have same number of cases
            if len(e) != n_cases:
                e0 = effects[0]
                err = ("All effects contained in a model need to describe"
                       " the same number of cases. %r has %i cases, %r has"
                       " %i cases." % (e0.name, len(e0), e.name, len(e)))
                raise ValueError(err)

            #
            if iseffect(e):
                effects.append(e)
            elif ismodel(e):
                effects += e.effects
            else:
                err = ("model needs to be initialized with effects (vars, "
                       "factors, interactions, ...) and/or models (got %s)"
                       % type(e))
                raise TypeError(err)

        # beta indices
        self.beta_index = beta_index = {}
        i = 1
        for e in effects:
            k = i + e.df
            beta_index[e] = slice(i, k)
            i = k

        # dfs
        self.df_total = df_total = n_cases - 1  # 1=intercept
        self.df = df = sum(e.df for e in effects)
        self.df_error = df_error = df_total - df

        if df_error < 0:
            raise ValueError("Model overspecified")

        # names
        self.name = ' + '.join([str(e.name) for e in self.effects])

    def __repr__(self):
        names = self.effects.names()
        if preferences['short_repr']:
            return ' + '.join(names)
        else:
            x = ', '.join(names)
            return "model(%s)" % x

    def __str__(self):
        return str(self.get_table(cases=50))

    # container ---
    def __len__(self):
        return self._n_cases

    def __getitem__(self, sub):
        if isinstance(sub, str):
            for e in self.effects:
                if e.name == sub:
                    return e
            raise ValueError("No effect named %r" % sub)
        else:
            return model(*(x[sub] for x in self.effects))

    def __contains__(self, effect):
        return id(effect) in map(id, self.effects)

    def sorted(self):
        """
        returns sorted model, interactions last

        """
        out = []
        i = 1
        while len(out) < len(self.effects):
            for e in self.effects:
                if len(e.factors) == i:
                    out.append(e)
            i += 1
        return model(*out)

    # numeric ---
    def __add__(self, other):
        return model(self, other)

    def __mul__(self, other):
        return model(self, other, self % other)

    def __mod__(self, other):
        out = []
        for e_self in self.effects:
            for e_other in model(other).effects:
                out.append(e_self % e_other)
        return model(*out)

    # repr ---
    @property
    def model_eq(self):
        return self.name

    def get_table(self, cases='all'):
        """
        :returns: the full model as a table.
        :rtype: :class:`fmtxt.Table`

        :arg cases: maximum number of cases (lines) to display.

        """
        full_model = self.full
        if cases == 'all':
            cases = len(full_model)
        else:
            cases = min(cases, len(full_model))
        n_cols = full_model.shape[1]
        table = fmtxt.Table('l' * n_cols)
        table.cell("Intercept")
        for e in self.effects:
            table.cell(e.name, width=e.df)

        # rules
        i = 2
        for e in self.effects:
            j = i + e.df - 1
            if e.df > 1:
                table.midrule((i, j))
            i = j + 1

        # data
        for line in full_model[:cases]:
            for i in line:
                table.cell(i)

        if cases < len(full_model):
            table.cell('...')
        return table

    # coding
    @LazyProperty
    def as_effects(self):
        return np.hstack((e.as_effects for e in self.effects))

    def fit(self, Y):
        """
        Find the beta weights by fitting the model to data

        Parameters
        ----------
        Y : var | array, shape = (n_cases,)
            Data to fit the model to.

        Returns
        -------
        beta : array, shape = (n_regressors, )
            The beta weights.
        """
        Y = asvar(Y)
        beta = dot(self.Xsinv, Y.x)
        return beta

    @LazyProperty
    def full(self):
        "returns the full model including an intercept"
        out = np.empty((self._n_cases, self.df + 1))

        # intercept
        out[:, 0] = np.ones(self._n_cases)
        self.full_index = {'I': slice(0, 1)}

        # effects
        i = 1
        for e in self.effects:
            j = i + e.df
            out[:, i:j] = e.as_effects
            self.full_index[e] = slice(i, j)
            i = j
        return out

    # checking model properties
    def check(self, v=True):
        "shortcut to check linear independence and orthogonality"
        return self.lin_indep(v) + self.orthogonal(v)

    def lin_indep(self, v=True):
        "Checks the model for linear independence of its factors"
        msg = []
        ne = len(self.effects)
        codes = [e.as_effects for e in self.effects]
#        allok = True
        for i in range(ne):
            for j in range(i + 1, ne):
#                ok = True
                e1 = self.effects[i]
                e2 = self.effects[j]
                X = np.hstack((codes[i], codes[j]))
#                V0 = np.zeros(self._n_cases)
                # trash, trash, rank, trash = np.linalg.lstsq(X, V0)
                if rank(X) < X.shape[1]:
#                    ok = False
#                    allok = False
                    if v:
                        errtxt = "Linear Dependence Warning: {0} and {1}"
                        msg.append(errtxt.format(e1.name, e2.name))
        return msg

    def orthogonal(self, v=True):
        "Checks the model for orthogonality of its factors"
        msg = []
        ne = len(self.effects)
        codes = [e.as_effects for e in self.effects]
#        allok = True
        for i in range(ne):
            for j in range(i + 1, ne):
                ok = True
                e1 = self.effects[i]
                e2 = self.effects[j]
                e1e = codes[i]
                e2e = codes[j]
                for i1 in range(e1.df):
                    for i2 in range(e2.df):
                        dotp = np.dot(e1e[:, i1], e2e[:, i2])
                        if dotp != 0:
                            ok = False
#                            allok = False
                if v and (not ok):
                    errtxt = "Not orthogonal: {0} and {1}"
                    msg.append(errtxt.format(e1.name, e2.name))
        return msg

    # category access
    @property
    def unique(self):
        full = self.full
        unique_indexes = np.unique([tuple(i) for i in full])
        return unique_indexes

    @property
    def n_cat(self):
        return len(self.unique)

    def iter_cat(self):
        full = self.full
        for i in self.unique:
            cat = np.all(full == i, axis=1)
            yield cat

    def repeat(self, n):
        "Analogous to numpy repeat method"
        effects = [e.repeat(n) for e in self.effects]
        out = model(effects)
        return out

    @LazyProperty
    def Xsinv(self):
        X = self.full
        XT = X.T
        Xsinv = dot(inv(dot(XT, X)), XT)
        return Xsinv

