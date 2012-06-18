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

import cPickle as pickle
import operator
import os
import collections

import numpy as np
import scipy.stats

from eelbrain import fmtxt
from eelbrain import ui




defaults = dict(fullrepr = False,  # whether to display full arrays/dicts in __repr__ methods
                repr_len = 5,      # length of repr
                dataset_str_n_cases = 500,
                var_repr_n_cases = 100,
                factor_repr_n_cases = 100,
                var_repr_fmt = '%.3g',
                factor_repr_use_labels = True,
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
    X = np.empty((n, n-1), dtype=np.int8)
    X[:n-1] = np.eye(n-1, dtype=np.int8)
    X[n-1] = -1
    return X


def rank(A, tol=1e-8):
    """
    Rank of a matrix, from 
    http://mail.scipy.org/pipermail/numpy-discussion/2008-February/031218.html 
    
    """
    s = np.linalg.svd(A, compute_uv=0)
    return np.sum(np.where(s > tol, 1, 0))


def isstr(item):
    return isinstance(item, basestring)

def ismodel(X):
    return hasattr(X, '_stype_') and X._stype_ == "model"

def isvar(Y):
    return hasattr(Y, '_stype_') and Y._stype_ == "var"

def isndvar(Y):
    return hasattr(Y, '_stype_') and Y._stype_ == "ndvar"

def isfactor(Y):
    return hasattr(Y, '_stype_') and Y._stype_ == "factor"

def isinteraction(Y):
    return hasattr(Y, '_stype_') and Y._stype_ == "interaction"

def iscategorial(Y):
    "factors as well as interactions are categorial"
    return hasattr(Y, '_stype_') and Y._stype_ in ["factor", "interaction"]

def isuv(Y):
    "univariate (var, factor)"
    return hasattr(Y, '_stype_') and Y._stype_ in ["factor", "var"]

def isdataobject(Y):
    if hasattr(Y, '_stype_'):
        if  Y._stype_ in ["model", "var", "ndvar", "factor", "interaction",
                          "nonbasic"]:
            return True
    return False

def isdataset(Y):
    return hasattr(Y, '_stype_') and Y._stype_ == 'dataset'


def asmodel(X, sub=None):
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

def asvar(Y, sub=None):
    if isvar(Y):
        pass
    else:
        Y = var(Y)
    
    if sub is not None:
        return Y[sub]
    else:
        return Y

def asfactor(Y, sub=None):
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

def ascategorial(Y, sub=None):
    if iscategorial(Y):
        pass
    else:
        Y = asfactor(Y)
    
    if sub is not None:
        return Y[sub]
    else:
        return Y
    

def combine(items):
    """
    combine a list of items of the same type into one item (dataset, var, 
    factor or ndvar)
    
    """
    item0 = items[0]
    if isdataset(item0):
        out = dataset()
        for name in item0:
            out[name] = combine([ds[name] for ds in items])
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
            kwargs = dict(name = item0.name, 
                          random = item0.random)
    #                      colors = item0._colors, # FIXME: inherit colors
        return factor(x, **kwargs)
    elif isndvar(item0):
        dims = item0.dims
        x = np.concatenate([v.x for v in items], axis=0)
        return ndvar(x, dims=dims, name=item0.name, properties=item0.properties)
    else:
        raise ValueError("Unknown data-object: %r" % item0)



#   Primary Data Containers ---

class _regressor_(object):
    """
    baseclass for factors, variables, and interactions
    
    """    
    def __len__(self):
        return self.N
    
    # __ combination - methods __
    def __add__(self, other):
        return model(self) + other
    
    def __mul__(self, other):
        return model(self, other, self % other)
    
    def __mod__(self, other, name='{name}%{other}'):
#        if any([type(e)==nonbasic_effect for e in [self, other]]):
#            multcodes = _inter
#            name = ':'.join([self.name, other.name])
#            factors = self.factors + other.factors
#            nestedin = self._nestedin + other._nestedin
#            return nonbasic_effect(multcodes, factors, name, nestedin=nestedin)
#        else:
        if isvar(other):
            other = other.x
            other_name = other.name
        elif  ismodel(other):
            return model(self) % other
        elif isdataobject(other):
            return interaction([self, other])
        else:
            other_name = str(other)[:15]
        
        name = name.format(name=self.name, other=other_name)
        return var(self.x % other, name=name)
    
    def contains_factor(self, item):
        item_id = id(item)
        if any([item_id == id(f) for f in self.factors]): 
            return True
        else:
            return any([np.all(f.as_effects == item.as_effects) for f in self.factors])
    
    def nestedin(self, item):
        "returns True if self is nested in item, False otherwise"
        if hasattr(self, '_nestedin'):
            return any([id(e) == id(item) for e in self._nestedin])
        else:
            return False
    
    def _interpret_y(self, y):
        return y
    
    def __eq__(self, y):
        y = self._interpret_y(y)
        return self.x == y
    
    def __ne__(self, y):
        y = self._interpret_y(y)
        return self.x != y
    
    def export(self, fn=None, fmt='%s', delim=os.linesep):
        "write all values to a plain text file"
        if fn is None:
            msg = "Save var %s" % self.name
            fn = ui.ask_saveas(msg, msg, None)
        
        with open(fn, 'w') as FILE:
            FILE.write(delim.join(fmt % v for v in self))
    
    def isany(self, *values):
        """
        Returns an index array that is True in all those locations that match 
        one of the provided ``values``::
        
            >>> a = factor('aabbcc')
            >>> b.isany('b', 'c')
            array([False, False,  True,  True,  True,  True], dtype=bool)
        
        """
        values = self._interpret_y(values)
        return np.any([self.x == v for v in values], axis=0)
    
    def isnot(self, *values):
        """
        returns a boolean array that is True where the data does not equal any 
        of the values
        
        """
        values = self._interpret_y(values)
        return np.all([self.x != v for v in values], axis=0)
    
    def iter_beta(self):
        for i, name in enumerate(self.beta_labels):
            yield i, name

                    
                



class var(_regressor_):
    """
    Container for scalar data.
    
    """
    _stype_ = "var"
    def __init__(self, x, name=None):
        """
        Initialization:
        
        :arg array x: data
        :arg name: name of the variable
        
        Apart from model building operations, the following operations are
        available for var objects:
        
        :Operations:
        
            * ``var + scalar``
            * ``var - scalar`` 
            * ``var * scalar`` 
            * ``var / scalar`` 
        
        """
        _regressor_.__init__(self)
        x = np.asarray(x)
        if x.ndim > 1:
            raise ValueError("Use ndvar class for data with more than one dimension")
        self.__setstate__((x, name))
    
    def __setstate__(self, state):
        x, name = state
        # raw
        self.name = name
        self.x = x
        # derived
        self.N = len(x)
        self.mu = x.mean()
        self.centered = self.x - self.mu
        self.SS = np.sum(self.centered**2)     
        # constants
        self.df = 1
        self.visible = True
        self.random = False
    
    def __getstate__(self):
        return (self.x, self.name)
    
    def __repr__(self, full=False):
        n_cases = defaults['var_repr_n_cases']
        
        if self.x.dtype == bool:
            fmt = '%r'
        else:
            fmt = defaults['var_repr_fmt']
        
        if full or len(self.x) <= n_cases:
            x = [fmt % v for v in self.x]
        else:
            x = [fmt % v for v in self.x[:n_cases]]
            x.append('<... N=%s>' % len(self.x))
        
        x = '[' + ', '.join(x) + ']'
        args = [x, 'name=%r' % self.name]
        
        return "var(%s)" % ', '.join(args)
    
    def __str__(self):
        return self.__repr__(True)
    
    def __contains__(self, value):
        return value in self.x
    
    def __add__(self, other):
        if np.isscalar(other):
            return var(self.x + other,
                       name='+'.join((self.name, str(other))))
        else:
            return _regressor_.__add__(self, other)
    
    def __sub__(self, other):
        "subtract: values are assumed to be ordered. Otherwise use .sub method."
        if np.isscalar(other):
            return var(self.x - other,
                       name='-'.join((self.name, str(other))))
        else:
            assert len(other.x) == len(self.x)
            x = self.x - other.x
            n1, n2 = self.name, other.name
            if n1 == n2:
                name = n1
            else:
                name = "%s-%s"%(n1, n2)
            return var(x, name)
    
    def __mul__(self, other):
        if iscategorial(other):
            return _regressor_.__mul__(self, other)
        elif isvar(other):
            return var(self.x * other.x,
                       name='*'.join((self.name, other.name)))            
        else: #  np.isscalar(other)
            if len(str(other)) < 10:
                name = '*'.join((self.name, str(other)))
            else:
                name = self.name
            return var(self.x * other, name=name)
    
    def __floordiv__(self, other):
        if isvar(other):
            x = self.x // other.x
            name = '//'.join((self.name, other.name))
        elif np.isscalar(other):
            x = self.x // other
            name = '//'.join((self.name, str(other)))
        else:
            x = self.x // other
            name = '//'.join((self.name, '?'))
        return var(x, name=name)
    
    def __gt__(self, y):
        y = self._interpret_y(y)
        return self.x > y  
          
    def __lt__(self, y):
        y = self._interpret_y(y)
        return self.x < y        
    
    def __getitem__(self, index):
        "if factor: return new variable with mean values per factor category"
        if isfactor(index):
            f = index
            x = []
            for v in np.unique(f.x):
                x.append(np.mean(self.x[f==v]))
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
                       name='/'.join((self.name, str(other))))
        elif isvar(other):
            return var(self.x / other.x,
                       name='/'.join((self.name, other.name)))            
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
            name = ' per '.join([self.name, categories.name])
            labels = categories.dummy_complete_labels
            out = nonbasic_effect(codes, [self, categories], name, 
                                  beta_labels=labels)
            return out
    
    @property
    def as_effects(self):
        "for effect initialization"
        return self.centered[:,None]
    
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
    
    def copy(self, name='{name}'):
        "returns a deep copy of itself"
        return var(self.x.copy(), name=name.format(name=self.name))
    
    def compress(self, X, func=np.mean, name='{name}'):
        """
        X: factor or interaction; returns a compressed factor with one value
        for each cell in X.
        
        """
        if len(X) != len(self):
            err = "Length mismatch: %i (var) != %i (X)" % (len(self), len(X))
            raise ValueError(err)
        
        x = []
        for i in X.values():
            x_i = self.x[X == i]
            x.append(func(x_i))
        
        x = np.array(x)
        name = name.format(name=self.name)
        out = var(x, name=name)
        return out
    
    @property
    def beta_labels(self):
        return [self.name]
    
    def diff(self, X, i1, i2, match):
        """
        subtracts X==i2 from X==i1; sorts values in ascending order according 
        to match
        
        """
        assert isfactor(X)
        I1 = X==i1;             I2 = X==i2
        Y1 = self[I1];          Y2 = self[I2]
        m1 = match.x[I1];       m2 = match.x[I2]
        s1 = np.argsort(m1);    s2 = np.argsort(m2)
        y = Y1[s1] - Y2[s2]
        name = "{n}({x1}-{x2})".format(n=self.name, 
                                       x1=X.cells[i1],
                                       x2=X.cells[i2])
        return var(y, name)
    
    def export(self, fn=None, fmt='%.10g', delim=os.linesep):
        "Write all values to a plain text file"
        if self.x.dtype == np.bool_:
            fmt = '%r'
        
        _regressor_.export(self, fn, fmt=fmt, delim=delim)
    
    @property
    def factors(self):
        return [self]
    
    def repeat(self, repeats, name='{name}'):
        "Like :py:func:`numpy.repeat`"
        return var(self.x.repeat(repeats), name=name.format(name=self.name))



def find_time_point(timevar, time):
    if time in timevar.x:
        i = np.where(timevar==time)[0][0]
    else:
        i_next = np.where(timevar > time)[0][0]
        t_next = timevar[i_next]
        i_prev = np.where(timevar < time)[0][-1]
        t_prev = timevar[i_prev]
        if (t_next - time) < (time - t_prev):
            i = i_next
            time = t_next
        else:
            i = i_prev
            time = t_prev
    return i, time



class factor(_regressor_):
    """
    Container for categorial data. 
    
    """
    _stype_ = "factor"
    def __init__(self, x, name=None, random=False, 
                 labels={}, colors={}, retain_label_codes=False,
                 rep=1, chain=1):
        """        
        x : array-like 
            Sequence of values (initialization uses ravel(x) to create 1-d 
            array). If all conditions are coded with a single character, x can 
            be a string, e.g. ``factor('io'*8, name='InOut')``
        
        name : str
            name of the factor
            
        random : bool
            treat factor as random factor (important for ANOVA; default = False)
            
        labels : dict or None
            if provided, these labels are used to replace values in x when
            constructing the labels dictionary. All labels for values of 
            x not in `labels` are constructed using ``str(value)``.
            
        colors : dict {label: color, ...}
            Provide a color for each value, hich can be used by some plotting 
            functions. Colors should be matplotlib-readable.
        
        rep : int
            repeat values in x rep times e.g. ``factor(['in', 'out'], rep=3)``
            --> ``factor(['in', 'in', 'in', 'out', 'out', 'out'])``
        
        chain : int
            chain x; e.g. ``factor(['in', 'out'], chain=3)``
            --> ``factor(['in', 'out', 'in', 'out', 'in', 'out'])``
        
        
        Example - different ways to initialize the same factor::
        
            >>> factor(['in', 'in', 'in', 'out', 'out', 'out'])
            >>> factor([1, 1, 1, 0, 0, 0], labels={1: 'in', 2: 'out'})
        
        
        """
        _regressor_.__init__(self)
        # prepare arguments
        if isstr(x):
            x = list(x)
        
        x = np.ravel(x)
        if rep > 1: x = x.repeat(rep)
        if chain > 1: x = np.tile(x, chain)
        
        # prepare data containers: _x are internal versions
        state = {'name': name, 'random': random, 'colors': {}}
        N = len(x)
        _x = state['x'] = np.empty(N, dtype=np.uint16)
        _labels = state['labels'] = {}
        categories = np.unique(x)
        if retain_label_codes and N > 0:
            if not issubclass(x.dtype.type, np.integer):
                msg = ("When retaining_label_codes is True, x must contain "
                       "integers")
                raise ValueError(msg)
            elif min(x) < 0 or max(x) > 65534:
                msg =  ("When retaining_label_codes is True, x must contain "
                       "unsigned 16-bit integers (0 <= x < 65534)")
                raise ValueError(msg)
            for i, cat in enumerate(categories):
                _x[x==cat] = cat
                _labels[cat] = labels.get(cat, str(cat))
        else: # reassign codes
            for cat in categories:
                label = labels.get(cat, str(cat))
                if label in _labels.values():
                    i = 0
                    while _labels[i] != label:
                        i += 1
                else:
                    i = max(_labels) + 1 if _labels else 0
                    _labels[i] = label
                
                _x[x==cat] = i
        
        if colors: # convert color keys from values to codes
            codes = {lbl: code for code, lbl in _labels.iteritems()}
            for label in colors:
                try:
                    code = codes[label]
                except KeyError:
                    msg = "Label %r in colors, but not in values" % label
                    raise KeyError(msg)
                else:
                    state['colors'][code] = colors[label]
        
        self.__setstate__(state)
    
    def __setstate__(self, state):
        self.x = x = state['x']
        self.name = state['name']
        self.random = state['random']
        self._labels = labels = state['labels']
        self._codes = {lbl: code for code, lbl in labels.iteritems()}
        self._colors = state['colors']
        # constants
        self.visible = True
        # derived
        self.N = N = len(x)
        
        # get unique categories and sort them in order of first occurrence
        categories = np.unique(x)
        self.df = df = max(0, len(categories) - 1)
        
        if N: 
            # x_deviation_coded
            categories = np.unique(x)
            cats = categories[:-1]
            contrast = categories[-1]
            shape = (N, df)
            codes = np.empty(shape, dtype=np.int8)
            for i, cat in enumerate(cats):
                codes[:,i] = x==cat
            codes -= (x==contrast)[:,None]
            self.x_deviation_coded = self.as_effects = codes
            
            # x_dummy_coded
            codes = np.empty(shape, dtype=np.int8)
            for i, cat in enumerate(cats):
                codes[:,i] = x==cat
            self.x_dummy_coded = self.as_dummy = codes
        else:
            self.x_deviation_coded = np.array([])
            self.as_effects = np.array([])
            self.x_dummy_coded = np.array([])
            self.as_dummy = np.array([])
    
    def __getstate__(self):
        state = {'x': self.x,
                 'name': self.name,
                 'random': self.random,
                 'labels': self._labels,
                 'colors': self._colors}
        return state
    
    def __repr__(self, full=False):
        use_labels = defaults['factor_repr_use_labels']
        n_cases = defaults['factor_repr_n_cases']
        
        if use_labels:
            values = self.as_labels()        
        else:
            values = self.x.tolist()
        
        if full or len(self.x) <= n_cases:
            x = str(values)
        else:
            x = [repr(v) for v in values[:n_cases]]
            x.append('<... N=%s>' % len(self.x))
            x = '[' + ', '.join(x) + ']'
        
        args = [x, 'name=%r' % self.name]
        
        if self.random:
            args.append('random=True')
        
        if not use_labels:
            args.append('labels=%s' % self.cells)
        
        return 'factor(%s)' % ', '.join(args)
    
    def __str__(self):
        return self.__repr__(True)
    
    def __contains__(self, value):
        try:
            code = self._codes(value)
        except KeyError:
            return False
        return code in self.x
    
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
    
    def __iter__(self):
        return (self._labels[i] for i in self.x)
    
    def __setitem__(self, index, values):
        values = self._interpret_y(values, create=True)
        self.x[index] = values
    
    def __call__(self, other):
        "create a nested effect"
        assert other._stype_ in ["factor", "nonbasic", "model", "interaction"]
        name = self.name + '(' +  other.name + ')'
        nesting_base = other.as_effects
        # create effect codes
        value_map = map(tuple, nesting_base.tolist())
        codelist = []
        for v in np.unique(value_map):
            nest_indexes = np.where((value_map == v).mean(1) == 1)[0]
            
            self_local_values = self.x[nest_indexes]
            self_unique_local_values = np.unique(self_local_values)
            
            n = len(self_unique_local_values)
            nest_codes = _effect_eye(n)
            
            v_codes = np.zeros((self.N, n-1), dtype=int)
            
            i1 = set(nest_indexes)
            for v_self, v_code in zip(self_unique_local_values, nest_codes):
                i2 = set(np.where(self.x == v_self)[0])
                i = list(i1.intersection(i2))
                v_codes[i] = v_code
            
            codelist.append(v_codes)
        effect_codes = np.hstack(codelist)
        # out
        out = nonbasic_effect(effect_codes, [self], name, nestedin=other.factors)
        return out
    
    def _child_kwargs(self, name='{name}'):
        kwargs = dict(labels = self._labels,
                      name = name.format(name=self.name), 
                      random = self.random,
                      colors = self._colors,
                      retain_label_codes = True)
        return kwargs
            
    def _interpret_y(self, Y, create=False):
        """
        in: string or list of strings
        returns: list of values (codes) corresponding to the categories
        
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
                return 65535 # code for values not present in the factor 
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
    def as_dummy_complete(self):
        x = self.x[:,None]
        categories = np.unique(x)
        codes = np.hstack([x==cat for cat in categories])
        return codes.astype(np.int8)
        
    def as_labels(self):
        return [self._labels[v] for v in self.x]
    
    @property
    def beta_labels(self):
        labels = self.dummy_complete_labels
        txt = '{0}=={1}'
        return [txt.format(labels[i], labels[-1]) for i in range(len(labels)-1)]
    
    @property
    def cells(self):
        return self._labels.copy()
    
    def codes(self):
        return np.unique(self.x)
    
    def compress(self, X, name='{name}'):
        """
        :returns: a compressed :class:`factor` with one value for each cell in X
        :rtype: :class:`factor`
        
        :arg X: cell definition
        :type X: factor or interaction
        
        Raises an error if there are cells that contain more than one value.
        
        """
        if len(X) != len(self):
            err = "Length mismatch: %i (var) != %i (X)" % (len(self), len(X))
            raise ValueError(err)
        
        x = []
        for i in X.values():
            x_i = np.unique(self.x[X == i])
            if len(x_i) > 1:
                raise ValueError("non-unique cell")
            else:
                x.append(x_i[0])
        
        x = np.array(x)
        name = name.format(name=self.name)
        out = factor(x, name=name, labels=self.cells, random=self.random)
        return out
    
    def copy(self, name='{name}', rep=1, chain=1):
        "returns a deep copy of itself"
        f = factor(np.copy(self.x), rep=rep, chain=chain, 
                   **self._child_kwargs(name))
        return f
    
    @property
    def dummy_complete_labels(self):
        categories = np.unique(self.x)
        return [self._labels[cat] for cat in categories]

    @property
    def factors(self):
        return [self]
    
    def get_color(self, name):
        ":arg name: can be label or code"
        if isstr(name):
            code = self._codes[name]
        else:
            code = name
        
        if code in self._colors:
            return self._colors[code]
        else:
            raise KeyError("No color for %r"%name)
    
    def get_index_to_match(self, other):
        """
        Assuming that ``other`` is a shuffled version of self, this method
        returns ``index`` to transform from the order of self to the order of
        ``other``. To guarantee exact matching, wach value can only occur once
        in self. Example::
        
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
        Project the factor onto an index array ``target``:
        
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
        "Repeat elements of a factor (like :py:func:`numpy.repeat`)"
        return factor(self.x.repeat(repeats), **self._child_kwargs(name))
    
    def set_color(self, name, color):
        """
        :arg name: can be label or code
        :arg color: should be matplotlib compatible
        
        """
        if isstr(name):
            code = self._codes[name]
        else:
            code = name
        
        self._colors[code] = color
    
    def values(self):
        """
        returns a list of all values that occur in the factor. ``.values()`` 
        guarantees to only return values that actually occur in the data.
        
        """
        values = [self._labels[i] for i in np.unique(self.x)]
        return sorted(values)



class ndvar(object):
    _stype_ = "ndvar"
    def __init__(self, x, dims=('case',), properties=None, name=None, info=""):
        """
        Arguments
        ---------
        
        For each agument, the example assumes you are importing 600 epochs of 
        EEG data for 80 time points from 32 sensors.
        
        dims : tuple
            the dimensions characterizing the shape of each case. E.g., 
            ``(var('time', range(-.2, .6, .01)), sensor_net)``.
        
        x : array
            the first dimension should contain cases, and the subsequent 
            dimensions should correspond to the ``dims`` argument. E.g., 
            ``data.shape = (600, 80, 32).
        
        properties : dict
            data properties dictionary
        
        
         .. note::
            ``data`` and ``dims`` are stored without copying. A shallow
            copy of ``properties`` is stored. Make sure the relevant objects 
            are not modified externally later.
        
        """        
        # check data shape
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
                       "%i in dimension" % (dim.name, n, n_dim))
                raise DimensionMismatchError(err)
        
        state = {'dims': dims,
                 'x': x,
                 'name': name,
                 'info': info}
        
        # store attributes
        if properties is None:
            state['properties'] = {}
        else:
            state['properties'] = properties.copy()
        
        self.__setstate__(state)
    
    def __setstate__(self, state):
        self.dims = dims = state['dims']
        self._case = (dims[0] == 'case')
        self._truedims = truedims = dims[self._case:]
        
        # dimnames
        self.dimnames = tuple(dim.name for dim in truedims)
        if self._case:
            self.dimnames = ('case',) + self.dimnames
        
        self.x = x = state['x']
        self.name = state['name']
        self.info = state['info']
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
                 'info': self.info,
                 'properties': self.properties}
        return state
    
    def _align(self, other):
        "align data from 2 ndvars"
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
    
    def _ialign(self, other):
        "align for self-mofiying operations (+= ...)"
        assert all(dim in self.dimnames for dim in other.dimnames)
        i_other = []
        for dim in self.dimnames:
            if dim in other.dimnames:
                i_other.append(dim)
            else:
                i_other.append(None)
        return other.get_data(i_other)
       
    def __add__(self, other):
        if isndvar(other):
            dims, x_self, x_other = self._align(other)
            x = x_self + x_other
        elif np.isscalar(other):
            x = self.x + other
            dims = self.dims
        else:
            raise ValueError("can't add %r" % other)
        return ndvar(x, dims=dims, properties=self.properties)
    
    def __iadd__(self, other):
        self.x += self._ialign(other)
        return self
    
    def __sub__(self, other): # TODO: use dims
        if isndvar(other):
            dims, x_self, x_other = self._align(other)
            x = x_self - x_other
        elif np.isscalar(other):
            x = self.x - other
            dims = self.dims
        else:
            raise ValueError("can't subtract %r" % other)
        return ndvar(x, dims=dims, properties=self.properties)    
    
    def __isub__(self, other):
        self.x -= self._ialign(other)
        return self
    
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
        if self._case:
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
        if not self._case:
            raise DimensionMismatchError("%r has no case dimension" % self)
        if len(X) != len(self):
            err = "Length mismatch: %i (var) != %i (X)" % (len(self), len(X))
            raise ValueError(err)
        
        x = []
        for i in X.values():
            x_i = self.x[X == i]
            x.append(func(x_i, axis=0))
        
        # update properties for summary
        properties = self.properties.copy()
        for key in self.properties:
            if key.startswith('summary_') and (key != 'summary_func'):
                properties[key[8:]] = properties.pop(key)        
        
        x = np.array(x)
        name = name.format(name=self.name)
        info = os.linesep.join((self.info, "compressed by %r" % X)) 
        out = ndvar(x, self.dims, properties=properties, info=info, name=name)
        return out
    
    def copy(self):
        "returns a copy with a view on the object's data"
        x = self.x
        return self.__class__(x, dims=self.dims, name=self.name, 
                              properties=self.properties)
    
    def deepcopy(self):
        "returns a copy with a deep copy of the object's data"
        x = self.x.copy()
#        dims = tuple(dim.copy() for dim in self.dims)
        return self.__class__(x, dims=self.dims, name=self.name, 
                              properties=self.properties)
    
    def get_axis(self, dim):
        return self._dim_2_ax[dim]
    
    def get_case(self, index, name="{name}[{index}]"):
        "returns a single case (epoch) as ndvar"
        if not self._case:
            raise DimensionMismatchError("%r does not have cases" % self)
        
        x = self.x[index]
        name = name.format(name=self.name, index=index)
        case = ndvar(x, dims=self.dims[1:], properties=self.properties, name=name, 
                     info=self.info + ".get_case(%i)" % index)
        return case
    
    def get_data(self, dims):
        """
        returns the data with a specific ordering of dimension as indicated in 
        ``dims``.
        
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
        if name in self._dim_2_ax:
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
    
    def summary(self, *dims, **regions):
        r"""
        Returns a new ndvar with specified dimensions collapsed.
        
        .. warning::
            Data is collapsed over the different dimensions in turn using the 
            provided function with an axis argument. For certain functions 
            this is not equivalent to collapsing over several axes concurrently
            (e.g., np.var)
        
        \*dims : int | str
            dimensions specified are collapsed over their whole range.  
        \*\*regions : 
            If regions are specified through keyword-arguments, then only the 
            data over the specified range is included. Use like .subdata()
            kwargs.
        
        
        **additional kwargs:**
        
        func : callable
            Function used to collapse the data. Needs to accept an "axis" 
            kwarg (default: np.mean)
        name : str
            default: "{func}({name})"
        
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
            
            info = os.linesep.join((self.info, 'summary: %s' % func.__name__))
            
            # update properties for summary
            properties = self.properties.copy()
            for key in self.properties:
                if key.startswith('summary_') and (key != 'summary_func'):
                    properties[key[8:]] = properties.pop(key)
            
            if dims:
                return ndvar(x, dims=dims, name=name, properties=properties, info=info)
            else:
                return var(x, name=name)
    
    def mean(self, name="mean({name})"): # FIXME: Do I need this?
        if self._case:
            return self.summary(func=np.mean, name=name)
        else:
            return self
    
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
                dimvar = self.dims[dimax]
            except KeyError:
                err = ("Segment does not contain %r dimension." % name)
                raise DimensionMismatchError(err)
            
            if np.isscalar(args):
                if name == 'sensor':
                    i, value = args, args
                else:
                    i, value = find_time_point(dimvar, args)
                index[dimax] = i
                dims[dimax] = None
                properties[name] = value
            elif isinstance(args, tuple) and len(args) == 2:
                start, end = args
                if start is None:
                    i0 = None
                else:
                    i0, _ = find_time_point(dimvar, start)
                
                if end is None:
                    i1 = None
                else:
                    i1, _ = find_time_point(dimvar, end)
                
                s = slice(i0, i1)
                dims[dimax] = dimvar[s]
                index[dimax] = s
            else:
                index[dimax] = args
                if name == 'sensor':
                    dims[dimax] = dimvar.get_subnet(args)
                else:
                    dims[dimax] = dimvar[index]
                properties[name] = args
        
        # create subdata object
        x = self.x[index]
        dims = tuple(dim for dim in dims if dim is not None)
        return ndvar(x, dims=dims, name=self.name, properties=properties)



class dataset(collections.OrderedDict):
    """
    A dataset is a dictionary that stores a collection of variables (``var``, 
    ``factor``, and ``ndvar`` objects) that describe the same underlying cases. 
    Keys are inforced to be ``str`` objects and should preferably correspond 
    to the variable names.
    

    **Accessing Data:**
    
    - Use the the ``.get_case()`` method or iteration over the dataset to 
      retrieve individual cases/rows as {name: value} dictionaries.  
    - Use standard indexing (``dataset[x]``) for retrieving 
      variables (``str`` keys) and printing certain rows (``>>> print 
      dataset[1:4]``). 
    
    Standard indexing with *strings* is used to access the contained var and
    factor objects:
            
    - ``ds['var1']`` --> ``var1``. 
    - ``ds['var1',]`` --> ``[var1]``.
    - ``ds['var1', 'var2']`` --> ``[var1, var2]``
    
    Standard indexing with *integers* can be used to retrieve a subset of cases 
    (rows):
    
    - ``ds[1]``
    - ``ds[1:5]`` == ``ds[1,2,3,4]``
    - ``ds[1, 5, 6, 9]`` == ``ds[[1, 5, 6, 9]]``
    
    Case indexing is primarily useful to display only certain rows of the 
    table::
    
        >>> print ds[3:5]
    
    Case indexing is implemented by a call to the .subset() method, which 
    should probably be used preferably for anything but interactive table
    inspection.  
    
    """
    _stype_ = "dataset"
    def __init__(self, *items, **kwargs):
        """
        Datasets can be initialize with data-objects, or with 
        ('name', data-object) tuples.::

            >>> dataset(var1, var2)
            >>> dataset(('v1', var1), ('v2', var2))

        The dataset stores the input items themselves, without making a copy().
        The dataset class is a :class:`collections.OrderedDict` subclass, with
        different initialization and representation.
        
        
        **Naming:**
        
        While var and factor objects themselves need not be named, they need 
        to be named when added to a dataset. This can be done by a) adding a 
        name when initializing the dataset::
        
            >>> ds = dataset(('v1', var1), ('v2', var2))
        
        or b) by adding the var or factor witha key::
        
            >>> ds['v3'] = var3
        
        If a var/factor that is added to a dataset does not have a name, the new 
        key is automatically added as name to the var/factor. 
        
        
        **optional kwargs:**
        
        name : str
            name describing the dataset
                
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
        
        super(dataset, self).__init__(args)
        self.__setstate__(kwargs)
    
    def __setstate__(self, kwargs):
        self.name = kwargs.get('name', None)
        self.info = kwargs.get('info', {})
        
    def __reduce__(self):
        args = tuple(self.items())
        kwargs = {'name': self.name, 'info': self.info}
        return self.__class__, args, kwargs
        
    def __getitem__(self, name):
        """
        possible::
        
            >>> ds[9]        (int) -> case
            >>> ds[9:12]     (slice) -> subset with those cases
            >>> ds[[9, 10, 11]]     (list) -> subset with those cases
            >>> ds['MEG1']  (strings) -> var
            >>> ds['MEG1', 'MEG2']  (list of strings) -> list of vars; can be nested!
        
        """
        if isinstance(name, int):
            name = slice(name, name+1)
        
        if isinstance(name, slice):
            return self.subset(name)
        
        is_str = isinstance(name, basestring)
        is_sequence = np.iterable(name) and not is_str
        
        if is_sequence:
            all_str = all(isinstance(item, basestring) for item in name)
            if all_str:
                return [self[item] for item in name]
            else:
                if isinstance(name, tuple):
                    name = list(name)
                return self.subset(name)
        
        else:
            return super(dataset, self).__getitem__(name)
    
    def __repr__(self):
        rep_tmp = "<dataset %(name)r N=%(N)i: %(items)s>"
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
                lbl = '?'
            
            name = repr(key)
            items.append((name, lbl))
        items = ', '.join('%s(%s)'%(k, v) for k, v in items)
        args = dict(name=self.name, N=self.N, items=items)
        return rep_tmp % args
    
    def __setitem__(self, name, item, overwrite=True):
        try:
            name = str(name)
        except TypeError:
            raise TypeError("dataset indexes need to be strings")
        else:
            # test if name already exists
            if (not overwrite) and (name in self):
                raise KeyError("dataset already contains variable of name %r"%name)
            
            # coerce item to data-object
            if isdataobject(item):
                if not item.name:
                    item.name = name
            elif isinstance(item, list):
                pass # list of something
            else:
                try:
                    if all(np.isreal(item)):
                        item = var(item, name=name)
                    else:
                        item = factor(item, name=name)
                except:
                    msg = "%r could not be converted to a valid data object" % item
                    raise ValueError(msg)
            
            # make sure the item has the right length
            if len(self) == 0:
                self.N = len(item)
            else:
                if self.N != len(item):
                    msg = ("The item`s length (%i) is different from the "
                           "number of cases in the datase (%i)." % (len(item), self.N))
                    raise ValueError(msg)
            
            super(dataset, self).__setitem__(name, item)
    
    def __str__(self):
        maxn = defaults['dataset_str_n_cases']
        txt = str(self.as_table(cases=maxn, fmt='%.5g', midrule=True))
        if self.N > maxn:
            note = "... (use .as_table() method to see the whole dataset)"
            txt = os.linesep.join((txt, note))
        return txt
    
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
        
        Arguments
        ---------
        
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
            cases = self.N + cases
            if cases < 0:
                raise ValueError("Can't get table for fewer than 0 cases")
        else:
            cases = min(cases, self.N)
        
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
        
        Text and tex export use :py:meth:`.as_table`. You can use 
        :py:meth:`.as_table` directly for more control over the output. 
        
        
        Arguments
        ---------
        
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
            fn = ui.ask_saveas(ext = [('txt', "Tab-separated values"),
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
            elif ext =='tex':
                table.save_tex(fn)
            else:
                table.save_tsv(fn, fmt=fmt)
    
    def get_case(self, i):
        "returns the i'th case as a dictionary"
        return dict((k, v[i]) for k, v in self.iteritems())
    
    def get_subsets_by(self, factor, exclude=[], name='{name}[{case}]'):
        """
        splits the dataset by the cells of a factor and 
        returns as dictionary of subsets.
        
        """
        if isinstance(factor, basestring):
            factor = self[factor]
        
        out = {}
        for case in factor.values():
            if case not in exclude:
                setname = name.format(name=self.name, case=case)
                index = factor == case
                out[case] = self.subset(index, setname)
        return out
    
    def compress(self, X, name='{name}', count='n'):
        ds = dataset(name=name.format(name=self.name))
        
        if count:
            x = [np.sum(X == cell) for cell in X.values()]
            ds[count] = var(x)
        
        for k in self:
            ds[k] = self[k].compress(X)
                
        return ds
    
    def itercases(self, start=None, stop=None):
        "iterate through cases (each case represented as a dict)"
        if start is None:
            start = 0
        
        if stop is None:
            stop = self.N
        elif stop < 0:
            stop = self.N - stop
        
        for i in xrange(start, stop):
            yield self.get_case(i)
    
    @property
    def shape(self):
        return (self.N, len(self))
    
    def subset(self, index, name='{name}'):
        """
        Returns a dataset containing only the subset of cases selected by 
        `index`. 
        
        index : array | str
            index selecting a subset of epochs. Can be an valid numpy index or
            the name of a variable in dataset.
            Keep in mind that index is passed on to numpy objects, which means
            that advanced indexing always returns a copy of the data, whereas
            basic slicing (using slices) returns a view.
        name : str
            name for the new dataset
        
        """
        if isinstance(index, str):
            index = self[index]
        
        name = name.format(name=self.name)
        info = self.info.copy()
        
        if isvar(index):
            index = index.x
        
        ds = dataset(name=name, info=info)
        for k, v in self.iteritems():
            ds[k] = v[index]
        
        return ds 







#   Models ---

class interaction(_regressor_):
    _stype_ = "interaction"
    """
    
    attributes
    ----------
    factors: list of all factors contributing (i.e. nonbasic effects are broken
             up into factors)
    base:    all effects 
    
    """
    def __init__(self, base, beta_labels=None):
        self.factors = []
        self.base = []
        vars_ = 0
        for b in base:
            # check item is valid
            if not hasattr(b, 'factors'):
                raise ValueError('Invalid base item for interaction: %r'%b)
            
            for f in b.factors:
                if not self.contains_factor(f):
                    self.factors.append(f)
            
            if b._stype_ == "nonbasic":
                self.base.append(b)
            elif b._stype_ == "interaction":
                self.base += b.base
            else:
                self.base += b.factors
            
            if b._stype_ == "var":
                if vars_ == 0:
                    vars_ = 1
                else:
                    raise NotImplementedError("No Interaction between two variables")
        
        N = self.N = self.base[0].N; assert all([f.N == N for f in self.base[1:]])
        name = ' x '.join([f.name for f in self.base])
        _regressor_.__init__(self)
        self.name = name
        self.visible = True
        self.random = False
        self.beta_labels = beta_labels
        
        self.df =  reduce(operator.mul, [f.df for f in self.base])
        
        # determine cells:
        label_dicts = [f.cells for f in self.factors if isfactor(f)]
        self.cells = _permutate_address_dicts(label_dicts)
        self._cells = _permutate_address_dicts(label_dicts, link=False)
        self.indexes = sorted(self.cells.keys())
        self._colors = {}
        
        # effect coding
        codelist = [f.as_effects for f in self.base]
        codes = reduce(_effect_interaction, codelist)
        self.as_effects = codes
    
    def __repr__(self):
        names = [f.name for f in self.base]
        txt = "interaction({n})"
        return txt.format(n=', '.join(names))        
        
    def __str__(self):
        names = [f.name for f in self.base]
        return ' % '.join(names)
    
    @property
    def _nestedin(self):
        return set.union([set(f.nestedin) for f in self.factors])
    
    def __getitem__(self, index):
        if isvar(index):
            index = index.x
        
        out = [f[index] for f in self.base]
        if np.iterable(index):
            return interaction(out)
        else:
            return out
    
    def __eq__(self, other):
        out = np.ones(self.N, dtype=bool)
        for i, f in zip(other, self.factors):
            if i != None:
                out *= f==i
        return out
    
    def as_codes(self):
        out = []
        for i in xrange(self.N):
            code = tuple(f.x[i] for f in self.factors)
            out.append(code)
        return out
    
    def as_factor(self):
        name = self.name.replace(' ', '')
        x = self.as_labels()
        return factor(x, name)
    
    def as_labels(self):
        out = [self.cells[code] for code in self.as_codes()]
        return out
    
    def values(self):
        values = self._cells.values()
        return sorted(values)



def _effect_interaction(a, b):
    k = a.shape[1]
    out = [a[:,i,None] * b for i in range(k)]
    return np.hstack(out)






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
        i1 = X.code_for_label(c1)
        i2 = X.code_for_label(c2)
        self.I1 = X==i1;                self.I2 = X==i2
        
        if sub is not None:
            self.I1 = self.I1 * sub
            self.I2 = self.I2 * sub

        m1 = match.x[self.I1];          m2 = match.x[self.I2]
        self.s1 = np.argsort(m1);       self.s2 = np.argsort(m2)
        assert np.all(np.unique(m1) == np.unique(m2))
        self.name = "{n}({x1}-{x2})".format(n='{0}', 
                                            x1=X.cells[i1],
                                            x2=X.cells[i2])
#        self.sub = sub
    def subtract(self, Y):
        ""
        assert type(Y) is var
#        if self.sub is not None:
#            Y = Y[self.sub]
        Y1 = Y[self.I1]
        Y2 = Y[self.I2]
        y = Y1[self.s1] - Y2[self.s2]
        name = self.name.format(Y.name)
        #name = Y.name + '_DIFF'
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


def var_from_dict(name, base, values_dict, default=0):
    """
    Creates a variable containing a value defined in values_dict for each 
    category in key_factor.
    
    """
    x = np.empty(len(base))
    x.fill(default)
    for k,v in values_dict.iteritems():
        x[base==k] = v
    return var(x, name=name)


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
        y = (X**p - 1) / p
    
    return var(y, name=name)



def split(Y, n=2, name=True):
    """
    returns a factor splitting Y in n categories (e.g. n=2 for a median split)
    Y can be array or var
    
    """
    if isinstance(Y, var):
        y = Y.x
    d = 100. / n
    percentile = np.arange(d, 100., d)
    values = [scipy.stats.scoreatpercentile(y, p) for p in percentile]
    x = np.zeros(len(y))
    for v in values:
        x += y > v
    if name is True:
        if n == 2:
            name = Y.name + "_mediansplit"
        else:
            name = Y.name + "_split%s"%n
    elif not isstr(name):
        raise ValueError("name must be True or string")
    return factor(x, name)



class nonbasic_effect(_regressor_):
    _stype_ = "nonbasic"
    def __init__(self, effect_codes, factors, name, nestedin=[], 
                 beta_labels=None):
        self._nestedin = nestedin
        _regressor_.__init__(self)
        self.name = name
        self.visible = True
        self.random = False
        self.as_effects = effect_codes
        self.N, self.df = effect_codes.shape
        self.factors = factors
        self.beta_labels = beta_labels
    def __repr__(self):
        txt = "<nonbasic_effect: {n}>"
        return txt.format(n=self.name)
    


class multifactor(factor):
    "For getting categories from the combination of a list of factors"
    def __init__(self, factors, v=False):
        factors = [asfactor(f) for f in factors]
        
        self.N = factors[0].N
        
        if len(factors) == 1:
            f = factors[0]
            self.x = np.copy(f.x)
            self._labels = f._labels.copy()
            self.name = f.name
        else:
            x_full = np.hstack([f.x[:,None] for f in factors])
            x_tup = [tuple(x) for x in x_full]
            # categories            
            if v:
                print "x_tup:"
                print x_tup
            categories, c_sort = np.unique(x_tup, return_index=True)
            if v:
                print "categories:"
                print categories
                print c_sort
            categories = categories[np.argsort(c_sort)]
            if v:
                print "categories:"
                print categories
            # data containers
            categories = np.unique(x_tup)
            if v:
                print categories
            x = np.zeros(self.N)
            labels = {}
            for i, index in enumerate(categories):
                x[np.all(x_full==index, axis=1)] = i
                labels[i] = ', '.join([f.cells[j] for f,j in zip(factors, index)])
            # create factor TODO: random
            if v:
                print x
            self.x = x
            self._labels = labels
            self.name = ':'.join([f.name for f in factors])
    
    def __repr__(self):
        factors = ', '.join(f.name for f in self.factors)
        out = "multifactor(%s)" % factors
        return out
    
    def iter_n_i(self):
        for i, n in self.cells.iteritems():
            yield n, self.x==i

        
class model(object):
    """
    stores a list of effects which constitute a model for an ANOVA.
    
    a model's data is exhausted by its. .effects list; all the rest are
    @properties.
    
    x can be: factor
              effect
              model
              list of effects
    
    modify M.effects at own peril!
    
    """
    _stype_ = "model"
    def __init__(self, *x):
        # try to find effects in input
        effects = []
        N = 0
        for e in x:
            # sort out N
            if N == 0:
                N = e.N
            else:
                assert e.N == N, "%s has different N, stupid!"%e.name
            # 
            if e._stype_ in ["factor", "var", "nonbasic", "interaction"]:
                effects.append(e)
            elif e._stype_ == "model":
                effects += e.effects
            else:
                raise ValueError("model needs to be initialized with factors and"+\
                                 "/or models (got %s)"%type(e))
        self.effects = effects
        
        # some stable attributes
        self.name = ' + '.join([e.name for e in self.effects])
        self.factor_names = ', '.join([f.name for f in self.factors])
        self.N = N
        # dfs
        self.df_total = N - 1 # 1=intercept
        self.df = sum(e.df for e in self.effects)
        self.df_model = self.df
        self.df_error = self.df_total - self.df_model
    
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
      
    @property
    def model_eq(self):
        return self.name
    
    def __repr__(self):
        x = ', '.join(e.name for e in self.effects)
        return "model(%s)"%x
    
    def __str__(self):
        return str(self.get_table(cases=50))
    
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
            
    # dimensional properties
    def __len__(self):
        return self.N
    
    # coding
    @property
    def as_effects(self):
        out = np.empty((self.N, self.df))
        i = 0
        for e in self.effects:
            j = i+e.df
            out[:,i:j] = e.as_effects
            i = j
        return out
#        return np.hstack([e.as_effects for e in self.effects])
    
    @property
    def full(self):
        "returns the full model including an intercept"
        df = self.df
        assert df < self.N, "Model overspecified"
        out = np.empty((self.N, self.df+1))
        # intercept
        out[:,0] = np.ones(self.N)
        # effects
        i = 1
        for e in self.effects:
            j = i+e.df
            out[:,i:j] = e.as_effects
            i = j
        return out
        # old:
#        model = self.as_effects
#        if not any([len(np.unique(x))==1 for x in model.T]):
#            intercept = np.ones((self.N, 1), dtype=int)
#            model = np.hstack((intercept, model))
#        assert model.shape[1] <= self.N, "Model overspecified"
#        return model
    
    # MARK: coding access
    @property
    def as_dummy_complete(self):
        m = np.hstack(f.as_dummy_complete for f in self.factors)
        out = []
        for i in np.unique([tuple(i) for i in m]):
            out.append(np.all(m==i, axis=1)[:,None].astype(np.int8))
        return np.hstack(out)
    
    # checking model properties
    def check(self, v=True):
        return self.lin_indep(v) + self.orthogonal(v)
    
    def lin_indep(self, v=True):
        "Checks the model for linear independence of its factors"
        msg = []
        ne = len(self.effects)
        codes = [e.as_effects for e in self.effects]
#        allok = True
        for i in range(ne):
            for j in range(i+1, ne):
#                ok = True
                e1 = self.effects[i]
                e2 = self.effects[j]
                X = np.hstack((codes[i], codes[j]))
#                V0 = np.zeros(self.N)
                #trash, trash, rank, trash = np.linalg.lstsq(X, V0)
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
            for j in range(i+1, ne):
                ok = True
                e1 = self.effects[i]
                e2 = self.effects[j]
                e1e = codes[i]
                e2e = codes[j]
                for i1 in range(e1.df):
                    for i2 in range(e2.df):
                        dotp = np.dot(e1e[:,i1], e2e[:,i2])
                        if dotp != 0:
                            ok = False
#                            allok = False
                if v and (not ok):
                    errtxt = "Not orthogonal: {0} and {1}"
                    msg.append(errtxt.format(e1.name, e2.name))
        return msg
    
    ## OTHER STUFF
    @property
    def factors(self):
        f = set() 
        for e in self.effects:
            #print set(e.factors)
            f = f.union(set(e.factors))
        return list(f)
    
    # __combination_methods__
    def __add__(self, other):
        other =  model(other)
#        assert other.df_model <= self.df_error, "Model overspecified"
        return model(*(self.effects + other.effects))
    
    def __mul__(self, other):
        o = model(other)
        i = self % other
#        if self.df_error < o.df_model + i.df_model + 1:
#            txt = ["Model overspecified"]
#            for m in [self, o, i]:
#                txt.append("df({n}) = {i}".format(n=m.name, i=m.df))
#            raise ValueError('\n'.join(txt))
        return self + o + i
    
    def __mod__(self, other):
        out = []
        for e_self in self.effects:
            for e_other in model(other).effects:
                out.append(e_self % e_other)
        return model(*out)
    
    def __getitem__(self, sub):
        effects = [x[sub] for x in self.effects]
        return model(*effects)
    
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
            cat = np.all(full==i, axis=1)
            yield cat
    
#    @property
#    def cells(self):
#        # TODO: implement cells ??? what for? use self.unique?
#        fm = self.full
#        categories = []
#        for i, line in enumerate(fm):
#            if not any([i in cat_list for cat_list in categories]):
#                f = ((fm == line).mean(1) == 1)
#        return categories
    
    def repeat(self, n):
        "Analogous to numpy repeat method"
        effects = [e.repeat(n) for e in self.effects]
        out = model(effects)
        return out
    
    ## for access by aov 
    def iter_effects(self):
        "iterator over visible effects (name, location_slice, df)"
        index = 1
        for i, e in enumerate(self.effects):
            index_end = index + e.df
            #if vis:  
            yield i, e.name, slice(index, index_end), e.df
            index = index_end
#    def iter_regressors(self, random=False):
#        "random: also yield regressors involving random factors"
#        i = 1
#        for e in self.effects:
#            pass # FIXME: dshyg5erjbhaegr



# Structured Collections ---



def _permutate_address_dicts(label_dicts, link=' ', short=False):
    """
    returns combined indexes with labels 
    
    index -> label
    
    short: shorten labels to min length
    
    """
    if short:
        # replace labels with shortened labels
        short_label_dicts = []
        for label_dict in label_dicts:
            values = label_dict.values()
            l = int(short)
            while len(np.unique([v[:l] for v in values])) < len(values):
                l += 1
            d = dict((k, v[:l]) for k, v in var.dictionary.iteritems())
            short_label_dicts.append(d)
        label_dicts = short_label_dicts
    
    # collect possible indexes
    indexes = [()] # all possible indexes
    for label_dict in label_dicts:
        newindexes = []
        for i in indexes:
            for k in label_dict.keys():
                newindexes.append(i + (k,))
        indexes = newindexes
    
    # collect labels
    label_dic = {}
    for index in indexes: 
        label_components = []
        for k, ld in zip(index, label_dicts):
            label_components.append(ld[k])
        
        if isinstance(link, str):
            label = link.join(label_components)
        else:
            label = tuple(label_components)
        
        label_dic[index] = label
    
    return label_dic

