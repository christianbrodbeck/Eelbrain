'''
PLAN
====


Simple Tests
------------

A test:: 

    >>> m = ttest(ds, Y='MEG', X='condition', c1='X__', c2='__X', match=)

returns a dataset with vars
    
    - 'c1': data
    - 'c2': data (or constant)
    - 't': t-map
    - 'p': p-map
    - 'c1-c2': difference

it could have special attributes `all`= c1, c2, difference

??? how does the plot know that the p-map should produce contours over the 
difference? 

    # allow nesting: `plot([v1, [v2, v2overlay]])`


Models
------

ANOVA need a model as an intermediated step. MODELS should be datasets (or a
sister-class)::

    >>> m = model(ds, Y='MEG', X='condition*subject')

would return a model-dataset with:

    - cases averaged to one case per cell
    - Y
    - factors
    - factors for interaction effects

I would need to fit the model to get the results:

    - var 'effect'
    - var 'F'
    - var 'p'
    
    
??? should results be organized with cases? (vs items)
    
    YES: anova effects are all the same data type, and need 'F' and 'p' variables
    NO: ttest wants to store source data 






RESULTS are also datasets except there is only one case
-> why do I need epoch vs. ndvar? Plots need flat data. 
-> use a `ndvar.get_flattened_data(func=np.mean)` method?


Consequences for Plotting
-------------------------

- usage:
 
    - ndvars, or dataset and names of ndvars 
      
        - plotting *args to catch plot(ds, namelist) as well as plot(varlist) as 
          well as plot(ds) where ds has a default_DV
        - use `plot(epochs, ... ds=None) where ds can be None if epochs 
          contains ndvars   
       
- interpretation (by plotting functions):
 
    - ttest.all can be submitted, which should make the plots for 
      ['c1', 'c2', ['diff', 'p']], where each name is expanded to ['c1']
    
        - plot functions are called with `epochs` (nested list)
        - _ax_XXX functions are called with `layers` (lists), since they
          modify the axes
        - _plt_XXX functions only draw a single `layer`
     
    - in case a submitted ndvar contains more than one case, the plotting 
      function calls 
      
        - ndvar.get_summary()        (in _base.unpack_epochs_arg)
        - ndvar.get_epoch_data()     (in the immediate plotting function)
    
    - helper object: dataset['V1', V2'] could simply return a dataset instance


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

Additional classes are generated internally:

:class:`epoch`:
    stores a single epoch of n-dimensional data
:class:`ConditionPointer`:
    Convenience access to datasets. Provides a single Stores a subset of a dataset as well as 
    a single dependent variable

ndvar Plan
==========

basic data types:

    * var
    * factor
    * ndvar

managed by

    * dataset


Interface for plotting
----------------------

 *  requesting dimensions through attributes (`data.sensor`) implicitly
    checks for the presence of those dimensions
    
    - problem: presence of unwanted dimensions - plots should ``data.assert
    
 - dimensions: need to be fulfilled, use `data.assert_dims(('time', 'sensor'))`
 - ndvar vs epoch: automatic transform (mean) could be customized, use 
   `data.as_epoch(func='default')



ndvar
----

difference btw var and ndvar: an nd-dataset is large, but with a guaranteed 
regularity (e.g. time coordinate and topography in MEG), i.e., I
don't want to have the var 'time' replicated over and over

 ..note::
     I could virtualize it?? ]]]

on the other hand, 'trial' is not guaranteed to be the same across subjects, 
so I should store it as factor or var.



other than that I would want a similar interface like for a :class:`var`.



the dataset class should be like a dataframe ??
vs
a dataframe should hold datasets along with 
dataframe -> dataset


Idea
----

Construct as spanning: 
dataframend(space=(subject, t, trial)


or define space for each variable::

    >>> dataframend.add(gender, space=(subject,))
    >>> dataframend.add(MEG, space=(subject, t, sensor))

&& have a special 'map' property, e.g.

    >>> d.add(sensors, topo_obj)
    
Or better construct them as objects::

    >>> subject = S.factor(['s1', ..., 's22'])
    >>> gender = S.factor('mff...', space=(subject,))
    >>> ...
    >>> MEG = S.var(MEG_data, space=(subject, t, sensor))

topo_obj needs map object with ``getLoc2d`` method:

    >>> sensors = S.map(sensor_net)


Implementation
--------------

Problem: array for a full experiment would be way too big

 - 20 subjects
 - 200 sampling time points
 - 160 sensors
 - 6 conditions
 - 49 trials

np.prod([20, 200, 160, 6, 49]) = 188`160`000

Memory required in mb::

    >>> np.prod([20, 200, 160, 6, 49]) / (8 * 1024 * 1024) * 64  # 64 bit
    1408
    >>> np.prod([20, 200, 160, 6, 49]) / (8 * 1024 * 1024) * 32  # 32 bit
    704

'''

from __future__ import division

import cPickle as pickle
import logging
import operator
import os

import numpy as np
import scipy.stats

from eelbrain import fmtxt
from eelbrain import ui




defaults = dict(fullrepr = False,  # whether to display full arrays/dicts in __repr__ methods
                repr_len = 5,      # length of repr
                v_fmt = '%.2f',    # standard value formatting
                p_fmt = '%.3f',    # p value formatting
               )


class DimensionMismatchError(Exception):
    def __init__(self, data, dims):
        msg = "Dimensions of %r do not match %r"%(data, dims)
        Exception.__init__(self, msg)



def _effect_eye(n):
    """Returns effect coding for n categories.
    e.g. _effect_eye(4) = 1  0  0
                          0  1  0
                          0  0  1
                         -1 -1 -1
    """ 
    return np.vstack((np.eye(n-1, dtype=int), 
                      np.ones(n-1, dtype=int)*-1))


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
    


#   Primary Data Containers ---

class _regressor_(object):
    """
    baseclass for factors, variables, and interactions
    
    """
    def __init__(self, name, random):
        self.name = name
        self.visible = True
        self.random = random
    
    def __len__(self):
        return self.N
    
    # __ combination - methods __
    def __add__(self, other):
        return model(self) + other
    
    def __mul__(self, other):
        return model(self, other, self%other)
    
    def __mod__(self, other):
#        if any([type(e)==nonbasic_effect for e in [self, other]]):
#            multcodes = _inter
#            name = ':'.join([self.name, other.name])
#            factors = self.factors + other.factors
#            nestedin = self._nestedin + other._nestedin
#            return nonbasic_effect(multcodes, factors, name, nestedin=nestedin)
#        else:
        logging.debug(" __mod__({0}, {1})".format(self.name, other.name))
        if ismodel(other):
            return model(self) % other
        else:
            return interaction([self, other])
    
    def __contains__(self, item):
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
        if np.iterable(y):
            return np.all([self.x != v for v in y], axis=0)
        else:
            return self.x != y
    
    def __gt__(self, y):
        y = self._interpret_y(y)
        return self.x > y  
          
    def __lt__(self, y):
        y = self._interpret_y(y)
        return self.x < y        
    
    def isany(self, *values):
        """
        Returns an index array that is True in all those locations that match 
        one of the provided `values`::
        
            >>> a = factor('aabbcc')
            >>> b.isany('b', 'c')
            array([False, False,  True,  True,  True,  True], dtype=bool)
        
        """
        values = self._interpret_y(values)
        return np.any([self.x == v for v in values], axis=0)
    
    def iter_beta(self):
        for i, name in enumerate(self.beta_labels):
            yield i, name
    
    def get_dict_for_keys(self, key_factor, key_labels=True, value_labels=True, 
                          conflict='raise'):
        """
        :arg factor key_factor: factor which provides the keys
        :arg bool key_labels: whether to use labels as keys (if available)
        :arg bool value_labels: whether to use labels as values (if available)
        :arg conflict: value to substitute in case more than one values on 
            self map to the same key. Default behavior is to raise an error. 
        
        Returns a dictionary mapping categories of key -> values on self
        
        """
        assert key_factor.N == self.N, "Unequal number of values"
        if not iscategorial(key_factor):
            key_labels = False
        if not iscategorial(self):
            value_labels = False
        
        out = {}
        for i in xrange(self.N):
            key = key_factor.__getitem__(i)
            if key_labels:
                key = key_factor.cells[key]
            value = self.__getitem__(i)
            if value_labels:
                value = self.cells[value]
            # add to out
            if key in out:
                if out[key] == value:
                    pass
                elif np.isnan(out[key]) and np.isnan(value):
                    pass
                elif conflict == 'raise':
                    raise ValueError("Non-unique value for key %r" % key)
                else:
                    out[key] = conflict
            else:
                out[key] = value
        
        return out
                    
                



class var(_regressor_):
    """
    Container for scalar data.
    
    """
    _stype_ = "var"
    def __init__(self, x, name="Covariate"):
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
        _regressor_.__init__(self, name, True)
        self.x = x = np.array(x)
        if x.ndim > 1:
            raise ValueError("Use ndvar class for data with more than one dimension")
        self.N = len(x)
        "Number of data points"
        self.df = 1
        self.mu = x.mean()
        self.centered = self.x - self.mu
        self.SS = np.sum(self.centered**2)     
        
        self.__truediv__ = self.__div__ # to support future division
    
    def __eq__(self, y):
        if isstr(y):
            raise ValueError("Variables can only be compared with floats, not "
                             "with strings")
        else:
            return _regressor_.__eq__(self, y)
    
    def __repr__(self):
        temp = "var({x}, name='{n}')"
        
        if self.x.dtype == bool:
            fmt = '%r'
        else:
            fmt = defaults['v_fmt']
        
        if defaults['fullrepr']:
            x = [fmt % y for y in self.x]
        else:
            x = [fmt % y for y in self.x[:5]]
            if len(self.x) > 5:
                x.append('... n=%s' % len(self.x))
            x_str = '[' + ', '.join(x) + ']'
        return temp.format(n=self.name, x=x_str)
    
    def __str__(self):
        temp = "var({x}, name='{n}')"
        fmt = dict(n=self.name, x=self.x)
        return temp.format(**fmt)
    
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
        if np.isscalar(other):
            return var(self.x * other,
                       name='*'.join((self.name, str(other))))
        elif isvar(other):
            return var(self.x * other.x,
                       name='*'.join((self.name, other.name)))            
        else:
            return _regressor_.__mul__(self, other)
    
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
    
    def copy(self, suffix=''):
        return var(self.x.copy(), name=self.name + suffix)
    
    def compress(self, X, name=None, func=np.mean):
        """
        X: factor or interaction; returns a compressed factor with one value
        for each cell in X.
        
        """
        # find new x
        x = []
        for i in sorted(X.cells.keys()):
            x_i = self.x[X==i]
            x.append(func(x_i))
        x = np.array(x)
        
        # package and ship
        if name is None:
            name = self.name
        out = var(x, name=name)
        return out
    
    def as_factor(self, name=None, labels='%g'):
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
    @property
    def as_effects(self):
        "for effect initialization"
        return self.centered[:,None]
    @property
    def factors(self):
        return [self]
    @property
    def beta_labels(self):
        return [self.name]
    def __getitem__(self, values):
        "if factor: return new variable with mean values per factor category"
        if isfactor(values):
            f = values
            x = []
            for v in np.unique(f.x):
                x.append(np.mean(self.x[f==v]))
            return var(x, self.name)
        else:
            x = self.x[values]
            if np.iterable(values):
                return var(x, self.name)
            else:
                return x
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



class factor(_regressor_):
    """
    Container for categorial data. 
    
    """
    repr_temp = 'factor({v}, name="{n}", random={r}, labels={l})'
    _stype_ = "factor"
    def __init__(self, x, name="Factor", random=False, 
                 labels={}, colors={}, retain_label_codes=False,
                 rep=1, chain=1, sort=False):
        """
        :arg x: Value array (uses ravel to create 1-d array). If all conditions 
            are coded with a single character, x can be a string, e.g. 
            ``factor('io'*8, name='InOut')``
        :arg name:  name of the factor
        :arg bool random:  treat factor as random factor (important for ANOVA; 
            default = False)
        :arg dict labels:  if provided, these labels are used to replace values 
            in x when constructing the labels dictionary. All labels for values of 
            x not in `labels` are constructed using ``str(value)``.
        :arg dict colors:  similar to labels, provide a color for each value. 
            Colors should be matplotlib-readable.
        :arg int rep:  repeat values in x rep times e.g. ``factor(['in', 'out'], rep=3)``
            --> ``factor(['in', 'in', 'in', 'out', 'out', 'out'])``
        :arg int chain: chain x; e.g. ``factor(['in', 'out'], chain=3)``
            --> ``factor(['in', 'out', 'in', 'out', 'in', 'out'])``
        
        """
        _regressor_.__init__(self, name, random)
        # prepare arguments
        if isstr(x):
            x = list(x)
        x = np.ravel(x)
        if rep > 1: x = x.repeat(rep)
        if chain > 1: x = np.ravel([x]*chain)
        self.N = len(x)
        "Number of data points"

        # get unique categories and sort them in order of first occurrence
        categories, c_sort = np.unique(x, return_index=True)
        self.df = len(categories) - 1
        if sort==True:
            categories = categories[np.argsort(c_sort)]

        # prepare data containers
        if retain_label_codes:
            if min(labels.keys()) >= 0 and max(labels.keys()) < 256:
                dtype = np.uint8
            else:
                dtype = np.int32
        else:
            if len(np.unique(x)) < 256:
                dtype = np.uint8
            else:
                dtype = np.int32
        
        self.x = np.empty(self.N, dtype=dtype)
        self.cells = {}
        """
        {value -> label} dictionary, mapping ``int`` values in x to ``str`` 
        category labels
        """
        
        self.colors = {}
#        logging.debug("init FACTOR '{n}' with x={x}, categories->{c}".format(n=name, x=x, c=categories))
        if retain_label_codes:
            assert all(cat in labels for cat in categories)
            # retain codes provided in labels
            for i, cat in enumerate(categories):
                self.x[x==cat] = cat
                self.cells[cat] = labels[cat]
                if cat in colors:
                    self.colors[cat] = colors[cat]
        else:
            # reassign codes
            for i, cat in enumerate(categories):
                self.x[x==cat] = i
                if cat in labels:
                    self.cells[i] = labels[cat]
                else:
                    self.cells[i] = str(cat)
                
                if cat in colors:
                    self.colors[i] = colors[cat]
        
        # convenience arg
        self.indexes = sorted(self.cells.keys())
        
        # x_deviation_coded
        x = self.x
        categories = np.unique(x)
        cats = categories[:-1]
        contrast = categories[-1]
        shape = (self.N, self.df)
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
    
    def __repr__(self):
        fmt = dict(n=self.name, r=str(self.random))
        repr_len = defaults['repr_len']
        if defaults['fullrepr'] or len(self.x)<=repr_len:
            fmt['v'] = self.x
            fmt['l'] = str(self.cells)
        else:
            fmt['v'] = ''.join(['[',
                                ', '.join([str(x) for x in self.x.tolist()[:repr_len]] + \
                                          ['...']),
                                'n=%s]'%len(self.x)])
            if len(self.cells) > repr_len:
                l_repr = str(dict(self.cells.items()[:repr_len]))
                l_repr = l_repr[:-1] + ', ...}'
                fmt['l'] = l_repr
            else:
                fmt['l'] = str(self.cells)
        return self.repr_temp.format(**fmt)
    
    def __str__(self):
        fmt = dict(v=str(self.x.tolist()), n=self.name,
                   r=str(self.random), l=str(self.cells))
        return self.repr_temp.format(**fmt)
    
    def __getitem__(self, sub):
        """
        sub needs to be int or an array of bools of shape(self.x)
        this method is valid for factors and nonbasic effects
        
        """
        out = self.x[sub]
        if np.iterable(out):
            out = factor(out, name=self.name, random=self.random, 
                         labels=self.cells)
            return out
        else:
            return self.cells[out]
    
    def __iter__(self):
        return (self.cells[i] for i in self.x)
    
    def __setitem__(self, index, values):
        values = self._interpret_y(values)
        self.x[index] = values
    
    def __call__(self, other):
        "create a nested effect"
        assert type(other) in [factor, nonbasic_effect, model, interaction]
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
    
    def _get_ID_for_new_cell(self, name):
        "adds a new name to the cells dictionary"
        if self.x.dtype is np.uint8:
            for i in range(256):
                if i not in self.cells:
                    self.cells[i] = str(name)
                    return i
            self.x = np.array(self.x, dtype=np.uint16)
        i = np.max(self.x) + 1
        self.cells[i] = str(name)
        return i
        
    def _interpret_y(self, y):
        """
        in: string or list of strings
        returns: list of values (codes) corresponding to the categories
        
        """
        if np.iterable(y):
            rd = dict((v, k) for k, v in self.cells.iteritems())
            # TODO: implement reversable_dict type
            if isstr(y):
                try:
                    return rd[y]
                except KeyError:
                    return self._get_ID_for_new_cell(y)
            else:
                out = []
                for v in y:
                    if isstr(v):
                        try:
                            v = rd[v]
                        except KeyError:
                            v = self._get_ID_for_new_cell(v)
                    elif v not in self.cells:
                        raise ValueError("unknown cell code: %r" % v)
                    out.append(v)
                return out
        elif y in self.cells:
            return y
        else:
            raise ValueError("unknown cell code: %r" % v)
    
    @property
    def as_dummy_complete(self):
        x = self.x[:,None]
        categories = np.unique(x)
        codes = np.hstack([x==cat for cat in categories])
        return codes.astype(np.int8)
        
    def as_labels(self):
        return np.array([self.cells[v] for v in self.x])
        
    @property
    def beta_labels(self):
        labels = self.dummy_complete_labels
        txt = '{0}=={1}'
        return [txt.format(labels[i], labels[-1]) for i in range(len(labels)-1)]
    
    def code_for_label(self, label):
        for c, l in self.cells.iteritems():
            if label == l:
                return c
        raise KeyError("Label'%s' not in factor %s"%(label, self.name))
    
    def compress(self, X, name=None):
        """
        :returns: a compressed :class:`factor` with one value for each cell in X
        :rtype: :class:`factor`
        
        :arg X: cell definition
        :type X: factor or interaction
        
        Raises an error if there are cells that contain more than one value.
        
        """
        # find new x
        x = []
        for i in sorted(X.cells.keys()):
            x_i = np.unique(self.x[X==i])
            if len(x_i) > 1:
#                x.append(np.nan)
                raise ValueError("non-unique cell")
            else:
                x.append(x_i[0])
        x = np.array(x)
        
        # package and ship
        if name is None:
            name = self.name
        out =  factor(x, name=name, labels=self.cells,
                      random=self.random)
        return out
    
    @property
    def dummy_complete_labels(self):
        categories = np.unique(self.x)
        return [self.cells[cat] for cat in categories]

    @property
    def factors(self):
        return [self]
    
    def get_color(self, name):
        ":arg name: can be label or code"
        if isstr(name):
            code = self.code_for_label(name)
        else:
            code = name
        
        if code in self.colors:
            return self.colors[code]
        else:
            raise KeyError("No color for %r"%name)
    
    def get_index_to_match(self, other):
        """
        returns `index` so that::
        
            >>> index = factor1.get_index_to_match(factor2)
            >>> all(factor1[index] == factor2)
            True
        
        """
        assert self.cells == other.cells
        index = []
        for v in other.x:
            where = np.where(self.x == v)[0]
            if len(where) == 1:
                index.append(where[0])
            else:
                msg = "%r contains several cases of %r"
                raise ValueError(msg % (self, v))
        return np.array(index)
    
    def print_categories(self):
        ":returns: a table containing information about categories"
        table = fmtxt.Table('rll')
        table.title(self.name)
        for title in ['i', 'Label', 'n']:
            table.cell(title)
        table.midrule()
        for v, name in self.cells.iteritems():
            table.cell(v)
            table.cell(name)
            table.cell(np.sum(self.x==v))
        return table
    
    def set_color(self, name, color):
        """
        :arg name: can be label or code
        :arg color: should be matplotlib compatible
        
        """
        if isstr(name):
            code = self.code_for_label(name)
        else:
            code = name
        
        self.colors[code] = color
    
    def tolabels(self, values):
        out = []
        for v in values:
            if isstr(v):
                out.append(v)
            elif v in self.cells:
                out.append(self.cells[v])
            else:
                out.append(str(v))
        return out



class ndvar(object):
    _stype_ = "ndvar"
    _dim_order = ('time', 'sensor', 'freq')
    def __init__(self, dims, data, properties=None, name="???", info=""):
        """
        Arguments
        ---------
        
        For each agument, the example assumes you are importing 600 epochs of 
        EEG data for 80 time points from 32 sensors.
        
        data : array
            the first dimension should contain cases, and the subsequent 
            dimensions should correspond to the ``dims`` argument. E.g., 
            ``data.shape = (600, 80, 32).
        
        dims : tuple
            the dimensions characterizing the shape of each case. E.g., 
            ``(var('time', range(-.2, .6, .01)), sensor_net)``.
        
        properties : dict
            data properties dictionary
        
        
         .. note::
            ``data`` and ``dims`` are stored without copying. A shallow
            copy of ``properties`` is stored. Make sure the relevant objects 
            are not modified externally later.
        
        """
        try:
            dim_is = tuple(self._dim_order.index(dim.name) for dim in dims)
        except ValueError:
            raise ValueError("%r contains invalid dimension. Use %r" % (dims, self._dim_order))
        
        if len(dims) > 1:
            assert np.diff(dim_is).min() > 0
        
        # check data shape
        self.ndim = ndim = len(dims)
        if ndim != data.ndim - 1:
            raise ValueError("Dimension mismatch (data: %i, dims: %i)" % (data.ndim - 1, self.ndim))
        
        # interpret dims
        for dim in dims:
            if dim.name == 'time':
                self.time = dim
            elif dim.name == 'sensor':
                self.sensor = dim
        
        self._dim_dict = dict((dim.name, i) for i, dim in enumerate(dims))
        
        # store attributes
        self.dims = dims
        self.data = data
        self._len = len(data)
        if properties is None:
            self.properties = {}
        else:
            self.properties = properties.copy()
        self.name = name
        self.info = info
        
        # check dimensions
        for i in xrange(-1, -ndim, -1):
            n_data = data.shape[i]
            dim = dims[i]
            n_dim = len(dim)
            if n_data != n_dim:
                raise ValueError("Dimension %r length mismatch: %i in data, "
                                 "%i in dimension" % (dim, n_data, n_dim))
    
    def __add__(self, other):
        data = self.data + other.data
        name = '+'.join((self.name, other.name))
        return ndvar(self.dims, data, properties=self.properties, name=name)

    def __getitem__(self, index):
        if np.iterable(index):
            data = self.data[index]
            if data.shape[1:] != self.data.shape[1:]:
                raise NotImplementedError("Use subdata method")
            return ndvar(self.dims, data, properties=self.properties, name=self.name)
        else:
            index = int(index)
            return self.get_epoch(index)
    
    def __len__(self):
        return self._len
    
    def __repr__(self):
        rep = '<ndvar %(name)r: %(info)r -- %(n_cases)i cases, %(dims)s>'
        dims = ' X '.join('%r(%i)' % (dim.name, len(dim)) for dim in self.dims)
        args = dict(name=self.name, info=self.info, n_cases=self._len, dims=dims)
        return rep % args
    
    def __sub__(self, other):
        if hasattr(other, 'data'):
            data = self.data - other.data
            name = '-'.join((self.name, other.name))
        elif np.isscalar(other):
            data = self.data - other
            name = '%s-%r' % (self.name, other)
        else:
            raise ValueError("can't subtract %r" % other)
        return ndvar(self.dims, data, properties=self.properties, name=name)
    
    def assert_dims(self, dims):
        dim_names = tuple(dim.name for dim in self.dims)
        if dim_names != dims:
            raise DimensionMismatchError(self, dims)
    
    def copy(self):
        "returns a copy with a view on the object's data"
        data = self.data
        return self.__class__(self.dims, data, self.properties, self.name)
    
    def deepcopy(self):
        "returns a copy with a deep copy of the object's data"
        data = self.data.copy()
#        dims = tuple(dim.copy() for dim in self.dims)
        return self.__class__(self.dims, data, self.properties, self.name[:])
    
    def get_summary(self, func=None, name='{func}({name})'):
        """
        Returns a new ndvar with a single case summarizing all the cases in 
        the present ndvar. Normallt the styatistics used is the mean, but it
        can be customized through the `func` argument or the `'summary_func'`
        property.
        
        """
        if func is None:
            func = self.properties.get('summary_func', None)
        if func is None:
            func = np.mean
        
        data = func(self.data, axis=0)[None,...]
        name = name.format(func=func.__name__, name=self.name)
        info = os.linesep.join((self.info, 'summary: %s' % func.__name__))
        
        # update properties for summary
        properties = self.properties.copy()
        for key in self.properties:
            if key.startswith('summary_') and (key != 'summary_func'):
                properties[key[8:]] = properties.pop(key)
        
        return ndvar(self.dims, data, properties=properties, name=name, info=info)
    
    def get_epoch(self, Id, name="{name}[{Id}]"):
        "returns a single epoch (case) as ndvar"
        data = self.data[[Id]]
        name = name.format(name=self.name, Id=Id)
        epoch = ndvar(self.dims, data, properties=self.properties, name=name, 
                      info=self.info + ".get_epoch(%i)"%Id)
        return epoch
    
    def get_epoch_data(self, index=0):
        "returns the data for a single epoch (removing the 'case' dimension)"
        return self.data[index]
    
    def mean(self, name="mean({name})"):
        data = self.data.mean(axis=0)[None,...]
        name = name.format(name=self.name)
        # properties
        properties = self.properties.copy()
        for prop in ['ylim', 'colorspace']:
            if '%s_mean'%prop in properties:
                properties[prop] = properties.pop('%s_mean'%prop)
        return ndvar(self.dims, data, properties=properties, name=name)
    
    def subdata(self, time=None):
        """
        
        """
        try:
            t_dim = self._dim_dict['time']
            t_var = self.dims[t_dim]
        except KeyError:
            raise KeyError("Segment does not contain 'time' dimension.")
        
        has_cases = self._stype_ == 'ndvar'
        data = self.data
        properties = self.properties.copy()
#        dims = list(self.dims)
        rm_dims = []
        
        if np.isscalar(time):
            if time in t_var.x:
                i = np.where(t_var==time)[0][0]
            else:
                i_next = np.where(t_var > time)[0][0]
                t_next = t_var[i_next]
                i_prev = np.where(t_var < time)[0][-1]
                t_prev = t_var[i_prev]
                if (t_next - time) < (time - t_prev):
                    i = i_next
                    time = t_next
                else:
                    i = i_prev
                    time = t_prev
            
            if has_cases:
                data = data[:,i]
            else:
                data = data[i]
            rm_dims.append(t_dim)
            properties['t'] = time
        
        # cerate subdata object
        dims = tuple([dim for i,dim in enumerate(self.dims) if i not in rm_dims])
        out = self.__class__(dims, data, properties, self.name)
        
        # copy special overlay attribute that statictics functions add to certain epochs
        if hasattr(self, 'overlay'):
            out.overlay = self.overlay.subdata(time=time)
        
        return out



class dataset(dict):
    def __init__(self, name, *items, **named_items):
        """
        stores input items, does not make a copy
        
        default_DV : str
            name of the default dependent variable (DV). It is stored in the
            dataset's `default_DV` attribute, which can be modified safely 
            later.
        
        """ 
        self.name = name
        self.default_DV = named_items.pop('default_DV', None)
        dict.__init__(self)
        for item in items:
            name = item.name
            self.__setitem__(name, item, overwrite=False)
        for name, item in named_items.iteritems():
            self.__setitem__(name, item, overwrite=False)
    
    def __getitem__(self, name):
        """
        possible::
        
            >>> ds[9]        (int) -> case
            >>> ds[9:12]     (slice) -> subset with those cases
            >>> ds['MEG1']  (strings) -> var
            >>> ds['MEG1', 'MEG2']  (list of strings) -> list of vars; can be nested!
        
        """
#        if index in self:
        if isinstance(name, int):
            name = slice(name, name+1)
        
        if isinstance(name, slice):
            index = np.zeros(self.N, dtype=bool)
            index[name] = True
#            if name.start is None:
#                slbl = 
#            i_numbers = 
#            name = self.name + '[%s]'
            return self.subset(index)
#        if isstr(index):
        elif isinstance(name, (list, tuple)):
            out = []
            for item in name:
                out.append(self[item])
            return out
        else:
            return dict.__getitem__(self, name)
#        else:
#            items = dict((k, v[index]) for k, v in self.iteritems())
#            return dataset(**items)
    
    def __repr__(self):
        rep_tmp = "<dataset %(name)r N=%(N)i: %(items)s"
        items = []
        for key in sorted(self):
            v = self[key]
            if isinstance(v, var):
                lbl = 'V'
            elif isinstance(v, factor):
                lbl = 'F'
            elif isinstance(v, ndvar):
                lbl = 'Vnd'
            else:
                lbl = '?'
            
            if key == self.default_DV:
                name = '>%r<' % key
            else:
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
            if (not overwrite) and (name in self):
                raise KeyError("dataset already contains variable of name %r"%name)
            else:
                if len(self) == 0:
                    self.N = len(item)
                dict.__setitem__(self, name, item)
    
    def __str__(self):
        txt = str(self.as_table(cases=10, fmt='%.5g', midrule=True))
        if self.N > 10:
            note = "(use .as_table() method to see the whole dataset)"
            txt = ' '.join((txt, note))
        return txt
    
    def __add__(self, other):
        return self.as_epoch() + other.as_epoch()
    
    def __sub__(self, other):
        return self.as_epoch() - other.as_epoch()
    
    def add(self, item):
        "= dataset[item.name] = item"
        self[item.name] = item
    
#    def as_epoch(self, *args, **kwargs):
#        "returns an epoch of the default dependent variable (default_DV)"
#        return self[self.default_DV].as_epoch(*args, **kwargs)
#    
    def as_table(self, cases=0, fmt='%.6g', f_fmt='%s', match=None, sort=True,
                 midrule=False):
        r"""
        returns a fmtxt.Table containing all vars and factors in the dataset 
        (ndvars are skipped). Can be used for exporting in different formats
        such as csv.
        
        Arguments
        ---------
        
        cases : int
            number of cases to include
        fmt : str
            format string for numerical variables. Hast to be a valid 
            `format string,
            <http://docs.python.org/library/stdtypes.html#string-formatting>`
        f_fmt : str
            format string for factors (None -> code; e.g. `'%s'`)
        match : factor
            create repeated-measurement table
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
        
        keys = [k for k, v in self.iteritems() if not isndvar(v)]
        if sort:
            keys = sorted(keys)
        values = [self[key] for key in keys]
        
        table = fmtxt.Table('l'*len(keys))
        
        for name in keys:
            table.cell(name)
        
        if midrule:
            table.midrule()
        
        for i in range(cases):
            for v in values:
                if isfactor(v):
                    if f_fmt is None:
                        table.cell(v.x[i], fmt='%i')
                    else:
                        label = v.cells[v.x[i]]
                        table.cell(f_fmt % label)
                elif isvar:
                    table.cell(v.x[i], fmt=fmt)
        
        if cases < self.N:
            table.cell('...')
        return table
    
#    def deepcopy(self, index):
#        """
#        Like __getitem__, but makes a deep copy (i.e., all the data is 
#        duplicated, and the data in both copies can be manipulated without 
#        affecting the data in the other copy)
#        
#        """
#        if index in self:
#            return dict.__getitem__(self, index).copy()
#        elif len(index) == self.N:
#            items = dict((k, v[index]) for k, v in self.iteritems())
#            return dataset(**items)
    
    def export(self, fn=None, fmt='%.10g'):
        """
        Allows saving dataframes in different formats. 
        
        :kwarg str fn: target filename 
            with None (default) a system file dialog will be displayed
            otherwise, the extesion is used to determine the format:
             - 'txt':  tsv
             - 'tex':  as TeX table
             - 'pickle':  use pickle.dump (defunct)
        
        """
        if not isinstance(fn, basestring):
            fn = ui.ask_saveas(ext = [('txt', "Tab-separated values"),
                                      ('tex', "Tex table"),
                                      ('pickle', "Pickle")])
        ext = fn.split(os.extsep)[-1]
        if ext == 'pickle':
            pickle.dump(self, open(fn, 'w'))
        else:
            table = self.AsTable(fmt=fmt)
            if ext == 'txt':
                table.save_tsv(fn)
            elif ext =='tex':
                table.save_tex(fn)
            else:
                raise IOError("can only export .pickle, .txt and .tex")
    
#    def get_condition_pointers(self, factor, dep_var='MEG', exclude=[], name='{name}[{case}]'):
#        if isinstance(factor, basestring):
#            factor = self[factor]
#        
#        out = {}
#        for case in factor.cells.values():
#            if case not in exclude:
#                setname = name.format(name=self.name, case=case)
#                index = factor == case
#                out[case] = ConditionPointer(self, dep_var, index, setname)
#        return out
    
    def get_case(self, i):
        return dict((k, v[i]) for k, v in self.iteritems())
    
    def get_subsets_by(self, factor, default_DV=None, exclude=[], name='{name}[{case}]'):
        """
        convenience function; splits the dataset by the cells of a factor and 
        returns as dictionary of subsets.
        
        """
        if isinstance(factor, basestring):
            factor = self[factor]
        if default_DV is None:
            default_DV = self.default_DV
        
        out = {}
        for case in factor.cells.values():
            if case not in exclude:
                setname = name.format(name=self.name, case=case)
                index = factor == case
                out[case] = self.subset(index, setname, default_DV=default_DV)
        return out
    
    def get_summary(self, func=None, name='{func}({name})'):
        """
        -> self[self.default_DV].get_summary(func=func, name=name) 
        (convenience function for submitting conditon datasets to plotting)
        
        """
        return self[self.default_DV].get_summary(func=func, name=name)
    
    def itercases(self):
        for i in xrange(self.N):
            yield self.get_case(i)
    
    def mark_by_threshold(self, DV=None, threshold=2e-12, above=True, below=False, 
                          target='reject'):
        """
        Marks epochs based on a threshold criterion (any sensor exceeding the 
        threshold at any time) 
        
        above: True, False, None
            How to mark segments that exceed the threshold: True->good; 
            False->bad; None->don't change
        below:
            Same as ``above`` but for segments that do not exceed the threshold
        threshold : float
            The threshold value.
        target : factor or str
            Factor (or its name) in which the result is stored. If ``var`` is 
            a string and the dataset does not contain that factor, it is 
            created.
        
        """
        if DV is None:
            DV = self.default_DV
            if DV is None:
                raise ValueError("No valid DV")
        if isinstance(DV, basestring):
            DV = self[DV]
        
        # get the factor on which to store results
        if isfactor(target) or isvar(target):
            assert len(target) == self.N
        elif isinstance(target, basestring):
            if target in self:
                target = self[target]
            else:
                x = np.zeros(self.N, dtype=bool)
                target = var(x, name=target)
                self.add(target)
        else:
            raise ValueError("target needs to be a factor")
        
        # do the thresholding
        if isndvar(DV):
            for ID in xrange(self.N):
                data = DV.get_epoch_data(ID)
                v = np.max(np.abs(data))
                
                if v > threshold:
                    if above is not None:
                        target[ID] = above
                elif below is not None:
                    target[ID] = below
        else:
            for ID in xrange(self.N):
                v = DV[ID]
                
                if v > threshold:
                    if above is not None:
                        target[ID] = above
                elif below is not None:
                    target[ID] = below
    
    @property
    def shape(self):
        return (self.N, len(self))
    
    def subset(self, index, name='{name}', default_DV=None):
        """
        Returns a dataset containing only the subset of cases selected by 
        `index`.
        
        """
        items = dict((k, v[index]) for k, v in self.iteritems())
        name = name.format(name=self.name)
        
        if default_DV is None:
            default_DV = self.default_DV
        items['default_DV'] = default_DV
        
        return dataset(name, **items)







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
                if f not in self:
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
        _regressor_.__init__(self, name, False)
        self.beta_labels = beta_labels
        
        self.df =  reduce(operator.mul, [f.df for f in self.base])
        
        # determine cells:
        label_dicts = [f.cells for f in self.factors if f._stype_=='factor']
        self.cells = _permutate_address_dicts(label_dicts)
        self.indexes = sorted(self.cells.keys())
        self.colors = {}
        
        # effect coding
        codelist = [f.as_effects for f in self.base]
        codes = reduce(_effect_interaction, codelist)
        self.as_effects = codes
    
    def __repr__(self):
        names = [f.name for f in self.base]
#        txt = "interaction({n})"
#        return txt.format(n=', '.join(names))        
        return ' % '.join(names)
    
    @property
    def _nestedin(self):
        return set.union([set(f.nestedin) for f in self.factors])
    
    def __getitem__(self, sub):
        out = [f[sub] for f in self.base]
        if np.iterable(sub):
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




""" #####     #####     #####     #####     #####     #####     #####     #####
Factors
-------

"""


def factor_from_comp(comp, name=None):
    if not isstr(name):
        name = 'comp'
    out = factor(comp, name=name, labels={0:'False', 1:'True'})
    return out


def factor_from_dict(name, key_factor, values_dict):
    """
    Creates a factor containing a value defined in values_dict for each 
    category in key_factor.
    
    """
    x = [values_dict[label] for label in key_factor.as_labels()]
    return factor(x, name=name)


def var_from_dict(name, key_factor, values_dict, default=0):
    """
    Creates a variable containing a value defined in values_dict for each 
    category in key_factor.
    
    """
    x = np.empty(len(key_factor))
    x[:] = default
    for k in key_factor.as_labels():
        x[key_factor==k] = values_dict[k]
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
        _regressor_.__init__(self, name, False)
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
        # convert vars to factors
        clean_factors = []
        for f in factors:
            if isvar(f):
                ux = np.unique(f.x)
                if all([int(y)==y for y in ux]):
                    ux = ux.astype(int)
                labels = dict(zip(ux, ux.astype('S10')))
                f = factor(f.x, labels=labels, random=f.random)
            if isfactor(f):
                clean_factors.append(f)
            else:
                raise ValueError("Can only create multifactor from categorial factors")
        factors = clean_factors
        
        self.N = factors[0].N
        
        #
        if len(factors) == 1:
            f = factors[0]
            self.x = f.x
            self.cells = f.cells.copy()
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
            self.cells = labels
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
        :rtype: :class:`psystats.fmtxt.Table`
        
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


def _split_Y(Y, X, match=None, sub=None, datalabels=None):
    """
    returns 2  (factor, model of factors)
    Y       dependent measurement
    X       factor model
    match   factor on which cases are matched for repeated measures comparisons
    sub     Bool Array of len==N specifying which cases to include
    
    
    out
    ---
    data:   lists with those values in Y (ndarray, factorm var) which lie in 
            each cell defined by X
    data_labels: 
    names: 
    within: (bool)
    """
    # prepare input
    if type(Y) == nonbasic_effect:
        raise ValueError("Y cannot be nonbasic_effect")
    elif not isfactor(Y):
        Y = asvar(Y)
        
    if X is None:
        X = model(factor([0]*Y.N))
    else:
        X = asmodel(X) # make sure X is list of factors
    
    assert Y.N == X.N
    
    # sub
    if sub is not None:
        Y = Y[sub]
        X = X[sub]
        if match:
            match = match[sub]
        if datalabels:
            datalabels = datalabels[sub]
    X = X.factors
    Y = Y.x
    mf = multifactor(X)
    
    # prepare data labels
    if datalabels:
        datalabels = datalabels.as_labels()
        do_dlbls = True
    elif match:
        datalabels = match.as_labels()
        do_dlbls = True
    else:
        datalabels = None
        do_dlbls = False
    data = []
    data_labels = []
    names = []
    sorted_indexes = [] # for repeated measures
    
    ## Collect Data ######
    for n, i in mf.iter_n_i(): # cells
        cell_data = Y[i]
        if do_dlbls: cell_labels = datalabels[i]
        if match:
            indexes = match.x[i] # vp ids in cell
            i_sorted = sorted(np.unique(indexes))
            sorted_indexes.append(i_sorted)
            if len(i_sorted) == len(indexes):
                indexes_argsort = np.argsort(indexes)
                cell_data = cell_data[indexes_argsort]
                if do_dlbls: cell_labels = cell_labels[indexes_argsort]
            else:
                d_list = []
                l_list = []
                for j in i_sorted:
                    j_indexes = np.where(indexes==j)[0]
                    d_list.append(np.mean(cell_data[j_indexes]))
                    if do_dlbls: # get label
                        label = cell_labels[j_indexes[0]]
                        if len(j_indexes) > 1:
                            if any([label != cell_labels[li] for li in j_indexes[1:]]):
                                raise ValueError("cell label mismatch -- combining cells of different grouping blargh")
                        l_list.append(label) # will not notice if labels differ between cases!
                cell_data = np.array(d_list)
                if do_dlbls: cell_labels = np.array(l_list)
        if len(cell_data) > 0: # drop empty cells (in case of empty dict entries in factors)
            data.append(cell_data)
            names.append(n)
            if do_dlbls: data_labels.append(cell_labels)
    # determine repeated measures status
    within = False
    if match:
        icomp = sorted_indexes.pop(0)
        if all([i==icomp for i in sorted_indexes]):
            within = True
            logging.debug("SPLIT Y: repeated measures")
        else:
            logging.debug("SPLIT Y: independent measures")
#    out = {'data': data,
#           'data_labels': data_labels,
#           'names': names,
#           'within': within}
#    return out
    return data, data_labels, names, within



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
        label = link.join(label_components)
        label_dic[index] = label
    
    return label_dic

