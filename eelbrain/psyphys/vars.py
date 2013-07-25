"""
Classes for managing variables associated with segments

VarCommander:
    stores variable properties like data type and value labels for nominal variables

Colony:
    act like {variable: value} dictionaries
    Colony[Var] returns value
    Colony[Var, 'l'] returns string label for nominal variables
    Colony[Var, 's'] returns string description for all variables
    .
    .
    .

Address:
    an filter Colonies


- each segment has a VarColony.
- The Experiment has a VarMothership which manages all the variables and keeps
    track of VarColonies.
- The VarMothership has VarCommanders, each of which manages one variable.

- Address objects can efficiantly retrieve sets or dictionaries of segments

"""

import os

import numpy as np
import matplotlib.pyplot as P
# from sympy import Interval

from .. import fmtxt
from ..utils.basic import loadtable



# TODO: Parasite that maps one variable to another (e.g. subject-->age)

def isvar(x):
    """
    returns True if x is a Variable type (VarCommander or derivative)

    """
#    types = [VarCommander, Parasite]
#    return any(isinstance(x, t) for t in types)
# to avoid problem with import as eelbrain.biovars.VarCommander
    tests = [hasattr(x, 'values'),
             hasattr(x, 'labels')]
    return all(tests)

def isaddress(x):
    conditions = (hasattr(x, 'contains'),
                  hasattr(x, 'filter'))
    return all(conditions)

def asaddress(x):
    if not isaddress(x):
        x = Address(x)
    return x

def iscollection(x):
    "True if x is list, tuple, set..."
    return type(x) in [list, tuple, set]

def makeset(x):
    if not iscollection(x):
        x = [x]
    return set(x)


class _Aset(object):
    """
    defines a set of numbers

    """
    def __init__(self, include=[], exclude=[],
                 g=-np.inf, geq=True,
                 l=np.inf, leq=False,):
        """
        var:    variable
        op:     operaton

        evaluation priority:
        - exclude
        - include
        - comparators
        """
        if g >= l:
            raise ValueError("empty set (g >= l)")
        self.include = set(include)
        self.exclude = set(exclude)
        self.g = g
        self.geq = geq
        self.l = l
        self.leq = leq
    def contains(self, x):
        "returns True if x is in self, False if not"
        if x in self.exclude:
            return False
        elif x in self.include:
            return True
        else:
            if self.geq:
                out = (x >= self.g)
            else:
                out = (x > self.g)

            if out == True:
                if self.leq:
                    return (x <= self.l)
                else:
                    return (x < self.l)
            else:
                return False
    def union(self, other):
        raise NotImplementedError()
    def intersect(self, other):
        raise NotImplementedError()


class ASet(dict):
#    def __repr__(self):
#        temp = "{d}"
#        return temp.format(d=dict.__repr__(self))
    def __init__(self, address):
        for var, i in address.iteritems():
            assert isvar(var), "%s not a variable" % var
            assert type(i) is Interval
        dict.__init__(self, address)
    def contains(self, colony):
        for var, i in self.iteritems():
            if colony[var] in i:
                continue
            else:
                return False
        return True


class NewAddress(object):
    """
    variable -> Interval


    !!!
    IQ<40 + subjects==[1,2,3]
    -> subjects 1,2,3 in addition to all with IQ<40
       * IQ>20
       ->subjects 1,2,3 are added separately, but I need to
         remember only taking them if IQ>20
         -> IQ must appear 2 times

    trial==3 * subjects==[1,2,3]
    -> trial 3 only for subjects 1,2,3


    -> list of address-dicts
    .contains(colony)


    -> for each dict entry, I need to keep in mind whether it is additive
       or multiplicative

    self._alist: list of additive

    self * x  ->  each element of _alist * x
    self + x  ->  if vars in each element: modify var
              ->  else: add to _alist
    self - x == self * (-x)

    """
    def __init__(self, address):
        """
        address: list of {Variable: interval} dictionaries

        """
        self._alist = []
        for a in address:
            if isinstance(a, ASet):
                self._alist.append(a)
            else:
                self._alist.append(ASet(a))
    @property
    def vars(self):
        return self._address.keys()
    def contains(self, colony):
        for aset in self._alist:
            if colony in aset:
                return True
        return False
    def isin(self, colony):
        return self.contains(colony)
    def __add__(self, other):
        "union"
        v1 = set(self.vars)
        v2 = set(other.vars)
        vdouble = v1.intersect(v2)

        address = self._address.copy()
        address.update(other._address)

        for v in vdouble:
            i1 = self[v]
            i2 = other[v]
            i = i1.union(i2)
            address[v] = i

        return Address(address)
    def __mul__(self, other):
        "intersection"
        v1 = set(self.vars)
        v2 = set(other.vars)
        vdouble = v1.intersect(v2)

        address = self._address.copy()
        address.update(other._address)

        for v in vdouble:
            i1 = self[v]
            i2 = other[v]
            i = i1.union(i2)
            address[v] = i

        return Address(address)

    def address(self, colony):
        if self.isin(colony):
            a = tuple([colony[var] for var in self.keys()])
            return a
        else:
            return None
    def label(self, index):
        lbls = []
        if hasattr(index, 'keys'):
            index = self.address(index)
        for i, var in zip(index, self.keys()):
            lbls.append(var.label(i))
        return ' '.join(lbls)
    # sorting
    def filter(self, colonies):
        return [ c for c in colonies if self.isin(c) ]
    def dict(self, colonies, warn_on_nonunique=False):
        dictionary = {}
        for c in colonies:
            if self.isin(c):
                index = tuple([ c[var] for var in self.keys() ])
                if index in dictionary:
                    if self.warn_on_nonunique:
                        print "Warning: non-unique address: {0}".format(index)
                    dictionary[index].append(c)
                else:
                    dictionary[index] = [c]
        return dictionary
    def dict_over_(self, colonies, over):
        dictionary = {}
        for c in colonies:
            if self.isin(c):
                index = c[over]
                if index in dictionary:
                    dictionary[index].append(c)
                else:
                    dictionary[index] = [c]
        return dictionary
#    def name(self):
#        name = []
#        for var, values in self.iteritems():
#            if values[0]:
#                name.append( ' & '.join([ var[val] for val in values[1] ]) )
#        return ' '.join(name)





class Address(dict):  # {var:[valid, set()], ... }
    """
    create with Address(addressDict)

    """
    def __init__(self, address):
        """
        Is usually easier to initialize via Variable operations, e.g.
        >>> a = (Cell == [1, 2]) * (subject != [1, 3])

         + --> union
         * --> intersection


        address (Parameter Specification)
        -----------
        var                          --> {var: True} (allow all values)
        [var1, var2, ...]            --> all {varX: True}
        {var: value}                 --> include only value
        {var: [values...]}           --> include values
        {var: (valid, [values...])}  --> (valid = bool, values = list of values)


        e.g.
        >>> Address({ subject:(False, [10, 12, 18]) })
        to exclude subjects 10, 12, and 18

        help
        ----
        call Address.keys() for address schema

        """
        dict.__init__(self)
        # convert non-dict input
        if isvar(address):
            address = {address: True}
        elif type(address) in [tuple, list]:
            for var in address:
                if not isvar(var):
                    raise ValueError("Address initialized with list contiaining" + \
                                     "item which is no a variable")
            address = dict([(var, True) for var in address])
        elif not type(address) == dict:
            raise ValueError("Invalid address parameter")
        # interpret the dic
        for k, v in address.iteritems():
#            logging.debug(str(v))
            # catch global addresses (all vales in var)
#            if type(v)==tuple and v==(False, None):
#                v = None
            if (v == None) or (type(v) == bool):
                self[k] = (False, None)
            else:
                # translate single values -> collections
                if np.isscalar(v) or isinstance(v, basestring):
                    v = [v]
                # interpret (bool, [collection]) case
                if type(v[0]) == bool:
                    valid, v = v
                    if np.isscalar(v) or isinstance(v, basestring):
                        v = [v]
                else:
                    valid = True
                # convert labels to values
                if v is None:
                    self[k] = (False, None)
                else:
                    v = k.values(v)
                    self[k] = (valid, set(v))
    def __repr__(self):
        txt = "Address(%s)"
        items = []
        temp = "{var}{sign}{val}"
        if len(self) > 1:
            temp = temp.join(['(', ')'])
        for var, (include, values) in self.iteritems():
            if values:
                sign = ['!=', '=='][include]
                items.append(temp.format(var=var.name,  # __repr__(),
                                         sign=sign,
                                         val=list(values).__repr__()))
            else:
                items.append(var.name)  # __repr__())
        return txt % ' + '.join(items)
    def __str__(self):
        txt = [ '<Address>:' ]
        for k, v in self.iteritems():
            txt.append('\n\t' + k.name + ': ')
            if not v[0]:
                txt.append('NOT ')
            txt.append(str(v[1]).lstrip('set(').rstrip(')'))
        return ''.join(txt)
    def __add__(self, other):
        "union"
        # check input:
        if isvar(other):
            other = Address(other)
        else:
            assert isaddress(other), "cannot combine address and %s" % type(other)
        # combine
        out = {}
        out.update(self)
        for k in other.keys():
            if k in out:
                raise NotImplementedError()
            else:
                out[k] = other[k]
        return Address(out)
    def __mul__(self, other):
        "intersection"
        # check input:
        if isvar(other):
            other = Address(other)
        else:
            assert isaddress(other), "cannot combine address and %s" % type(other)
        # combine
        new = {}
        for key in set(self.keys() + other.keys()):
            if key in self and key in other and not key in new:
                st, s = self[key]; ot, o = other[key]
                if st and ot:
                    new[key] = (True, s.intersection(o))
                elif st:
                    new[key] = (True, s.difference(o))
                elif ot:
                    new[key] = (True, o.difference(s))
                else:
                    new[key] = (False, s.union(o))
            elif key in self:
                new[key] = self[key]
            elif key in other:
                new[key] = other[key]
        return Address(new)
    def isin(self, colony):
        return self.contains(colony)
    def contains(self, colony):
        for var, v in self.iteritems():
            if v[1] == None:
                continue
            elif v[0] == True:
                if not colony[var] in v[1]:
                    return False
            else:
                if colony[var] in v[1]:
                    return False
        return True
    def index(self, colony):
        if self.isin(colony):
            a = tuple([colony[var] for var in self.keys()])
            return a
        else:
            return None
    def address(self, colony):
        return self.index(colony)
    def label(self, index):
        "returns a label for the colony/index"
        lbls = []
        # if it is a colony, et index
        if hasattr(index, 'keys'):
            index = self.address(index)
        # crate label
        for i, var in zip(index, self.keys()):
            lbls.append(var.label(i))
        return ' '.join(lbls)
    def color(self, index):
        """
        Returns a color for the colony/index. Current implementation loops
        through address components and returns the earliest color it can find.

        """
        # if it is a colony, et index
        if hasattr(index, 'keys'):
            index = self.address(index)
        # crate label
        for i, var in zip(index, self.keys()):
            color = var.get_color_for_value(i)
            if color is not None:
                return color
    def short_key_labels(self):
        """
        returns a {VarCommander: short_label, ...} dictionary with labels that
        are as short as possible to be unique among each other

        """
        f_names = {}
        for var in self.keys():
            i = 1
            while var.name[:i] in f_names.values() + ['e']:
                i += 1
            f_names[var] = var.name[:i]
        return f_names
    def short_labels(self, link=''):
        """
        returns a dictionary containing short labels. link is the string
        inserted between label parts referring to different variables.

        """
#        keys = []
        label_dicts = []
        for var in self.keys():
#            keys.append(var.dictionary.keys())
            values = var.dictionary.values()
            l = 1
            while len(np.unique([v[:l] for v in values])) < len(values):
                l += 1
            d = dict((k, v[:l]) for k, v in var.dictionary.iteritems())
            label_dicts.append(d)

        indexes = [()]  # all possible indexes
        for label_dict in label_dicts:
            newindexes = []
            for i in indexes:
                for k in label_dict.keys():
                    newindexes.append(i + (k,))
            indexes = newindexes

        label_dic = {}
        for index in indexes:
            label_components = []
            for k, ld in zip(index, label_dicts):
                label_components.append(ld[k])
            label = link.join(label_components)
            label_dic[index] = label

        return label_dic
    # sorting
    def filter(self, colonies):
        return [ c for c in colonies if self.contains(c) ]
    def sort(self, colonies, warn_on_nonunique=False):
        """
        sorts colonies into a dictionary according to their values on self

        """
        dictionary = {}
        for c in colonies:
            if self.contains(c):
                address = self.address(c)
                if address in dictionary:
                    dictionary[address].append(c)
                    if warn_on_nonunique:
                        print "Warning: non-unique address: {0}".format(address)
                else:
                    dictionary[address] = [c]
        return dictionary
    def dict(self, colonies, warn_on_nonunique=False):
        print "Address.dict DEPRECATED; use sort_colonies"
        return self.sort(colonies, warn_on_nonunique)
    def dict_over_(self, colonies, over):
        """
        uses all colonies falling inside self and returns a dictionary
        over_value -> [colonies...]

        """
        dictionary = {}
        for c in colonies:
            if self.contains(c):
                index = over.index[c]
                if index in dictionary:
                    dictionary[index].append(c)
                else:
                    dictionary[index] = [c]
        return dictionary
    @property
    def name(self):
        name = []
        for var, values in self.iteritems():
            if values[0]:
                name.append('u'.join([ var[val] for val in values[1] ]))
        return ' '.join(name)



'''
class VarComparator(object):
    """
    returned by comparisons involving VarCommander objects, used to access
    subsets of VarTables

    """
    def __init__(self, op
'''


class VarCommander(object):
    """
    is one variable, has links to its occurrences

    Methods
    -------
    value(v) - return code of value
    label(v) - return str representation of value
    values(v) - can handle list input and always returns list
    labels(v) - "


    Behavior:
    values assigned to a colony are registered with
    self._registerValue_forColony_.


    self.dtype  ='dict'  (or dtype)
        governs dtype of retrieved values. other dtypes
        values can be saved (e.g. save float in int var, later
        convert var to float)
    self.dict_enabled
        if dict_enabled, Colonies can be assigned strings
        (VarCommand automatically looks up value in
        dictionary)

    """
    def __init__(self, name, mothership, dtype='dict', dtype_enforce=True, random=False):
        """
        kwargs:
        dtype ='dict'   1) 'dict' for var that stores nominal data (ints as-
                        sociated with strings in self.dictionary.
                        2) Any dtype that supports conversion upon calling
                        3) None: don't interfere with values
        dtype_enforce =True:   if True, values that are assigned to a colony are
                        converted to self.dtype. If False, values are stored un-
                        modified but converted when called. That way, floats
                        stored in an int VarCommander don't loose precision.

        """
        self.name = name
#        self.shortcut=name[0]
        self.dtype_enforce = dtype_enforce
        if dtype == 'dict':
            self.dtype = int
            self.dictionary = {}
            self.dict_enabled = True
        else:
            self.dtype = dtype  # int, float, string
            self.dictionary = None
            self.dict_enabled = False
        self.default_value = np.nan
        self.mothership = mothership
        self.random = random
        self.parasites = []
        self.unique_values = []  # DO NOT USE DIRECTLY; use self.uniqueValues property
        self.unique_values_needs_update = False
        # for colors
        self._color_dict = {}
        self._cm_min = 0
        self._cm_max = 1
        self._cmap = P.cm.jet
    def __repr__(self):
        temp = '<{c}:  "{n}", {dt}>'
        if self.dict_enabled:
            dt = "\'dict\', labels=%s" % self.dictionary
        else:
            dt = self.dtype.__name__
        return temp.format(c=self.__class__.__name__, n=self.name, dt=dt)
    def __str__(self):
        text = '{c} "{n}", {dt}'
        if self.dict_enabled:
            dt = "\'dict\'"
        else:
            dt = self.dtype.__name__
        out = [text.format(c=self.__class__.__name__, n=self.name, dt=dt)]
        if self.dict_enabled:
            keylen = max(len(str(key)) for key in self.dictionary.keys())
            temp = "{k}: {v}"
            for key, value in self.dictionary.iteritems():
                out.append(temp.format(k=str(key).rjust(keylen), v=str(value)))
        return os.linesep.join(out)
    # Address Creation
    def __eq__(self, value):
        if isvar(value):
            return id(self) == id(value)
        elif value is True:
            return Address({self: True})
        else:
            return Address({self: [True, value]})
    def __ne__(self, value):
        if isvar(value):
            return id(self) != id(value)
        else:
            return Address({self: [False, value]})
    def __lt__(self, other):
        return NotImplemented
    def __le__(self, other):
        return NotImplemented
    def __gt__(self, other):
        return NotImplemented
    def __ge__(self, other):
        return NotImplemented
    def __cmp__(self, other):
        return NotImplemented
    # combination
    def __add__(self, other):
        return Address(self) + other
    def __mul__(self, other):
        return Address(self) * other
    # dictionary
    @property
    def dictionary_reversed(self):
        out = dict(zip(self.dictionary.values(), self.dictionary.keys()))
        return out
    def intrep(self, values):
        """
        takes a value or a list of values; transforms any string elements into
        the corresponding int indexes using a reverse dictionary.
        """
        if type(values) not in [str, set, list, tuple]:
            return values
        elif isinstance(values, basestring):
            return self.dictionary_reversed[values]
        else:
            if not any([type(v) == str for v in values]):
                return values
            in_type = type(values)
            revdic = self.dictionary_reversed
            out = []
            for v in values:
                if type(v) == str:
                    out.append(revdic[v])
                else:
                    out.append(v)
            return out
    # # accessing data for DISPLAY
    # if int access dictionary, else access varColonies
    #
    def count(self, colonies=None, var=None, hspace=6):
        """
        Counts number of instances per value. var can be another VarCommander
        to get tables

        parameters
        ----------
        colonies: Dataset; iterable colonies container; if None, all colonies are used
        var: other VarCommander to create cells to count

        """
        # table = tex.Table()
        # for key, value in self.dictionary.iteritems():
        #    table.Cell()
        if colonies == None:
            colonies = self.mothership.colonies
        # TODO: use tex.Table
        headlen = max([len(val) for val in self.dictionary.values()]) + 11
        text = [self.name,
                ': VarCommander',
                '\n    dtype: %s' % (self.dtype) ]  # (self.scaleAsString) ]
        if var != None:
            text.append(('\n' + ' ' * headlen + 'n(%s):' % var.name))
        text.append('\n Dictionary:'.ljust(headlen))
        if var != None:
            for name in var.dictionary.values():
                text.append(name.rjust(hspace))
        for key, value in self.dictionary.iteritems():
            text.append(('\n    %s: %s' % (key, value)).ljust(headlen))
            if var != None:
                for val2 in var.dictionary.keys():
                    n = len([colony for colony in colonies if colony[self] == key and colony[var] == val2])
                    text.append(('%s' % n).rjust(hspace))
            else:
                n = len([colony for colony in colonies if colony[self] == key ])
                text.append(('%s' % n).rjust(hspace))
        print ''.join(text)
    # accessing data for use
    def __getitem__(self, name):
        """
        for accessing variable values, use V.label(colony) and V.value(colony)
        use

        name:
         varColony --> returns value
         value --> returns string label
         if name is value and not dict_enabled: returns name
        """
        # logging.debug("type of name: %s"%type(name))
        if isinstance(name, basestring) or np.isscalar(name):
            return self.repr(name)
        elif type(name) in [list, dict]:
            return self.reprs(name)
        else:
            return name[self]
        # elif self.dict_enabled:
        #    if np.iterable(name):
        #        return [self.dictionary[n] for n in name]
        #    else:
        #        return self.dictionary[name]
        # else:
        #    return name
    def value(self, value):
        "always returns numerical representation (code or value)"
        if isinstance(value, dict):
            value = value[self]
        if self.dict_enabled and isinstance(value, basestring):
            value = self.dictionary_reversed[value]
        # TODO: other dtypes call
        return value

    def repr(self, value):
        "returns label string for <dict> vars, and value for others"
        if isinstance(value, dict):
            value = value[self]
        if self.dict_enabled:
            return self.label(value)
        else:
            return value

    def label(self, value):
        "always returns string (label or str(value))"
        if isinstance(value, dict):
            value = value[self]
        if isinstance(value, basestring):
            out = value
        else:
            if self.dict_enabled and self.dictionary.has_key(value):
                out = self.dictionary[value]
            else:
                out = str(value)
        return out

    def reprs(self, values):
        if isinstance(values, basestring) or np.isscalar(values):
            values = [values]
        return [self.repr(e) for e in values]

    def values(self, values):
        if isinstance(values, basestring) or np.isscalar(values):
            values = [values]
        return [self.value(e) for e in values]

    def labels(self, values):
        if isinstance(values, basestring) or np.isscalar(values):
            values = [values]
        return [self.label(e) for e in values]

    def val_for_label(self, labels):
        """
        in: label or list of labels
            - list is more efficient than multiple single value calls
        out: value or list of values

        """  # ???: should list labels be allowed?
        revDict = self.dictionary_reversed
        if type(labels) in (list, tuple, set):
            return [revDict[l] for l in labels]
        else:
            return revDict[labels]

    def get_color_for_colony(self, colony):
        return self.get_color_for_value(colony[self])

    def get_color_for_value(self, value):
        """
        uses:
        self._color_dict:  value -> color mapping
        self._cm_min  --> mapping on self._color_map
        self._cm_max
        self._cmap: matplotlib cmap
        """
        if self.dict_enabled:
            if isinstance(value, basestring):
                value = self.val_for_label(value)
            if value in self._color_dict:
                return self._color_dict[value]
            else:
                return None
        else:
            v = (value - self._cm_min) * (self._cm_max / float(self._cmap._i_under))
            return self._cmap(v)

    def set_color_for_value(self, value, color):
        if isinstance(value, basestring):
            value = self.val_for_label(value)
        self._color_dict[value] = color

#    def set_dict_enabled(self, v=True):
#        self.dict_enabled = bool(v)
#        if self.dict_enabled and not self.dictionary:
#            self.dictionary = {}

    def __setitem__(self, name, value):
        if isinstance(name, basestring):
            key = self.dictionary_reversed[name]
            self[key] = value
        elif isinstance(name, int):
            if self.dict_enabled:
                self.dictionary[name] = value
            else:
                raise ValueError("dict is not enabled for %r" % self)
        else:
            raise KeyError("%s" % name)

    # # handling value assignment
    def _registerValue_forColony_(self, value, colony):
        "OLD"
        # if an old value gets overwritten, unique_values might have changed,
        # --> set flag to recheck when unique values are requested
        if colony[self] != self.default_value:
            self.unique_values_needs_update = True
        # handle value
    def _register_value_(self, value):
        """
        Converts the value to a value that is valid for this variable and
        returns the new value. For nominal varibales, the dictionary is
        aso updated.

        """
        if self.dtype:
            try:  # np.isnan raises an error for certain dtypes
                if (value is None) or np.isnan(value):
                    return np.nan
            except:
                pass

            if self.dict_enabled:
                if isinstance(value, basestring):
                    value = unicode(value)
                elif not isinstance(value, int):
                    try:
                        value = int(value)
                    except:
                        raise ValueError("Inappropriate Value For Dict Variable"
                                         ": %r with type %r" % (value, type(value)))

                if isinstance(value, int):
                    if value not in self.dictionary:
                        self.dictionary[value] = unicode(value)
                else:  # type(value) is str:
                    value = self.code_for_label(value, add=True)
            elif self.dtype_enforce:
                value = self.dtype(value)
        # update unique_values list
        if value not in self.unique_values:
            self.unique_values.append(value)
        # send back corrected value
        return value
    def code_for_label(self, label, add=False):
        """returns the code of the label in self.dictionary; if add=True,
        nonexistent labels are added. """
        if label in self.dictionary.values():
            # probably more efficient than reversing the whiole dict
            i = self.dictionary.values().index(label)
            return self.dictionary.keys()[i]
        elif add:
            n = max([-1] + self.dictionary.keys()) + 1
            self.dictionary[n] = label
            return n
        else:
            raise KeyError
    # # summaries
    @property
    def uniqueValues(self):
        if self.unique_values_needs_update:
            self.unique_values = np.unique([colony[self] for colony in self.mothership.colonies])
            self.unique_values_needs_update = False
        return self.unique_values
    def uniqueValuesForColonies_(self, segments):
        return np.unique([s[self] for s in segments])
    # plotting
    def correlation(self, *indexes):
        """
        -1 is self
        """
    def plot(self, segments=[]):
        pass





class VarMothership(object):
    """
    - self[varName] returns varCommander with name
    - iter cycles through carCommanders


    property variables
    ------------------
    variables for recurring things like time, duration, etc. retrieve/create
    through E.variables.get(name).

    """
    property_var_dtypes = {'time': float,
                           'duration': float,
                           'magnitude': float,
                           'subject': 'dict',
                           'cell': 'dict',
                           'event': 'dict',
                           'IBI': float,
                           'Rate': float,
                           }
    def __init__(self):
        self.commanders = []
        self.parasites = []

    def __repr__(self):
        temp = "<VarMothership:  commanders={c}, parasites={p}>"
        return temp.format(c=[c.name for c in self.commanders],
                           p=[p.name for p in self.parasites])

#    def __str__(self):
#        return '\n\n'.join([str(self.variables_as_table()),
#                            str(self.parasites_as_table())])

    def __iter__(self):
        for var in self.commanders + self.parasites:
            yield var.name

    def __getitem__(self, name):
        return self.asdict()[name]

    def _create_property_var(self, name):
        "creates the var; only call when you are sure it does not yet exist"
        if name in self.property_var_dtypes:
            dtype = self.property_var_dtypes[name]
        else:
            raise KeyError("No property var with name '%s'; see "
                           "variables.property_var_dtypes for list" % name)
        if name == 'subject':
            random = True
        else:
            random = False
        new_var = VarCommander(name, self, dtype, random=random)
        self.commanders.append(new_var)
        return new_var

    @property
    def all_vars(self):
        return self.commanders + self.parasites

    def as_table(self):
        table = fmtxt.Table('lll', title='Variables')
#        maxd = 30
        for p in self.commanders:
            table.cell(p.name)
            if p.dict_enabled:
                table.cell("'dict'")
                labels = dict(p.dictionary.items()[:2])
                labels = repr(labels)[:-1] + ', ...}'
                table.cell(labels)
#                labels = repr(p.dictionary)
#                if len(labels)
            else:
                table.cell(p.dtype.__name__)
                table.cell()
        return table

    def asdict(self):
        return {com.name: com for com in self.commanders + self.parasites}

    def getNewColony(self, copy=(), govern=True):
        """
        adds a new colony with a default value for all varCommanders

        copy :
            VarColony will copy the values from Colony

        """
        newColony = VarColony(self, items=copy)
#        if copy is None:
#            for v in self.commanders:
#                newColony[v] = v.default_value
#        else:
#            assert type(copy) == VarColony
#            for k, v in copy.iteritems():
#                newColony[k] = v
        return newColony

    def get(self, name):
        for var in self.commanders + self.parasites:
            if var.name == name:
                return var
        if name in self.property_var_dtypes:
            return self._create_property_var(name)
        raise KeyError("No Variable called '%s'" % var)

    def get_var_with_name(self, name):  # , dtype='dict', **kwargs):
        return self.get(name)

    def get_property_var(self, name):
        self.get(name)

    def new(self, name, dtype, random=False):
        "Create a new VarCommander"
        for var in self.commanders + self.parasites:
            if var.name == name:
                raise ValueError("variable with name '%s' already exists")
        if name in self.property_var_dtypes:
            return self._create_property_var(name)
        else:
            newCommander = VarCommander(name, self, dtype, random=random)
            self.commanders.append(newCommander)
            return newCommander

    def new_parasite(self, hosts, name, dtype, mapping):
        newParasite = Parasite(hosts, name, dtype, mapping)
        self.parasites.append(newParasite)
        return newParasite

    def ratings_from_table(self, table, hosts, dtype=float):
        """
        table: - the first row contains variable names
               - the first len(hosts) columns correspond to hosts
               - all remaining columns are added to the experiment as
                 VarParasites

                 ONLY NUMERICAL VARIABLES
        """
        if isvar(hosts):
            hosts = [hosts]
        n_start = len(hosts)
        n_end = len(table[0])
        if not hasattr(self, 'ratings'):
            self.ratings = []
        for i in range(n_start, n_end):
            name = table[0][i]
            mapping = [line[:n_start] + [line[i]] for line in table[1:]]
            p = self.new_parasite(hosts, name, dtype=dtype, mapping=mapping)
            self.ratings.append(p)

    def keys(self):
        return [var.name for var in self.commanders + self.parasites]

    def parasites_as_table(self):
        table = fmtxt.Table('lll', title='Parasites')
#        name_temp = '%s (%s)'
        source_temp = '<- [%s]'
        for p in self.parasites:
            table.cell(p.name)
            if p.dict_enabled:
                table.cell("'dict'")
            else:
                table.cell(p.dtype.__name__)
            table.cell(source_temp % ', '.join([h.name for h in p.hosts]))
        return table

    @property
    def var_names(self):
        "list of all names, including property vars"
        names = []
        for var in self.all_vars:
            names.append(var.name)
        names += (self.property_var_dtypes.keys())
        return names

    @property
    def varlist(self):
        slen = 20
        out = ' ' * slen
        for v in self.commanders:
            out += v.name.rjust(slen)[:slen]
#        for i,c in enumerate( self.colonies ):
#            out +=('\n%s'%(i+1)).ljust(slen)[:slen]
#            for v in self.commanders:
#                out+=str(c[v]).rjust(slen)[:slen]
        print out

    @property
    def varstructure(self):
        out = ''
        for v in self.commanders:
            out += '\n' + v.name + '\n'
            out += '    ' + str(v.uniqueValues)
            if v.dict_enabled:
                out += '\n    ' + str(v.dictionary)
        print out




class _Colony(dict):
    "Base Class (not usable by itself)"
    def __init__(self, items):
        if np.iterable(items):
            dict.__init__(self, items)
        else:
            dict.__init__(self)
    def __repr__(self):
        items = []
        for var, val in self.iteritems():
            if isvar(var):
                name = var.name
                if var.dict_enabled:
                    val = var.label(val)
            else:
                name = var
            items.append('%s=%r' % (name, val))
        return "{n}({d})".format(n=self.__class__.__name__, d=', '.join(items))
    def __str__(self):
        kvpairs = [': '.join([k, v]) for k, v in self.__str_items__()]
        return '\n\t'.join([self.__class__.__name__] + kvpairs)
    def __str_items__(self):
        for k, v in self.iteritems():
            yield k.__repr__(), str(v)
    """

    Pickling support
    ----------------
    ???
    """
    def __getstate__(self):
#        logging.debug("Colony GetState")
        if hasattr(self, 'mothership'):
            mothership = self.mothership
        else:
            mothership = None
        items = self.items()
        state = (mothership, items)
        return state
    def __setstate__(self, state):
#        logging.debug("Colony SetState")
        mothership, items = state
        if mothership is not None:
            setattr(self, 'mothership', mothership)
        for k, v in items:
            dict.__setitem__(self, k, v)
    def __reduce_initargs__(self):
        return ()
    def __reduce__(self):
        call = self.__class__
        init = self.__reduce_initargs__()
        state = self.__getstate__()
        return call, init, state


class VarColony(_Colony):  # contains all the values for a particular instance
    def __init__(self, varMothership, items=()):
        """
        Do not create directly. Called from varMothership

        VarColony needs link to mothership
         - access Parasite values when Parasite is suplied as name-string

        why does mothership need link to Colonies?
         -

        """
        _Colony.__init__(self, items)
        self.mothership = varMothership

    def __reduce_initargs__(self):
        return (self.mothership,)
    def __str_items__(self):
        for k, v in self.iteritems():
            yield k.name, k.label(v)
    def kill(self):
        pass
#        self.mothership.colonies.remove(self)
    # # accessing data
    def __get_var__(self, var):
        if isinstance(var, basestring):
            var = self.mothership.get(var)
        assert isvar(var)
        return var
    def __getitem__(self, var):
        """returns value for; use
        COLONY.num(var) to get numerical representation
        COLONY.repr(var) to get numerical representation

        """
        # logging.debug(" getitem: %s / %s /%s"%(self.variables, self, var))
        # TODO: enforce type; return str for dict var
        return self.num(var)
    def num(self, var):
        var = self.__get_var__(var)

        if type(var) == Parasite:
            return var[self]
        elif var in self:
            return dict.__getitem__(self, var)
        else:
            return var.default_value
    def __setitem__(self, var, value):
        var = self.__get_var__(var)
#        value = var._registerValue_forColony_(value, self)
        value = var._register_value_(value)
        dict.__setitem__(self, var, value)
    def copy(self):
        "return a copy of the VarColony"
        out = VarColony(self.mothership)
        out.update(self)
        return out
    def free(self):
        out = FreeColony(self)
        for par in self.mothership.parasites:
            out[par] = par[self]
        return out
### LEGACY ###
    def asdict(self):  # , parasites=True):#, labels=True):
        out = dict([(var, val) for var, val in self.iteritems()])
        for par in self.mothership.parasites:
            out[par] = par[self]
        return out
    def __iter__(self):
        for k, v in self.iteritems():
            yield k, v
        for par in self.mothership.parasites:
            yield par, par[self]


class FreeColony(_Colony):
    """
    Difference to VarColony:
    - has no link to mothership
    - Keys are strings, not variables
    - nominal var values are strings

    """
    def __init__(self, source=None):
        _Colony.__init__(self)
        if hasattr(source, 'iteritems'):
            for var, val in source.iteritems():
                self[var] = val
    def __setitem__(self, name, value):
        if isvar(name):
            if name.dict_enabled and isinstance(value, int):
                value = name.label(value)
            name = name.name
        if isinstance(name, basestring):
            dict.__setitem__(self, name, value)
        else:
            raise KeyError("Keys can be Variables or Strings")
    def __getitem__(self, name):
        if isvar(name):
            name = name.name
        return dict.__getitem__(self, name)

# # MARK: ## In Development ##


class Parasite(VarCommander):
    """
    dictionary mapping host values to Parasite values

    mapping initialization:
    1) as kwargs, value->value
    2) kwarg mapping=empty
       then map_from_table(table) method (calls map() method)
       table should contain (as value or label)
       - n columns with source
       - last column with target
    3) using map(source, target) method


    usage:
        Parasite[colony]
        Parasite[value]
        Parasite[label]


    WIP
    ---
    self.mapping: maps values from self.hosts --> self
    """  # TODO: everything
    def __init__(self, hosts, name, dtype='dict', mapping={}):
        if type(hosts) not in (list, tuple):
            hosts = [hosts]
        VarCommander.__init__(self, name, hosts[0].mothership, dtype)
        self.hosts = hosts
        if len(mapping) > 0:
            self.map_from_table(mapping, clear=True)
        else:
            self.mapping = {}
        # self.update(dic, labels=host.dict_enabled)
#    def __call__(self, *x):
#        """
#        transform factors containing the hosts into the parasite
#
#        """
#        assert len(x) == len(self.hosts)
#        t = type(x)
#        if t in [anova.factor, anova.var]:
#            source = x
#            x = source.x
#            out =
    def __repr__(self):
        temp = '<{c}:  {n} <- {h}, {dt}>'
        if self.dict_enabled:
            dt = "\'dict\', labels=%s" % self.dictionary
        else:
            dt = self.dtype.__name__
        return temp.format(c=self.__class__.__name__, n=repr(self.name), dt=dt,
                           h=' & '.join(h.name for h in self.hosts))
    def __str__(self):
        return str(self.map_as_table())
    def _registerValue_forColony_(self, value, colony):
        raise KeyError("Parasite Variable can not be assigned!")
    def __getitem__(self, name):
        if isinstance(name, basestring) or np.isscalar(name):
            name = (name,)
        t = type(name)
        if t is tuple:
            index = tuple(host.value(i) for host, i in zip(self.hosts, name))
            value = self.value_from_sourceindex(index)
            if self.dict_enabled:
                value = self.dictionary[value]
            return value
        elif t in [VarColony, dict]:
            if t == VarColony:
                v = tuple([host[name] for host in self.hosts])
            elif t == dict:
                v = tuple([var.value(name[var]) for var in self.hosts])
            if v in self.mapping:
                return self.mapping[v]
            else:
                return np.nan
#                raise KeyError("{0} not in parasite '{1}' mapping".format(v, self.name))
        else:
            return VarCommander.__getitem__(self, name)
    def __setitem__(self, name, value):
        if type(name) is VarColony:
            raise KeyError("Cannot assign Parasite")
        else:
            VarCommander.__setitem__(self, name, value)
    def values_from_sourceindexes(self, indexes):
        return [self.value_from_sourceindex(index) for index in indexes]
    def value_from_sourceindex(self, index):
        if np.isscalar(index):
            index = (index,)
        if len(index) != len(self.hosts):
            raise KeyError("Index Length != Host Length")
        out = self.mapping[index]
        return out
    """

    Maintaining the mapping
    """
    def map_from_table(self, table, clear=True):
        """
        sets self.mapping according to table:
        - file path (string)
        """
        if clear:
            self.mapping = {}
        if isinstance(table, basestring):
            if os.path.isfile(table):
                table = loadtable(table)
            else:
                raise IOError("%s not a valid file path" % table)
        if type(table) == dict:
            for k, v in table.iteritems():
                self.map(k, v)
        else:
            for row in table:
                source = row[:-1]
                target = row[-1]
                self.map(source, target)
    def map(self, source, target):
        id_source = []
        if isinstance(source, basestring) or np.isscalar(source):
            source = (source,)
        assert len(source) == len(self.hosts)
        for var, id in zip(self.hosts, source):
            if isinstance(id, basestring):
                if var.dict_enabled:
                    id_source.append(var.code_for_label(id, add=True))
                else:
                    id_source.append(float(id))
            else:
                id_source.append(id)
        if isinstance(target, basestring):
            if self.dict_enabled:
                target = self.code_for_label(target, add=True)
            else:
                target = float(target)
        self.mapping[tuple(id_source)] = target
    def map_as_table(self):
        """
        return a table listing source and target values
        """
        table = fmtxt.Table('l' * (len(self.hosts) + 2))
        for host in self.hosts:
            table.cell(host.name)
        table.cell()
        table.cell(self.name)

        table.midrule()

        for source, target in self.mapping.iteritems():
            for host, i in zip(self.hosts, source):
                table.cell(host.label(i))
            table.cell('-->')
            table.cell(self.label(target))

        return table




    # ???
    def update(self, dic, labels='auto'):
        """
        kwargs:
            labels='auto'   specifies whether keys in the dictionary are
                            labels (or values) of the host variable (bool)
                            'auto' sets to host.dict_enabled
        """
        if labels == 'auto':
            labels = self.host.dict_enabled
        if labels:
            keys = dic.keys()
            translated_keys = self.host.val_for_label(keys)
            dic = dict([ (k, v) for k, v in zip(translated_keys, dic.values())])
        dict.update(self, dic)



''' # UNUSED STUFF

class VarTable(np.ndarray):
    """
    Data Container for EventData

    args
    ----
    varlist         list of VarCommanders
    input_array     values -- Dict vars do not support str assignemtn yet!

    kwargs
    ------
    shape=0
    dtype=None
    buffer=None
    offset=0
    strides=None
    order=None


    example:
    t = VarTable([time], [[1.3],[3.4],[5.6]])


    Implemented after NumPy User Guide 1.3.0 pp. 23
    """
    #def __init__(*args, **kwargs):
    #    print "init called", args, kwargs
    def __new__(subtype, varlist, input_array,
                shape=0, dtype=None, buffer=None, offset=0, strides=None,
                order=None):
        # assert input
        if input_array:
            assert np.shape(input_array)[-1] == len(varlist)
        #print varlist
        #shape = (len(varlist), length)
        if dtype is None:
            dtype = ([(str(i), v.dtype) for i,v in enumerate(varlist)])
        obj = np.asarray(input_array).view(subtype)
        #obj = np.ndarray.__new__(subtype, shape, dtype, buffer, offset, strides,
        #                         order)
        obj.varlist = varlist
        return obj
### DOESN'T WORK ###### DOESN'T WORK ###### DOESN'T WORK ###### DOESN'T WORK ###
    def __reduce__(self):
        logging.debug("REDUCE")
        ndstate = np.ndarray.__reduce__(self)
        # (pickle.dumps(self.varlist),)
        state = ndstate
        print state
        return state
    def __setstate__(self, state):
        ndstate = state
        np.ndarray.__setstate__(self, ndstate)
### DOESN'T WORK ###### DOESN'T WORK ###### DOESN'T WORK ###### DOESN'T WORK ###
    def __array_finalize__(self, obj):
        # reset the attribute from passed original object
        self.varlist = getattr(obj, 'varlist', None)
    ## Management
    def __iter__(self):
        #if len(self.varlist) == 1:
        for i in range(self.shape[0]):
            yield np.ndarray.__getitem__(self, i)
        #else:
        #    for i in range(self.shape[0]):
        #        yield np.ndarray.__getitem__
    def __getitem__(self, name):
        #print "__getitem__(%s)"%str(name)
        if name in self.varlist:
            index = self.varlist.index(name)
            if self.ndim is 1:
                out = np.ndarray.__getitem__(self, index)
            else:
                out = np.ndarray.__getitem__(self, (Ellipsis, index, None))
                out.varlist = [name]
        elif len(self.varlist) is 1:
            out = np.ndarray.__getitem__(self, (name, 0))
        else:
            out = np.ndarray.__getitem__(self, name)# (name, Ellipsis))
        if False and np.iterable(name):
            print "iter!"
            out_varlist = None
            n = name[-1]
            if n in self.varlist:
                index = self.varlist.index(n)
                name = name[-1] + (index,)
                out_varlist = [n]
            out = np.ndarray.__getitem__(self, name)
            out.varlist = [name]
        return out
    def range(self, var, min=None, max=None):
        indices = []
        for i in range(len(self)):
            x = self[var][i]
            if min != None and min > x:
                continue
            elif max != None and max < x:
                continue
            elif min <= x < max:
                indices.append(i)
        return self[indices]
    # Representation
    def __repr__(self):
        if self.ndim is 1:
            _len = 1
        else:
            _len = len(self)
        out = "<VarTable (l={l}): [{v}]>".format(l=_len,
                                v=', '.join([v.name for v in self.varlist]))
        #out += np.ndarray.__repr__(self)
        return out
    def __str__(self):
        if self.ndim == 1:
            table = tex.Table('l'*len(self.varlist))
            [table.cell(v.name) for v in self.varlist]
            [table.cell(var[v]) for var,v in zip(self.varlist, self)]
        else:
            table = tex.Table('r'+'l'*len(self.varlist))
            table.cell()
            [table.cell(v.name) for v in self.varlist]
            table.midrule()
            for i,values in enumerate(self):
                table.cell(i, digits=0)
                [table.cell(var[v]) for var,v in zip(self.varlist, values)]
        return str(table)
    def names(self):
        return [v.name for v in self.varlist]





###  MARK:  ###  UNUSED  ###


class Event(object):
    """start/length in s !!!

    if start/length were represented in sample points
    --> PROCESSORS must transform it, it will not be good for low freq samplig (e.g. FFT)
    """
    def __init__(self, segment, start, length=0, facecolor='g'):
        self.start = start
        self.length = length
        self.segment=segment
        self.facecolor = facecolor
    @property
    def end(self):
        return self.start+self.length
    def setEnd(self, value):
        self.length = value - self.start



'''
