"""
Parameters
==========

Module for managing parameters in a dynamic way:

 - parameters that change can affect datasets

TODO:
create submodule with a class that contains the parameter + wx gui
that is imported flexibly like ui


Host
----

A Host object manages a set of parameters (Param objects). Both Host and
Param objects use private attributes except for functions useful to the 
user. This simplifies use in with autocompletion.


.. DANGER::
    if a param class has a default value, e.g.::
    
        class Dict(Param):
            def __init__(self, default={}):
                ...

    default will be the same instance for each Dict instance



Param
-----
The Param baseclass manages a parameter value. It can set a default value 
and can invalide different dataset attributes when the value changes

__init__ kwargs:

default=0,
    default value

can_be_None = False
    Apart from std value, the Param can be None

del_segs = False
    delete all segments when parameter changes

del_cache = True
    delete the cache (init with new shape)

del_props = True
    delete properties

del_children = False
    deletes all children; issues warning


Subclassing Param
^^^^^^^^^^^^^^^^^

Param subclasses should implement in the following steps:

#.    __init__ setting additional parameters and calling Param.__init__
#.    set providing help for parameters and calling set(value)
      with a single value
#.    _eval(value) set(value) calls 'value=self._eval(value)'. _eval should
      raise an Error id the value is invalid and can modify the value (e.g.
      int --> float)
      
can also implement:

3.    get() return the value in case it needs to be transformed (e.g. a 
      Param managing a folder path can return a list of file names; or a
      Window parameter can keep as its value window type and length, but
      return the actual window through get() 
4.    __value_repr__ called by __repr__ (default is value.__repr__())
5.    __subclass_repr__ can provide addition information for the param


some more things:

 - 
 - instance-specific help



"""

from __future__ import division

import os
import cPickle as pickle
from copy import deepcopy

import numpy as np

from eelbrain.utils._basic_ops_ import test_attr_name
from eelbrain import ui, fmtxt
from vars import isvar, isaddress, asaddress




def _slice2tuple_(s):
    start = s.start
    if start is None:
        start = 0
    stop = s.stop
    if stop is None:
        stop = -1
    return (start, stop)



class Host(object):
    """
    hosts a list of parameters, usually accessible as::
    
        >>> dataset.p
    
    for help to individual parameters use::
    
        >>> datatset.p.HELP()
    
    there are two equivalent methods for setting parameters::
    
        >>> dataset.p.name.set(value)
        >>> dataset.p.name = value
    
    """
    def __init__(self, dataset):
        self._params = {} # name -> Param dictionary 
        self._itemlist = [] # ordered list containing Params & structuring elements
        self._dataset = dataset
    
    def __repr__(self):
        lines = []
        name_len = max(len(name) for name in self._params.keys())
        for param in self._itemlist:
            if isinstance(param, basestring):
                lines.append('')
                lines.append(('-<'+param+'>').ljust(80, '-'))
#                lines.append(' ')#*length)
#                lines.append('-'*80)
#                lines.append(param + ' |')
#                lines.append('-'*len(param) + '-+')
#                lines.append(param + ':')
            else:
                value = param.__repr__().split(os.linesep)
                name_repr = ' ' + param._name.ljust(name_len)
                lines.append(': '.join((name_repr, value[0])))
                if len(value) > 1:
                    for v_line in value[1:]:
                        lines.append(' '*(name_len+3) + v_line)
        return os.linesep.join(lines)
    
    def _compiled(self):
        """
        returns a dict with compiled values for self._params
        
        """
        out = {}
        for k, v in self._params.iteritems():
            out[k] = v.get()
        return out
    
    def _section_(self, name):
        self._itemlist.append(name)
    
    def _add_(self, name, parameter):
        if hasattr(self, name):
            raise ValueError("Name %s already taken."%name)
        else:
            object.__setattr__(self, name, parameter)
            self._params[name] = parameter
            self._itemlist.append(parameter)
            parameter._name = name
            parameter._p = self
    
    def __setattr__(self, name, attr):
        """
        internally: 
            set private attributes (name has to start with '_')
        
        While building dataset.p:
            add parameter (name has to be a Param with a name that has not 
            previously been added)
        
        For user interaction: 
            set a parameter value (the Host needs to have a Param at name)

        """
        if name[0] == '_':
            object.__setattr__(self, name, attr)
        else:
            if hasattr(self, name):
                p = getattr(self, name)
                p.set(attr)
            elif isinstance(attr, Param):
                # not ideal, but have to make sure only real parameters are added
                self._add_(name, attr)
            else:
                raise KeyError("No parameter named %r" % attr)
    
    def _delete(self):
        "remove circular links to self from parameters "
        for p in self._params.values():
            del p._p
    #def _add_param(self, p):
    #    " used by params to add themselves"
    #    setattr(self, p.name, p)
    #    self._params[p.name] = p
        
    def __len__(self):
        return len(self._params)
    
    def HELP(self):
        """
        prints help for the dataset operation and the meaning of all all the
        parameters
        
        """
        lines = []
        name_len = max(len(name) for name in self._params.keys())
        desc_len = 80 - name_len - 2
        for param in self._itemlist:
            if isinstance(param, basestring):
                length = len(param)
                lines.append(' '*length)
                lines.append(param)
                lines.append('-'*length)
            else:
                desc = param._desc
                if desc is None:
                    value = ['']
                else:
                    value = []
                    line_r = ''
                    for word in desc.split():
                        if len(line_r) + len(word) < desc_len:
                            line_r += ' '+word
                        else:
                            value.append(line_r)
                            line_r = ''
                    if len(line_r) > 0:
                        value.append(line_r)

                name_repr = param._name.ljust(name_len)
                lines.append(': '.join((name_repr, value[0])))
                if len(value) > 1:
                    for v_line in value[1:]:
                        lines.append(' '*(name_len+2) + v_line)
        
        print os.linesep.join(lines)
    def __str__old(self):
        txt = "{n} = {v} ({desc}{dt})"
        strings = []
        for name, param in self._params.iteritems():
            desc = param.desc
            if desc:
                d = desc + ', '
            else:
                d = ''
            strings.append(txt.format(n=name, v=param.get(), dt=param.dtype, 
                                      desc=d))
        return '\n'.join(strings)


class Param(object):
    """
    
    stores its value in self._value
    default is what is assigned to self._value on init and reset

    """
    def __init__(self, default=0,
                 desc = None, # description of the parameter
                 dtype = None, # force the dtype of the value
                 can_be_None = False, # Apart from dtype, the Param can be None
                                    # (only applies with if dtype != None)
                 del_segs = False, # delete all segments when parameter changes
                 del_cache = True, # delete the cache (init with new shape)
                 del_props = True, # delete properties
                 del_children = False, # deletes all children; issues warning
#                 notify=[], # function or list of functions that are called
#                            # when the value changes (with self as argument)
#                            # !!! not picklable
                 ):
        
        # save all args
        self._default = default
        # WARNING: if default is __init__ arg, each instance gets the same
        # instance
        self._can_be_None = can_be_None
        self._desc = desc
        self._dtype = dtype
        
        self._del_segs = del_segs
        self._del_cache = del_cache
        self._del_props = del_props
        self._del_children = del_children
#        if not np.iterable(notify):
#            notify = [notify]
#        self._notify = []#notify
        # set default
        self.reset()
    def reset(self):
        if hasattr(self, '_value'):
            if self._value == self._default:
                pass
            else:
                self._value = self._default
                self._changed()
        else:
            self._value = self._default
    def _changed(self, ids=None, _ids=None):
        if self._del_props:
            self._p._dataset.delete_properties()
        if self._del_cache:
            self._p._dataset.delete_cache(ids=ids, _ids=_ids)
        if self._del_segs:
            self._p._dataset.delete_segments()
        if self._del_children:
            for c in self._p._dataset.children:
                c.delete()
#        for n in self._notify:
#            n(self)
### SUBCLASS ###  SUBCLASS ###  SUBCLASS ###  SUBCLASS ###  SUBCLASS ###  SUBCLA
    def __repr__(self):
        return repr(self._value)
    
    def __call__(self, value):
        msg = "Param setting using __call__ is deprecated. Use .set() method"
        print msg
        self.set(value)
    
    def set(self, value):
        """
        stores whatever value is assigned.
        
        """
        if self._dtype:
            if value is None:
                if self._can_be_None:
                    value = None
                else:
                    raise ValueError("None is not allowed")
            else:
                if not isinstance(value, self._dtype):
                    value = self._dtype(value)
        
        # implement change
        if self._different_(self._value, value):
            self._value = value
            self._changed()
    
    def _different_(self, v1, v2):
        """
        check whether v1 and v2 are different values for the Parameter. The 
        baseclass method works for a range of cases, but fails when
         - values have nonstandard __eq__ output
        
        """
        if type(v1) is not type(v2):
            return True
        eq = (v1 != v2)
        if np.iterable(eq):
            return np.any(eq)
        else:
            return eq 
    def get(self):
        """
        returns the final value
        
        """
        if (self._value is None) and not (self._can_be_None):
            name = "<e>.%s.p.%s"%(self._p._dataset.name, self._name)
            raise ValueError("Variable %s cannot be None" % name)
        else:
            return self._value            


class PerParentSegment(Param):
    """
    Param that stores a different value for each segment. Use::
    
        >>> dataset.p.param[id] = value
    
    to assign different values for different segments, and use on of::
    
        >>> dataset.p.param = value
        >>> dataset.p.param(value)
    
    to assign the same value for all segments.
    
    The PerParentSegment parameter has special methods for exporting values as 
    a file and reimport them: export_pickled and import_pickled.
    
    """
    def reset(self):
        "set the value for all segments to the default value"
        self._value = {}
        self._refresh_keys_()
    def _refresh_keys_(self):
        # get ids present in the input dataset
        if hasattr(self, '_p'):
            ids = self._current_ids = [s._id for s in self._p._dataset.parent.segments]
        else:
            ids = []
        
        # remove from self._value ids no longer present
        for id in self._value.keys():
            if (id not in ids) and (self._value[id] is self._default):
                del self._value[id]
        
        # add new ids to self._value
        for id in ids:
            if id not in self._value:
                self._value[id] = self._default
    def __repr__(self):
        # test if all values are the same
        values = self._value.values()
        if len(values) > 1:
            v = values[0]
            if all(v==v2 for v2 in values[1:]):
                return "%r (all segments)" % v
        
        present = []
        past = []
        temp = "%s%s: %r"
        
        self._refresh_keys_()
        for k in sorted(self._value.keys()):
            if k in self._current_ids:
                present.append(temp % ('  ', k, self._value[k]))
            else:
                past.append(temp % ('* ', k, self._value[k]))
        
        if len(past) > 0:
            past += ['', '(* segments that are not in the dataset any longer)']
        
        return '\n'.join(present + past)
    def __setitem__(self, id, value):
        if id not in self._value:
            self._refresh_keys_()
            if id not in self._value:
                raise KeyError('%r is not a valid segment id' % id)
        
        if self._value[id] == value:
            pass
        else:
            self._value[id] = value
            self._changed(_ids=[id])
    def __getitem__(self, id):
        if id not in self._value:
            self._refresh_keys_()
            if id not in self._value:
                raise KeyError('%r is not a valid segment id' % id)
        
        return self._value[id]
    def set(self, value):
        "sets the value for all segments to value"
        for k in self._value.keys():
            self._value[k] = value
    def get_dict(self, decimals = 'all'):
        """
        Return the values as dictionary (can be used to update() the parameter)
        
        :arg int decimals: number of decimals to return in values (if the 
            dictionary is used in text form in a script it can be convenient 
            to reduce the size)
        
        """
        if decimals == 'all':
            out = deepcopy(self._value)
        elif decimals in [0, None]:
            return dict((k, int(v)) for k,v in self._value.iteritems())
        else:
            out = {}
            for k, v in self._value.iteritems():
                out[k] = round(v, decimals)
        return out
    def update(self, dictionary):
        for key, value in dictionary.iteritems():
            self.__setitem__(key, value)
    def export_pickled(self, path=None):
        "Exports the parameter's values as a pickled dictionary file."
        if path is None:
            path = ui.ask_saveas(title="Export Parameter Values", 
                                 message="Select a filename to pickle the parameter values:", 
                                 ext=[('pickled', "pickled Python object")])
        if path:
            with open(path, 'w') as f:
                pickle.dump(self._value, f)
    def import_pickled(self, path=None):
        "Imports the values from a pickled python object."
        if path is None:
            path = ui.ask_file(title="Import Parameter Values", 
                               message="Select a file that contains the pickle parameter values:",
                               ext=[('pickled', "pickled Python object")])
        
        with open(path, 'r') as f:
            self._value.update(pickle.load(f))
            
            




class Time(Param):
    """
    holds time value, can be specified in sample points or seconds;
    
    If it is initialized with canbevar it also accepts VarCommanders so that
    a different time can be specified for each segment segment
     
    """
    def __init__(self, can_be_var=False, can_be_None=False, **kwargs):
        self._can_be_var = can_be_var
        Param.__init__(self, can_be_None=can_be_None, **kwargs)
    def __repr__(self):
        s = self.get()
        if isvar(s):
            return s.name
        elif s is None:
            return "None"
        elif np.isreal(s):
            return "%f sec."%s
        elif np.iscomplex(s):
            return "%i samples"%(s.imag)
    def set(self, time):
        """
        Use real values to indicate time in seconds, and complex values to 
        indicate sample points (e.g. at 200 Hz, 100i == .5)
        
        Note
        ----
        the value is stored in sample points and does not adapt when the 
        samlingrate of the dataset changes; the value has to be changed 
        manually.
        
        """
        if isvar(time):
            if not self._can_be_var:
                raise ValueError("Parameter does not accept variable")
        else:
            assert np.isscalar(time)
        Param.set(self, time)
    def in_seconds(self, samplingrate, varcolony=None):
        "returns None if Time is a variable and segment is None"
        v = self.in_samples(samplingrate, varcolony=varcolony)
        if v is None:
            return v
        else:
            return v / samplingrate
    def in_samples(self, samplingrate, varcolony=None):
        "returns None if Time is a variable and segment is None"
        v = Param.get(self)
        if isvar(v):
            if varcolony is None:
                return None
            else:
                v = varcolony[v]
        
        if np.isreal(v):
            return int(v * samplingrate)
        elif np.iscomplex(v):
            out = v.imag
            return out


class FileList(Param):
    """
    chiefly for importers
    if ext is specified, only files with this extension are imported
    
    """
    def __init__(self, ext='',
                 default=None, 
                 del_segs=True, **kwargs):
        default = {'f':[], 'd':[]}
        Param.__init__(self, default=default, del_segs=del_segs, **kwargs)
        self._ext = ext
    
    def __repr__(self):
        dirs  = ['- %r' % d for d in self._value['d']]
        files = ['- %r' % f for f in self._value['f']]
        
        out = ["Extension: %r  (use set_ext() to change)"%self._ext]
        
        if dirs:
            if len(dirs) == 1:
                out.append("1 directory:")
            else:
                out.append("%s directories:" % len(dirs))
            out.extend(dirs)
        
        if files:
            if len(files) == 1:
                out.apend("1 separate file:")
            else:
                out.append("%s separate files:" % len(files))
            out.exptend(files)
        
        return os.linesep.join(out)
    
#    def __str__(self):
#        files = self._value['f']
#        dirs = self._value['d']
#        if len(dirs) + len(files) == 0:
#            return "Empty"
#        else:
#            out = []
#            i = 0
#            if len(dirs) > 0:
#                out.append('Directories:')
#                for f in dirs:
#                    num = str(i).rjust(4)
#                    out.append('  %s. %s'%(num, f))
#                    i += 1
#            if len(files) > 0:
#                out.append('Files:')
#                for f in files:
#                    num = str(i).rjust(4)
#                    out.append('  %s. %s'%(num, f))
#                    i += 1
#            return os.linesep.join(out)
    
    def set(self, path=None):
        """
        path can be file or directory
        if path==None: a dialog will be displayed to select a directory
        
        (use .set_ext('ext') method to modify file extension)
        
        """
        if path is None:
            path = ui.ask_dir()
            if not path:
                return
        
        if isinstance(path, basestring):
            if os.path.isdir(path):
                self._value['d'].append(path)
            elif os.path.exists(path):
                self._value['f'].append(path)
            else:
                raise IOError("%s does not exists"%path)
            self._changed()
        else:
            raise ValueError("must be string")
    
    def set_ext(self, ext):
        assert isinstance(ext, basestring) or ext is None
        if ext == self._ext:
            pass
        else:
            if isinstance(ext, basestring) and ext[0] == '.':
                ext = ext[1:]
            self._ext = ext
            self._changed()
    
    def pop(self, i):
        "pops an element (i=number according to self.__str__)"
        dir_n = len(self._value['d']) 
        if i < dir_n:
            return self._value['d'].pop(i)
        else:
            file_n = len(self._value['f'])
            i -= dir_n
            if i < file_n:
                return self._value['f'].pop(i)
            else:
                raise ValueError("i=%s greater than len(self)"%i)
    
    def rm(self, i):
        "removes an element (i=number according to self.__str__)"
        self.pop(i)
    
    def get_as_pathlist(self):
        out = []
        for path in self._value['d']:
            if self._ext:
                out.append(path + '/*.' + self._ext)
            else:
                out.append(path + '/*')
        for path in self._value['f']:
            out.append(path)
        return out
    
    def get_as_filelist(self):
        out = []
        for path in self._value['d']:
            for file in os.listdir(path):
                if (file[0] != '.'):
                    if (not self._ext) or file.endswith('.'+self._ext):
                        out.append(os.path.join(path, file))
        out += self._value['f']
        return out



class Dict(Param):
    """
    a dictionary
    
    key_dtype and value_dtype can be used to specify a dtype that is enforced
    for keys or values, using:
    
    if not isinstance(x, dtype): 
        try:
            dtype(x)
        except:
            raise 
    
    """
    def __init__(self, key_dtype=None, value_dtype=None, **kwargs):
        Param.__init__(self, default=None, **kwargs)
        self._value_dtype = value_dtype
        self._key_dtype = key_dtype
    def reset(self):
        "Clear all values."
        # needs its own reset function because 
        if hasattr(self, '_value'):
            if len(self._value) == 0:
                return
            else:
                self._value = dict()
                self._changed()
        else:
            self._value = dict()
    def __repr__(self):
        if len(self._value) == 0:
            out = ['empty']
        else:
            str_pairs = []
            for k, v in self._value.iteritems():
                str_pairs.append((repr(k), repr(v)))
            k_len = max(len(k) for k,v in str_pairs)
            out = [' : '.join((k.ljust(k_len), v)) for k,v in str_pairs]
        return os.linesep.join(out)
    def __setitem__(self, name, value):
        # avoid invalidating cache if nothing changes
        if isinstance(name, slice):
            name = _slice2tuple_(name)
        if name in self._value:
            if self._value[name] == value:
                return
        # data type verification
        if self._value_dtype:
            if not isinstance(value, self._value_dtype):
                value = self._value_dtype(value)
        self._value[name] = value
        self._changed()
    def __getitem__(self, name):
        if isinstance(name, slice):
            name = _slice2tuple_(name)
        return self._value[name]
    def __delitem__(self, name):
        if isinstance(name, slice):
            name = _slice2tuple_(name)
        del self._value[name]
    def set(self, key, value):
        """
        set one key-value pair. Equivalent to:
        >>> p[key] = value
        
        to remove an item:
        >>> del p[key]
        """
        self.__setitem__(key, value)


class VarsFromNames(Dict):
    """
    usage:
    
    param[index] = var
    
    index: can be int or slice (e.g. param[1:5] = subject)
    var: can be string or experiment variable. Strings are converted to 
         experiment variables.
    """
    def __init__(self, namesource=None, **kwargs):
        """
        kwargs:
        filelist: FileList param that provides list of filenames to be used as 
        example filenames (needs function .get_as_filelist)
        
        """
        # FIXME: default handling
        Dict.__init__(self, **kwargs)
        self._sample_names = []
        self._namesource = namesource
    def set_sample_names(self, namelist):
        "set the names that are used for sample extraction in __repr__"
        assert all(isinstance(name, basestring) for name in namelist)
        self._sample_names = namelist
    def __setitem__(self, name, value):
        isslice = isinstance(name, slice)
        isint = isinstance(name, int)
        if isslice:
            name = _slice2tuple_(name)
        if isint or isslice:
            if isvar(value):
                var = value
            elif isinstance(value, basestring):
                try:
                    var = self._p._dataset.experiment.variables.get(value)
                except:
#                    create = ui.ask(title="Create Variable",
#                                    message="No variable named '%s', create variable?"%value,
#                                    cancel=False)
                    create = True
                    if create is True:
                        var = self._p._dataset.experiment.variables.new(value, dtype='dict')
                    else:
                        ui.msg("Aborted")
                        return
            self._value[name] = var
            self._changed()
        else:
            raise ValueError("key needs to be int or slice")
    def split(self, filename):
        """
        splits filename into the variables
        
        """
        out = {}
        for index, name in self._value.iteritems():
            if isinstance(index, tuple):
                out[name] = filename[slice(*index)]
            elif isinstance(index, int):
                out[name] = filename[index]
            else:
                raise ValueError("Invalid index: %s"%type(index))
        return out
    def get_vars(self):
        "returns list of all variables (converts strings to variables)"
        vars = self._value.values()
        return vars
    def names_to_vars(self):
        """
        converts all the variables provided as strings into real
        variables
        """
        for k, v in self._value.items():
            if not isvar(v):
                var = self._p._dataset.experiment.get_var_with_name(v)
                self._value[k] = var
    def __repr__(self):
        if len(self._value) == 0:
            out = ['(No Variables Extracted)']
        else:
            out = ['index   name', '-'*18]
            for index, name in self._value.iteritems():
                if hasattr(index, 'start'):
                    line = '-'.join((str(index.start), str(index.stop)))
                else:
                    line = str(index)
                line = line.ljust(8) + name.name
                out.append(line)
            out.append('')

        # get sample names        
        if self._namesource != None:
            filelist = self._namesource.get_as_filelist()
            sample_names = [os.path.basename(path) for path in filelist]
        else:
            sample_names = None
            
        # create text output
        if sample_names:
            keys = self._value.keys()
            
            table = fmtxt.Table('l'*(1 + len(self._value)))
            max_n = 5
            fn_len = max(len(name) for name in sample_names[:max_n]) + 2
            count = ('0123456789' * (1 + fn_len//10))[:fn_len]
            table.cell(count)
            for k in keys:
                table.cell(self._value[k].name)
            
            # fn/variable list 
            vars = self.get_vars()  
            for fname in sample_names[:max_n]:
                table.cell(fname)
                values = self.split(fname)
                for k in keys:
                    var = self._value[k]
                    table.cell(values[var])
            
            out.append(str(table))
            out.append("...")
        else:
            out.append('(No example filenames available)')
        
        text = os.linesep.join(out)
        return text




class DataChannels(Dict):
    """
    Parameter for specifying channels to extract in an UTS-importer
    
    stores a dictionary with a mapping  
    {channel-nr  -->  extraction parameters}
    
    set using::
    
        >>> ds.p.channels[1] = "skin_conductance", 'uts'
    
    remove (don't extract channel) using::
    
        >>> del ds.p.channels[1]
    
    
    extraction parameters can be:
    
    * for data channel:    'name', 'uts'
    * for event channel:   ('name', 'evt', [threshold, [targets]] )
        
    The first parameter is a name for the extracted data. It should be a string
    appropriate for an attribute of the parent Experiment instance. The second 
    parameter specifies the data type ('uts' is the default and is optional). 
    The third and fourth parameter specify options depending on the 
    data type:
    
    
    event extraction: 
    
    By default, any change to a non-zero value is counted as an event onset,
    with the event duration lasting until the next value change. With noisy 
    event channels, the following parameters can help dissociate events:
    
    threshold: 
        (scalar) if threshold is provided, all data points > threshold 
        are assumed to be part of an event. Unless targets (valid event values)
        are provided, all events are assigned magnitude 1.
    
    targets: 
        (list of scalars) values of events to look for. If no threshold is
        provided, it is assumed that the threshold is at half the first target
        value.
    
    
    """
#    Topographic Data:
#    
#    * topographic (eeg):   ('name', 'topo', sensor_net)
#    
#    e.g.::
#    
#    >>> i.p.channels[:164] = "EEG", 'topo', sensors.from_xyz("/path/to.xyz")
#    
    
    def __init__(self, **kwargs):
        self._n = 0
        Dict.__init__(self, **kwargs)
    def __repr__(self):
        if self._n == 0:
            out = ['No Channels in Data (select files first)']
        elif len(self._value) == 0:
            out = ['No Channels selected (of %i)'%self._n]
        else:
            out = ['%i Channels:' % self._n]
            for k, v in self._value.iteritems():
                # key
                if isinstance(k, int):
                    k_repr = "[%s]" % k
                else:
                    if k[0] is None:
                        start, stop = k[1:]
                        if start == None:
                            start = ''
                        if stop == None:
                            stop = ''
                        k_repr = "[%s:%s]" % (start, stop)
                    else:
                        k_repr = '[%s]' % ', '.join(str(ki) for ki in k)
                
                # value
                name, kind, arg1, arg2 = v
                v_repr = '"%s"'%name
                if kind:
                    v_repr += ", '%s'"%kind
                if arg1:
                    v_repr += ", %r"%arg1
                if arg2:
                    v_repr += ", %r"%arg2
                
                out.append(' = '.join((k_repr, v_repr)))
        return os.linesep.join(out)
    def _set_n_channels(self, n):
        "set the number of channels available"
        if n != self._n:
            self._n = n
            self.reset()
    def _key_from_index(self, name):
        if isinstance(name, int):
            if 0 <= name < self._n:
                return name
            else:
                raise KeyError("Fewer than %i channels!"%name)
        
        elif isinstance(name, slice):
            assert name.step in [None, 1]
            
            if name.start == None:
                start = 0
            else:
                start = self._key_from_index(name.start)
            
            if name.stop == None:
                stop = self._n
            else:
                stop = self._key_from_index(name.stop)

            return None, start, stop
        
        elif isinstance(name, tuple):
            if all(0 <= v < self._n for v in name):
                return name
            else:
                raise KeyError("Data has only %i channels!"%self._n)
        
        else:
            raise KeyError("Invalid channel index: %s"%str(name))
    def _index_from_key(self, key):
        if isinstance(key, tuple) and (key[0] is None):
            return slice(key[1], key[2])
        else:
            return key
    def __setitem__(self, chan, value):
        chan = self._key_from_index(chan)
        
        # fill tuple
        if isinstance(value, basestring):
            value = (value, 'uts')
        while len(value) < 4:
            value = value + (None,)
        
        # check properties
        name, conversion, arg1, arg2 = value
        test_attr_name(name)
        assert conversion in ['uts', 'evt', 'topo']
        if conversion == 'evt':
            if arg1 is not None:
                assert np.isscalar(arg1)
            if arg2 is not None:
                assert np.iterable(arg2)
        elif conversion =='topo':
            assert isinstance(chan, tuple)
            if arg1 is not None:
                n_s = arg1.n
                start, stop = chan
                if start == None:
                    start = 0
                if stop == None:
                    stop = self._n
                
                n_chan = stop - start
                if n_chan != n_s:
                    raise ValueError("Number of Channels mismatch between "
                                     "Channel selection (%i) and sensor Net "
                                     "(%i)" % (n_chan, n_s))
        
        # set new value
        self._value[chan] = value
        self._changed()
    def __delitem__(self, key):
        key = self._key_from_index(key)
        if key in self._value:
            self._value.pop(key)
        else:
            raise KeyError
    def iterchannels(self):
        for key, settings in self._value.iteritems():
            index = self._index_from_key(key)
            yield index, settings



class VarDict(Dict):
    """
    Dict subclass that ascertains keys are variables
    
    """
    def __setitem__(self, var, value):
        if isinstance(var, basestring):
            var = self._p._dataset.experiment.variables.var_with_name(var)
        if not isvar(var):
            raise ValueError("Key needs to be variable")
        Dict.__setitem__(self, var, value)



class Choice(Param):
    """
    lets the user choose between one of the options
    
    _value is selection id
    """
    def __init__(self, options=['yes', 'no'], default=0, **kwargs):
        assert len(options) > default
        self._options = [unicode(ch) for ch in options]
        Param.__init__(self, default=default, **kwargs)
    def __repr__(self):
        i_len = (len(self._options)-1) // 10
        out = []
        for i, name in enumerate(self._options):
            if i == self._value:
                arrow = '->'
            else:
                arrow = '  '
            txt = str(i).ljust(i_len) + arrow + name
            out.append(txt)
        return os.linesep.join(out)
    def set(self, value):
        """
        int or str corresponding to the relevant option
        
        """
        if value in self._options:
            value = self._options.index(value)
        if value in range(len(self._options)):
            self._value = int(value)
            self._changed()
        else:
            raise ValueError("Invalid selection: %s"%value)
    def get_string(self):
        "returns selection as string"
        return unicode(self._options[self._value])
    def get_i(self):
        "returns selection id as int"
        return self._value


class Window(Param):
    """
    Used to select a window. Set with two parameters: window type and length. 
    Window type is specified as index:
        
    0: Hann
    1: Hamming
    2: Bartlett
    3: Blackman
    
    Length is in samples.
    
    """
    # Options are stored in self._windows
    _windows = (("Hann", np.hanning),
                ("Hamming", np.hamming),
                ("Bartlett", np.bartlett),
                ("Blackman", np.blackman))
    def __init__(self, default = (1, 500), #can_be_None=True,
                 normalize=True, **kwargs):
        self._normalize = normalize
        self._options = ["None"] + [name for name, f in self._windows]
        Param.__init__(self, default=default, **kwargs)
    def __repr__(self):
        selection, width = self._value
        out = []
        # window type
        i_len = (len(self._options)-1) // 10
        for i, name in enumerate(self._options):
            if i == selection:
                arrow = '->'
            else:
                arrow = '  '
            txt = str(i).ljust(i_len) + arrow + name
            out.append(txt)
        # width
        out.append("Window Length: %s"%width)
        return os.linesep.join(out)
    def set(self, window, length=500):
        """
        window: 0 - None
                1 - Hann
                2 - Hamming
                3 - Bartlett
                4 - Blackman
        
        length: in samples
                 
        """
        if isinstance(window, basestring):
            window = self._options.index(window)
        elif window is None:
            window = 0
        self._value = (int(window), int(length))
        self._changed()
    def get(self):
        if self._value[0] == 0:
            return None
        else:
            selection, length = self._value
            func = self._windows[selection-1][1]
            out = func(length)
            out /= np.sum(out)
            return out


class Color(Param):
    def __init__(self, default=(0.078125, 0.578125, 0.99609375),
                 del_cache=False, **kwargs):
        Param.__init__(self, default=default, del_cache=del_cache, **kwargs)
    def set(self, value=None):
        """
        call without a value to open color selection dialog, or call with
        color description (currently only rgb-tuples work)

        """
        if not value:
            value = ui.ask_color(default=self.get())
        assert len(value) == 3, "need len=3 rgb tuple"
        Param.set(self, value)


class VarCommander(Param):
    def __init__(self, default=None, can_be_None=True, **kwargs):
        if not can_be_None:
            assert default is not None
        Param.__init__(self, default=default, can_be_None=can_be_None, **kwargs)
    def set(self, var):
        """
        var can be a Variable or a string (in which case the variable is 
        retrieved from the experiment
        
        """
        if isinstance(var, basestring):
            e = self._p._dataset._experiment
            var = e.get_var_with_name(var)
        Param.set(self, var)


class Address(Param):
    def __init__(self, default=None, can_be_None=True, **kwargs):
        Param.__init__(self, default=default, can_be_None=can_be_None, **kwargs)
    def _different_(self, v1, v2):
        return v1 is not v2
    def set(self, address):
        """
        This parameter stores a biovars.Address object.
         
        """
        if isinstance(address, basestring):
            address = self._p._dataset.experiment.variables.get(address)
        
        if address is None or isaddress(address):
            Param.set(self, address)
        else:
            Param.set(self, asaddress(address))


class Variable(Param):
    def __init__(self, default=None, can_be_None=True, 
                 vartype=[int, float],
                 **kwargs):
        if not np.iterable(vartype):
            vartype = [vartype]
        self._vartype = vartype
        Param.__init__(self, default=default, can_be_None=can_be_None, **kwargs)
    def _different_(self, v1, v2):
        return v1 is not v2
    def set(self, variable):
        """
        This parameter stores a Variable object. If a string is submitted, the
        parameter tries to cerate a new variable with that name.
         
        """
        if isinstance(variable, basestring):
            v = self._p._dataset.experiment.variables
            try:
                variable = v.var_with_name(variable)
            except:
                variable = v.new(variable, self._vartype[0])
            
        if variable.dict_enabled:
            if 'dict' not in self._vartype:
                raise "Wrong Variable Type (need: %s)"%str(self._vartype)
        else:
            if variable.dtype not in self._vartype:
                raise "Wrong Variable Type (need: %s)"%str(self._vartype)
        
        Param.set(self, variable)




