import logging
from copy import deepcopy

import numpy as np
from matplotlib import mlab

from ..utils.basic import intervals

from vars import isvar
from datasets_base import Derived_Event_Dataset
import param

__hide__ = ['logging', 'deepcopy', 'np', 'mlab', 'intervals', 'isvar',
            'Derived_Event_Dataset', 'param']


# MARK: functions

def _evts_from_data(data, samplingrate=None, t0=0, forcelength=None,
                    threshold=None, targets=None):
    """
    Extracts events from one-dimensional data array (data is supposed to be
    1-dimension, np.ravel is applied)

    returns array of shape (t, 3) with the 3 columns being: [time, duration,
    magnitude]

    kwargs
    ------
    threshold: (scalar) if threshold is provided, all data points > threshold
        are assumed to be part of an event. Unless targets (valid event values)
        are provided, all events are assigned magnitude 1.

    targets: (list of scalars) values of events to look for. If no threshold is
        provided, it is assumed that the threshold is at half the first target
        value.


    """
    # logging.debug(" _evts_from_data, data.shape: {0}".format(data.shape))
    data = np.ravel(data)

    # prepare threshold
    if targets is None:
        if threshold is not None:
            data = (data > threshold).astype(float)
    else:
        targets = sorted(targets)
        if threshold is None:
            threshold = targets[0] / 2
        thresholds = [threshold] + [(x1 + x2) / 2 for x1, x2 in intervals(targets)]

        # restrict data to thresholds
        clean_data = np.zeros(len(data))
        for threshold, target in zip(thresholds, targets):
            index = (data > threshold)
            clean_data[index] = target

        # prune short events
        minstep = 5
        breaks = mlab.find(np.diff(clean_data)) + 1
        while min(np.diff(breaks)) < minstep:
            for t0, t1 in intervals(breaks):
                if t1 - t0 < minstep:
                    clean_data[t0:t1] = clean_data[t1]
            breaks = mlab.find(np.diff(clean_data)) + 1

        data = clean_data

    events = []
    indexes = mlab.find(np.diff(data)) + 1
    for start, end in intervals(indexes):
        if data[start] == 0:
            pass
        else:
            mag = data[start]
            events.append([start, end - start, mag])

#            noisy = data
#            data = np.zeros(len(noisy))
#            data = (data > threshold).astype(float)
#
#            threshold = np.array(threshold)
#            data = np.sum(data > threshold[:,None], axis=0)
#        # step
#            data_max = np.max(data)
#            steps = np.arange(step/2, data_max, step)
#            data = np.sum(data > steps[:,None], axis=0)
#            i = mlab.find(np.diff(data))+1
#            for start, end in intervals(i):
#                if end - start <= seplen: # sep
#                    if data[start] < data[end]:
#                        data[start:end] = 0
#                    else:
#                        data[start:end] = data[start]

#        indexes = mlab.find(np.diff(abs)) + 1 # finds the first index of each value change
#        for start, end in intervals(indexes):
#            if abs[start] == 0:
#                pass
#            else:
#                i = int((start + end) / 2)
#                mag =  data[i]
#                if thresholds is not None:
#                    iv = np.sum(mag > thresholds)
#                    mag = values[iv]
#                events.append([start, end - start, mag])
    events = np.array(events)
    if samplingrate:
        events[:, [0, 1]] /= float(samplingrate)
    if forcelength:
        events[:, 1] = forcelength
    if t0:
        events[:, 0] -= t0
    logging.debug(" _evts_from_data found %s events" % (events.shape[0]))  # find in raw signal:  len(where(diff(where(b[1]>.5)[0]) > 1)[0])
    return events


class Extract(Derived_Event_Dataset):
    """
    Extracts events from one-dimensional data array. Assumes that the value for
    no event is 0 or below a certain threshold, and that an increase to one of a
    number of fixed values indicates a certain event.


    Parameters:

    :arg scalar threshold: if threshold is provided, all data points > threshold
        are assumed to be part of an event. Unless targets (valid event values)
        are provided, all events are assigned magnitude 1.

    :arg targets: (list of scalars) values of events to look for. If no threshold is
        provided, it is assumed that the threshold is at half the first target
        value.

    """
    def _addparams_(self, p):
        desc = "(Scalar) All values exceeding the threshold are interpreted as events"
        p.threshold = param.Param(default=None, can_be_None=True, desc=desc)
        desc = ("(List of scalars) input values that are valid event codes. If "
              "None is provided, each occurring value is interpreted as a "
              "separate code")
        p.targets = param.Param(default=None, can_be_None=True, desc=desc)
        p.var = param.Variable(default='magnitude', can_be_None=False,
                               desc="Variable for event codes.")
    def _create_varlist_(self):
        time = self.experiment.variables.get('time')
        duration = self.experiment.variables.get('duration')
        magnitude = self.p.var.get()
        varlist = [time, duration, magnitude]
        return varlist
    def _derive_segment_data_(self, segment, preview=False):
        parent = segment._p_attr['source']
        data = parent.data[:, 0]
        samplingrate = parent['samplingrate']
        t0 = parent['t0']

        e_list = _evts_from_data(data, samplingrate, t0=t0,
                                 threshold=self.p.threshold.get(),
                                 targets=self.p.targets.get())
        return e_list



# MARK: Dataset Classes

class _Event_to_Event(Derived_Event_Dataset):
    """
    Baseclass for datasets with events as in- as well as output.

    """
    def _validate_input_properties_(self, properties):
        if properties['data_type'] != 'event':
            raise ValueError("Need input with data_type 'event'")
        else:
            return True
    def _create_varlist_(self):
        return self.parent.varlist[:] + self._new_vars()
    def _new_vars(self):
        return []
### SUBCLASS ###### SUBCLASS ###### SUBCLASS ###### SUBCLASS ###### SUBCLASS ###
    def _derive_segment_data_(self, segment, preview=False):
        """
        !!! make sure to set segment['duration'] if it is != source segment
        """
        raise NotImplementedError()
#        segment['duration'] = 325
#        return e_list


class Conflate(_Event_to_Event):
    """
    takes a number of events and unifies them into one event

    d = Conflate(parent)

    d.p.add[var] = i, i_var       var: variable in the new event
                                  i: index of the source event
                                  i_var: original variable

    d.p.add[var] = i, i_var, map      var: variable in the new event
                                      i: index of the source event
                                      i_var: original variable
                                      map: {i_var value : var value, ...}
    """
    def _addparams_(self, p):
        p.n = param.Param(default=1, desc="Number of events to conflate")
        p.base_i = param.Param(default=0, desc="Index of the basis event")
        p.add = param.Dict(desc="Index of the basis event")
    def _new_vars(self):
        out = []
        mod = self.p.add.get()
        for var in mod:
            if var not in self.parent.varlist:
                out.append(var)
        return out
    def _derive_segment_data_(self, segment):
        n = self.compiled['n']
        base_i = self.compiled['base_i']
        add = self.compiled['add']
        add_len = len(self.varlist) - len(self.parent.varlist)

        source = segment._p_attr['source']
        data = source.data

        out = deepcopy(data[base_i::n])
        out = np.hstack([out, np.zeros((len(out), add_len))])
        for var, params in add.iteritems():
            if len(params) == 2:
                i, i_var = params
                map = None
            elif len(params) == 3:
                i, i_var, map = params
                # make sure map values are codes, not str
                for k, v in map.items():
                    if isinstance(k, basestring):
                        code = i_var.code_for_label(k, add=True)
                        map[code] = v
                        del map[k]
                        k = code
                    if isinstance(v, basestring):
                        code = var.code_for_label(v, add=True)
                        map[k] = code
            else:
                raise ValueError("Can only interpret len=2 or len=3 items in 'add' Param")
            i_target = self.varlist.index(var)
            i_source = source.varlist.index(i_var)
            source_array = data[i::n, i_source]
            if map:
                transfd = []
                for v in source_array:
                    if v in map:
                        transfd.append(map[v])
                    else:
                        transfd.append(np.NaN)
                source_array = np.array(transfd)
            out[:, i_target] = source_array

        return out



class Append(_Event_to_Event):
    """
    Append a variable with a fixed list of values to each segment.
    Setup with::

        >>> self.p.add[VarCommander] = LIST

    where LIST can be:
     - array-like (the same list is added to each segment)
     - [VarCommander, {val_x: LIST_x, ...}]
       VarCommander: determines a variable in the source segment-variable, the
       segment's value on which (val_x) determines which list is appended
       (LIST_x)

    """
    def _addparams_(self, p):
        self.p._add_('add', param.VarDict(desc="Variables that should be added "
                                          "to data"))
    def _new_vars(self):
        return self.p.add.get().keys()
    def _derive_segment_data_(self, segment, preview=False):
        source = segment._p_attr['source']
        add = self.compiled['add']
        e_list = deepcopy(source.data)
        for var, LIST in add.iteritems():
            if isvar(LIST[0]):
                keyvar, list_dic = LIST
                key = source[keyvar]  # ???
                if key not in list_dic:
                    key_label = keyvar.label(key)
                    if key_label in list_dic:
                        key = key_label
                    else:
                        msg = "Append: key {k}='{l}' not in list-dic"
                        raise KeyError(msg.format(k=key, l=key_label))
                new = np.array(list_dic[key])
            else:
                new = np.array(LIST)
            if new.ndim == 1:
                new = new[:, None]
            l_diff = e_list.shape[0] - new.shape[0]
            if l_diff > 0:
                logging.warning(" Append-LIST too short; nan-padded")
                new = np.vstack((new, np.array([[np.NAN]] * l_diff)))
            elif l_diff < 0:
                logging.warning(" Append-LIST too long; clipped")
                new = new[:l_diff, :]
            e_list = np.hstack((e_list, new))
        return e_list


class range_correction(_Event_to_Event):
    """
    Range correction of event values using powers or the Box-Cox family
    of transforms

    parameters
    ----------
    var: Variable: data to transform

    pre_mult: can be: scalar: input is multiplied by x before transformation;
              function: function is applied to the input and is expected to
              return a scalar by which to multiply
    pre_add: value is added before transformation, AFTER pre_mult
    box: (bool) Use Box-Cox family of transformations

    p: (default=1) x**p  or  (x**p-1)/p

    post_div: (default=np.max) value by which to divide input after transformation
              correction. valid: scalar, func, None

    """
#    def __init__(self, parent, name=None):
#        _Event_to_Event.__init__(self, parent, name)
    def _addparams_(self, p):
        self.p.var = param.Param(default=None, desc="Variable: data to transform")
        self.p.pre_mult = param.Param(default=None, desc="can be: scalar: input is "
                          "multiplied by x before transformation; function: function "
                          "is applied to the input and is expected to return a "
                          "scalar by which to multiply; None")
        self.p.pre_add = param.Param(default=0, desc="value is added before "
                                     "transformation, AFTER pre_mult")
        self.p.box = param.Param(dtype=bool, default=True, desc="Use Box-Cox "
                                 "family of transformations")
        self.p.p = param.Param(default=1, desc="x**p  or  (x**p-1)/p")
        self.p.post_div = param.Param(default=np.max, desc="value by which to "
                                      "divide input after transformation "
                                      "correction. valid: scalar, func, None")
    def _derive_segment_data_(self, segment, preview=False):
        source = segment._p_attr['source']
        data = deepcopy(source.data)
        var = self.compiled['var']
        pre = self.compiled['pre_mult']
        add = self.compiled['pre_add']
        post = self.compiled['post_div']
        p = self.compiled['p']
        box = self.compiled['box']

        # catch invalid cases
        if data.shape == (0,):
            return data
        elif var is None:
            logging.warning("range_correction 'var' = None")
            return data

        # find data to modify
        index = source.varlist.index(var)
        mod_data = data[:, index]
#        logging.debug("pre: %s"%mod_data)
        # PRE div
        if callable(pre):
            x = pre(mod_data)
        elif np.isscalar(pre):
            x = pre
        else:
            x = False
        if x:
            mod_data *= x

        # PRE add
        if add:
            mod_data += add
        # TRANSFORMATION
        if box:
            if p == 0:
                mod_data = np.log(mod_data)
            else:
                mod_data = (mod_data ** p - 1) / p
        else:
            if p == 0:
                mod_data = np.log10(mod_data)
            else:
                mod_data **= p

        # POST div
        if callable(post):
            x = post(mod_data)
        elif np.isscalar(post):
            x = post
        else:
            x = False
        if x:
            mod_data /= x

        # return data
#        logging.debug("post: %s"%mod_data)
        data[:, index] = mod_data
        return data


class Enum(_Event_to_Event):
    """
    counts the number of occurrence of a specific combination of values on
    the variables specified in p.count

    """
    def _addparams_(self, p):
        self.p.count = param.Address(desc="An address; each unique index of "
                                     "this address will be assigned one value")
        self.p.seq = param.Param(default=None, dtype=list, can_be_None=True,
                                 desc="A sequence can be assigned that will be "
                                 "used in place of counting")
        self.p.var = param.Variable(vartype=[int, float], can_be_None=False,
                                    desc="target variable")
    def _new_vars(self):
        return [self.compiled['var']]
    def _derive_segment_data_(self, segment, preview=False):
        source = segment._p_attr['source']
        count = self.compiled['count']
        seq = self.compiled['seq']
        data = deepcopy(source.data)

        # create enume sequence
        new = []
        if self.compiled['var']:  # if there is a variable to assign new data
            if count:
                indexes = {}  # keep track of occurrence of individual combinations
                for evt in source:
                    index = count.address(evt)
                    if index in indexes:
                        indexes[index] += 1
                    else:
                        indexes[index] = 0
                    i = indexes[index]
                    new.append(i)
            else:
                new = range(len(source))

            # apply seq
            if seq:
                diff = max(new) + 1 - len(seq)
                if diff > 0:
                    seq += [seq[-1]] * diff
                seq = np.array(seq)
                new = seq[new]
            else:
                new = np.array(new, dtype=int)
            data = np.hstack((data, new[:, None]))
        return data


class Shift(_Event_to_Event):
    """

    """
    def _addparams_(self, p):
        self.p.var = param.Param(del_segs=True, can_be_None=False,
                                 default=self.experiment.variables.get('time'),
                                 desc="Variable that is shifted")
        self.p.shift = param.Param(del_segs=True,
#                                   type='multi', valid=[list, dict, 'scalar'],
                                   desc="Shift values: scalar value / list of "
                                   "values / dict providing values")
        self.p.key = param.Param(del_segs=True,
                                 desc="Variable providing key if values is dict "
                                 "for enumeration")
    def _derive_segment_data_(self, segment, preview=False):
        source = segment._p_attr['source']
        var = self.compiled['var']
        key = self.compiled['key']
        shift = self.compiled['shift']
        data = deepcopy(source.data)

        var_i = source.varlist.index(var)

        if np.isscalar(shift):
            data[:, var_i] += shift

        elif type(shift) == list:
            data_len = data.shape[0]
            shift = deepcopy(shift)
            while len(shift) < data_len:
                shift.append(shift[-1])
            if len(shift) > data_len:
                shift = shift[:data_len]
            data[:, var_i] += np.array(shift)

        else:
            if isvar(key):
                for i, evt in enumerate(source):
                    data[i, var_i] += shift[evt[key]]
            else:
                for i, evt in enumerate(source):
                    data[i, var_i] += evt[shift]


        return data



