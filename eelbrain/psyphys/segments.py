"""
Segments
========

This module defines classes for representing segments of different types of data. Each segment
contains one continuous record of data, and stores properties such as the sampling-
rate or a sensor layout (for a list of possible properties see the bionetics module
documentation).


"""

from __future__ import division

import logging, time
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as P

from eelbrain import fmtxt
from eelbrain.vessels import colorspaces as _cs

from vars import isvar, asaddress, isaddress


def _get_data(other):
    "helper func for Segment operations"
    if hasattr(other, 'data'):
        data = other.data
    else:
        data = other
    return data


class Segmentlist(list):
    """
    list subclass to avoid oversized __repr__ and __str__ representations.
    class attribute _maxlen_ determines maximum length to print in full
    form.

    but see:
    http://stackoverflow.com/questions/5112019/customize-python-slicing-please-advise

    """
    _maxlen_ = 50
    def __repr__(self):
        length = len(self)
        if length > self._maxlen_:
            return "[%r, ...; N = %i]" % (self[0], length)
        else:
            return list.__repr__(self)
    def __getslice__(self, i, j):
        out = list.__getslice__(self, i, j)
        if isinstance(out, list):
            return Segmentlist(out)
        else:
            return out



class Segment(object):
    """
    VARIABLES:
     use Segment[varCommander] to access variables
     use Segment[int] or [slice] to access data

    DATA:
     .data: time x sensor x ...data...



    """
    _seg_type_ = 'single'
    repr_temp = "{c}({n}{dt})"
    def __init__(self, properties=None,
                 dataset=None, p_attr=None, id=None,
                 data=None,
                 variables=None,  # submit variables as dictionary
                 varsource=None,  # alternative to variables, in which the object is copied first
                 name="New Segment", symbol=None):
        # store simple args
        self.name = name
        self.symbol = symbol
        self._p_attr = p_attr

        if data is not None:
            if properties is None:
                properties = {}
            self._data = data
            properties['shape'] = data.shape

        if properties is not None:
            self._properties = deepcopy(properties)

        if dataset is None:
            if variables is not None:
                self.variables = variables
            elif varsource is None:
                self.variables = {}
            else:
                self.variables = varsource.copy()
        else:
            self.attach_to_dataset(dataset, varsource=varsource, id=id)

    def __len__(self):
        return self.shape[0]

    def __repr__(self):
        fmt = dict(c=self.__class__.__name__,
                   n='',
                   dt=self['data_type'])
        if self.name:
            fmt['n'] = '"%s", ' % str(self.name)
        return self.repr_temp.format(**fmt)

    def __getitem__(self, name):
        "accepts: VarCommanders, str(->properties), indices(->data)"
        if isvar(name):
            return self.variables[name]
        elif isinstance(name, basestring):
            if name in self.properties:
                return self.properties[name]
            elif hasattr(self, 'dataset'):
                return self.dataset[name]
            else:
                raise KeyError("%r not in properties" % name)
        else:
            return self.data[name]

    def __setitem__(self, name, value):
        if isvar(name):
            self.variables[name] = value
        elif isinstance(name, basestring):
            self.properties[name] = value
        else:
            raise KeyError("Key must be VarCommander or String (for setting properties)")

    def attach_to_dataset(self, dataset, varsource=None, id=None):
        """
        Delete Segment._data, and set Segment.dataset, which will cause calls
        to several attributes to be redirected to the dataset.

        """
        if hasattr(self, 'dataset'):
            raise ValueError("Segment is already attached")

        self.dataset = dataset
        self._id = id

        # append to dataset
        if not hasattr(dataset, '_segments'):
            dataset._segments = Segmentlist()
            dataset._id2id = {}  # maps _id -> id
        dataset._id2id[self._id] = self.id = len(dataset._segments)
        dataset._segments.append(self)

        # TODO: copy variable values from source dataset
        self.variables = dataset.experiment.variables.getNewColony(copy=varsource)

        # store data
        if hasattr(self, '_data'):
            data = self._data
            del self._data
            self.dataset._set_data(self.id, data)

    @property
    def data(self):
        if hasattr(self, '_data'):
            return self._data
        else:
            return self.dataset._get_data(self.id)

    @property
    def properties(self):
        "returns properties; can not be modified directly"
        if not hasattr(self, '_properties'):
            self._properties = self.dataset._derive_segment_properties_(self)
        return deepcopy(self._properties)

    def get_full_properties(self):
        if hasattr(self, 'dataset'):
            properties = self.properties
            properties.update(self.dataset.properties)
            if hasattr(self.dataset, 'compiled'):
                properties.update(self.dataset.compiled)
            if hasattr(self.dataset, 'varlist'):
                properties['varlist'] = [v for v in self.dataset.varlist]
            return properties
        else:
            return deepcopy(self.properties)

    # some important properties:
    @property
    def ndim(self):
        return self['ndim']

    @property
    def shape(self):
        try:
            return self['shape']
        except:
            shape = self.data.shape
            self.properties['shape'] = shape
            return shape

    @property
    def t0(self):
        return self['t0']

    @property
    def tstart(self):
        return -self.t0

    @property
    def tend(self):
        return self.duration - self.t0

    @property
    def color(self):
        if hasattr(self, 'dataset'):
            return self.dataset.p.color.get()
        else:
            try:
                return self['color']
            except:
                return None

    def set_color(self, *color_args):
        """
        matplotlib color, e.g.
        >>> segment.set_color(1, 0, 0)
        >>> segment.set_color(...)

        """
        if len(color_args) == 1:
            color = color_args[0]
        else:
            color = color_args
        if color != self['color']:
            self._properties['color'] = color




class Event_Segment(Segment):
    """
    requires the following properties:

    ds:    varlist
    local: duration

    """
    def __init__(self, properties=None, varlist=None, **kwargs):
        Segment.__init__(self, properties=properties, **kwargs)
        self._varlist = varlist
#    def _set_data(self, data, properties={}):
#        "properties update self._properties"
#        data = np.array(data)
#        # check data
#        if data.ndim == 2:
#            assert len(self.varlist) == data.shape[1]
#        elif data.ndim==0 or data.shape[0]==0:
#            data = np.zeros((0, len(self.varlist)))
#        else:
#            raise ValueError("EventSegment cannot be initialized with: "+str(data))
#        # set data
#        self.properties.update(properties)
#
#        self.properties['shape'] = data.shape
#        self.data = data

    def __str__(self):
        return str(self.astable())

    def __iter__(self):
        for evt in self.sub_iter(None):
            yield evt

    def __reversed__(self):
        for evt in self.sub_iter(None, rev=True):
            yield evt

    def __getitem__(self, name):
        """
        returns, for the following inputs:
        int -> return event with index
        Address ->
        else -> Segment.__getitem__

        """
#        if hasattr(name, 'contains'):
        if isvar(name) or isinstance(name, basestring):
            return Segment.__getitem__(self, name)
        elif np.isscalar(name):
            if int(name) != name:
                raise ValueError("index needs to be int")
            if name >= len(self.data):
                raise KeyError("n events < %s" % name)
            else:
                values = self.data[name]
                evt = self._get_static_vars()
                for var, val in zip(self.varlist, values):
                    evt[var] = val
                return evt
        else:
            properties = self.get_full_properties()
            if isaddress(name):
                select = []
                for i, evt in enumerate(self):
                    if name.contains(evt):
                        select.append(i)
                data = self.data[select]
            else:
                data = self.data[name]
                assert data.shape[1] == len(self.varlist)

            return Event_Segment(properties, self.varlist[:],
                                 varsource=self.variables, data=data,
                                 name=self.name)

    def _get_static_vars(self):
#        varlist = [(var, val) for var, val in self.variables.asdict()
#        return dict([(var, val) for var, val in self.variables.variables.iteritems() if var not in self.varlist])
        return self.variables.copy()
        out = {}
        var_ids = [id(v) for v in self.varlist]
        for var, val in self.variables:
            if id(var) not in var_ids:
                out[var] = val
        return out

    def astable (self):
        data = self.data
        varlist = self.varlist

        parasites = []
        for par in varlist[0].mothership.parasites:
            if all([h in varlist for h in par.hosts]):
                parasites.append(par)

        if data.ndim == 1:
            table = fmtxt.Table('l' * len(varlist))
            [table.cell(v.name) for v in varlist]
            [table.cell(var[v]) for var, v in zip(varlist, data)]
        else:
            table = fmtxt.Table('r' + 'l' * (len(varlist) + len(parasites)))
            table.cell()
            [table.cell(v.name) for v in varlist + parasites]
            table.midrule()
            for i, values in enumerate(data):
                table.cell(i)
                items = zip(varlist, values)
                [table.cell(var.repr(v)) for var, v in items]
                asdict = dict(items)
                for par in parasites:
                    v = par[asdict]
                    table.cell(par.repr(v))
        return table

    @property
    def duration(self):
        return self['duration']

    def range(self, var, min=None, max=None, out='seg'):
        d = self.subdata(var=var, out='data')
        # get indexes
        indexes = np.ones(len(d), dtype=bool)
        if min != None:
            indexes *= d >= min
        if max != None:
            indexes *= d < max
        # collect subdata
        data = self.data[indexes, :]
        # return
#        print "RANGE shape %s"%str(data.shape)
        if out == 'data':
#            print "RANGE return data"
            return data
        elif out == 'seg':
#            print "RANGE return seg"
            return Event_Segment(data, self.get_full_properties(),
                                 varlist=self.varlist, name=self.name)
        else:
            raise NotImplementedError("only out in ['seg', 'data']")

    def sub_iter(self, var, min=None, max=None, rev=False):
        """
        generator function that iterates over a subset of events

        """
        if var is None:
            data = self.data
        else:
            data = self.range(var, min, max, out='data')

        if rev:
            data = reversed(data)

        static = self._get_static_vars()
        for values in data:
            new = dict([(var, val) for var, val in zip(self.varlist, values)])
            static.update(new)
            # parasotes will work automatically
#            for par in self.variables.mothership.parasites:
#                static[par] = par[static]
            yield static

    def subdata(self,
                tstart=None, tend=None, t0=None,
                var=None,
                vars=None,
                pad=False,  # add 1 event at the back and at the front
                out='segment', dataset=None):
        """
        t0: reset time axis. (relative to tstart)
        if t0==None: time axis remains constant

        var: returns data of one variable
        vars: returns segment containing only vars

        tstart/tend follow the python principle of including [tstart, tend[

        dataset -> out is Bound

        """
        # get attributes
        varlist = self.varlist[:]
        data = self.data
        duration = self.duration
        ##   modify   #####   #####   #####   #####
        # variables
        if len(data) > 0:
            if var:
                var_i = self.varlist.index(var)
                data = data[:, var_i]
                out = 'data'
            elif vars:
                var_i = [self.varlist.index(v) for v in vars]
                data = data[:, var_i]
                varlist = vars
            # time
            if tstart or tend:
                t = self.t
                if tstart is not None:
                    index = t >= tstart
                else:
                    index = np.ones(len(t)).astype(bool)
                if tend is not None:
                    index -= (t >= tend)
                if pad:
                    frame = np.where(np.diff(index))[0]
                    if len(frame) == 2:
                        a, b = frame
                        b += 1
                    else:
                        logging.warning(" event for padding missing in %s" % self.name)
                        if len(frame) == 1:
                            if frame[0] == 0:
                                a = None
                                b = frame[0] + 1
                            else:
                                a = frame[0]
                                b = None
                        elif len(frame) == 0:
                            a = b = None
                        else:
                            raise NotImplementedError("non-contiguous")
                    index[[a, b]] = True
                # index = index[:,None]
                data = data[index]
            if t0:
                if tstart is not None:
                    t0 = tstart + t0
                index = self.varlist.index(self.timevar)
                data[:, index] -= t0
        ##   out   #####   #####
        if out == 'data':
            return data
        elif out == 'segment':
            if dataset:
                new_seg = Event_Segment(self.properties, dataset=dataset,
                                        data=data, name=self.name,
                                        varsource=self.variables)
            else:
                properties = self.get_full_properties()
#                varlist = [v.name for v in varlist]
                new_seg = Event_Segment(properties, data=data, varlist=varlist,
                                        name=self.name)
            new_seg['duration'] = duration
            return new_seg
        else:
            raise ValueError("out: 'data' or 'segment'")

    @property
    def t(self):
        return self.subdata(var=self.timevar)

    @property
    def timevar(self):
        if hasattr(self, 'dataset'):
            return self.dataset.experiment.variables.get('time')
        else:
            return 'time'

    def uts(self, var='magnitude',
            tstart=None, tend=None, dur=None,
            samplingrate=10,
            uts='lin',  # mw = moving window;
            winfunc=np.blackman,
            windur=1,
            out='data'):
        """
        dur: alternative to tend to ensure equal data length

        'mw' moving window
        'lin' linear interpolation
        """
        if isinstance(var, basestring):
            var = self.dataset.experiment.get_var_with_name(var)
        time = self.dataset.experiment.variables.get("time")
        step = 1 / samplingrate
        #
        if dur:
            T = np.arange(dur * samplingrate) / samplingrate + tstart
            tend = T[-1] + step
        else:
            T = np.arange(tstart, tend, step)
        if uts == 'lin':
            data = self.subdata(tstart, tend, vars=[time, var], pad=True,
                                out='data')
#            t1 = round(data[0,0] * samplingrate) / samplingrate
#            t2 = round(data[-1,0]* samplingrate) / samplingrate
            Y = np.interp(T, data[:, 0], data[:, 1])
        elif uts == 'mw':
            te = windur / 2
            data = self.subdata(tstart - te, tend + te, vars=[time, var],
                                out='data')
            w_len_samples = max(round(windur * samplingrate), 1)  # window length in samples
            if w_len_samples == 2:
                w_len_samples = 1
            window = winfunc(w_len_samples)
            window /= np.sum(window) / samplingrate
            hw = windur / 2
            Y = np.zeros(len(T))
            if w_len_samples == 1:
                raise NotImplementedError("Window len == 1")
            else:
                for t, y in data:
                    T_loc = np.arange(w_len_samples) * step + t - hw
                    Y += np.interp(T, T_loc, y * window, left=0, right=0)
        else:
            raise NotImplementedError()

        if out == 'data':
            return Y
        else:
            raise NotImplementedError

    @property
    def varlist(self):
        if hasattr(self, 'dataset'):
            return self.dataset.varlist
        else:
            return self._varlist

    ## Plotting---
    def asdata(self):
        samplingrate = self['samplingrate']
        n = self.duration * samplingrate
        out = np.zeros(n)
        #
        time = self.dataset.experiment.variables.get('time')
        dur = self.dataset.experiment.variables.get('duration')
        mag = self.dataset.experiment.variables.get('magnitude')
        for evt in self:
            start = evt[time] * samplingrate
            if dur in evt:
                end = int(start + (evt[dur] * samplingrate))
            else:
                end = int(start + .5 * samplingrate)
            if mag in evt:
                y = evt[mag]
            else:
                y = 1
            out[start:end] = y
        return out[:, None]

    def toax(self, start=None, end=None, ax=None, mag=True, dur=False,
             labels=None, colors=None, **kwargs):
        """
        mag:     varCommander to var providing magnitude
        labels:  varCommander
        colors:  varCommander with .color_for_value
        """
        if not ax:
            ax = P.gca()
        if mag == True:
            mag = self.dataset.experiment.variables.get('magnitude')
        time = self.dataset.experiment.variables.get('time')
#        time_i = self.varlist.index(time)
#        duration = self.dataset.experiment.variables.get('duration')
        kwargs.update(self.plot_kwargs)
        for evt in self.sub_iter(time, start, end):
            # print evt
#            e_start = evt[time]
#            if dur:
#                e_end = e_start + dur
#            elif duration in evt:
#                e_end = e_start + evt[duration]
#            else:
#                e_end = e_start + 1.
#            if mag in evt:
#                y = evt[mag]
#            else:
#                y = 1
            if colors:
                kwargs['color'] = colors.color_for_value(evt[colors])
            else:
                kwargs['color'] = self.color
            # print "plot: {0} - {1}".format(e_start, e_end)
#            span = ax.axvspan(e_start, e_end, ymax=y, **kwargs)
            if labels:
                raise NotImplementedError
#                ax.annotate(labels[data[labels]], (start, y/2.))
        if not start:
            start = self.tstart
        if not end:
            end = self.tend
        ax.set_xlim(start, end)




class UTS_Segment(Segment):
    @property
    def duration(self):
        """ in seconds """
        return self.__len__() / float(self['samplingrate'])
    @property
    def t(self):
        "the t-axis of the data"
        x = self.shape[0]
        t = np.arange(x, dtype=np.float) / self.samplingrate + self.tstart
        # t = np.arange(self.tstart, self.tend, 1. / self.samplingrate)
        return t
    @property
    def samplingrate(self):
        return self['samplingrate']
    @property
    def sensors(self):
        return self['sensors']
    def defaultColorspace(self, vmax='auto', vmin='auto', **kwargs):
        "When symmetric, vmax is more important"
        if vmax == 'auto':
            vmax = np.max(np.abs(self.data))
        kwargs['vmax'] = vmax

        if vmin != 'auto':
            kwargs['vmin'] = vmin

        if self.ndim == 'wavelet':
            raise NotImplementedError
        else:
            return _cs.get_MEG(**kwargs)
    # # operations
    # TODO: operations on VarColonies
    def _get_name(self, other, bracket=[], maxlen=20):
        if hasattr(other, 'name'):
            name = other.name
        else:
            name = str(other)[:maxlen]
        # cracket
        if any([b in name for b in bracket]):
            return '(' + name + ')'
        else:
            return name
    def __add__(self, other):
        data = self.data + _get_data(other)
        selfname = self._get_name(self)
        othername = self._get_name(other)
        name = '+'.join([selfname, othername])
        return self._create_child(data, name=name)
    def __sub__(self, other):
        data = self.data - _get_data(other)
        selfname = self._get_name(self)
        othername = self._get_name(other)
        name = '-'.join([selfname, othername])
        return self._create_child(data, name=name)
    def __mul__(self, other):
        data = self.data * _get_data(other)
        selfname = self._get_name(self, bracket=['+', '-'])
        othername = self._get_name(other, bracket=['+', '-', '/'])
        name = '*'.join([selfname, othername])
        return self._create_child(data, name=name)
    def __div__(self, other):
        data = self.data / _get_data(other)
        selfname = self._get_name(self, bracket=['+', '-', '*'])
        othername = self._get_name(other, bracket=['+', '-', '*', '/', 'frac'])
        name = '{%s}/{%s}' % (selfname, othername)
        # name = '\\frac{%s}{%s}'%(selfname, othername)
        return self._create_child(data, name=name)
    def _get_child_properties(self, mod_props, rm_props, full=False):
        # get self's properties
        if full:
            properties = self.get_full_properties()
        else:
            properties = deepcopy(self.properties)

        # modify properties
        if mod_props:
            properties.update(mod_props)

        # remove properties
        if rm_props:
            for prop in rm_props:
                if prop in properties:
                    properties.pop(prop)

        return properties

    def _create_child(self, data, mod_props={}, rm_props=[], dataset=None,
                      name="{name}", **kwargs):
        kwargs['name'] = name.format(name=self.name)
        full = not bool(dataset)
        properties = self._get_child_properties(mod_props, rm_props,
                                                full=full)
        if dataset:
            return self.__class__(properties, data=data, dataset=dataset, **kwargs)
        else:
            return UTS_Segment(properties, data=data, **kwargs)
    # # properties
    def timeTitleForSample(self, i, short=False, unit='ms'):
        timestep = int(1000. / self['samplingrate'])
        baseline = int(self.t0 * 1000)
        if short:
            return "%sms" % (i * timestep - baseline)
        else:
            return "%s-%sms" % (i * timestep - baseline, (i + 1) * timestep - baseline)
    # accessing sub Data
    def i_for_t(self, t):
            i = round((t + self.t0) * self.samplingrate)
            if i < 0:
                i = 0
            return int(i)
    def sliceForInterval(self, interval):
        start, end = interval
        if start != None:
            start = self.i_for_t(start)
        if end != None:
            end = self.i_for_t(end)
        return slice(start, end)
    def toax(self, start=None, end=None, ax=None, mag=False, dur=False,
             labels=None, colors=None, **kwargs):
        """
        mag:     varCommander to var providing magnitude
        labels:  varCommander
        colors:  varCommander with .color_for_value
        """
        if not ax:
            ax = P.gca()
        T = self.t
        data = self.data
        if start:
            start = int((self.t0 + start) * self.samplingrate)
            T = T[start:]
            data = data[start:]
        if end:
            end = int((self.t0 + end) * self.samplingrate) - start
            T = T[:end]
            data = data[:end]
        ax.plot(T, data)
        ax.set_xlim(start, end)
    """

    the subdata method for UTS data;
    requires dimensionality:

    time x sensor x ...data... [ x subject ]

    """
    def subdata(self,
                # montage modification
                sensors=None, sensor=None, ref=None,
                ROI=None, ROIs=None,
                # temporal cropping
                interval=None,  # data within interval
                tstart=None, tend=None,
                padding=False,
                # temporal aggregating
                tw=None,  # time window (aggregated data)
                t=None,  # pick the closest sample
                tw_func=np.mean,  # function for aggregating data; must take axis kwarg
                merge=False,
                # sampling frequency
                downsample=None, samplingrate=None,
                # baseline
                baseline=None,  # TODO: blstart, blend, ...
                # output options
                out='segment',
                # ONLY for StatsSegments
                cases=None,  # select  subset of cases
                exclude=None,  # exclude subjects (list of s ids/names)
                ** seg_kwargs
                ):
        """
        Returns a modified version of the segment's data.

        Timing
        ------

        downsample: Int
            divide current samplingrate by <int>
        samplingrate:
            Change the samplingrate; needs to be a divisor of the
            current samplingrate

        interval: tuple(start, end)
            extract data from a specified time interval, specified as, in
            seconds (e.g. interval=(.4,.6))
        tw: tuple(start, end)
            like interval, but returns the average over time for the specified
            time window
        t: float
            return the sample closest to t


        Montage
        -------

        sensors:
            list of sensor ids (default is all)
        ROIs:
            list of lists specifying sensors which will be averaged to
            constitute one data channel in the output

        baseline: tuple(start, end)
            specifies baseline for data (tuple like interval)
        ref:
            list of sensors for re-referenceing (True=av-ref; default None)


        Data output
        -----------

        out: string
            can be ``'data'``, ``'tsv'``, or ``'segment'`` (default)
        padding:
            How to handle cases wher ethe requested segment exceeds the
            current segment
            ``False``: raise Error when requested interval exceeds data
            ``True``: pad with constant
            float: pad with this value

        """
        mod_props = {}
        rm_props = []

        ###   ARGUMENTS   ################################################

        # translate convenience arguments
        if sensor is not None:
            assert sensors is None, "sensors/sensor are mutually exclusive arguments"
            assert isinstance(sensor, int), "type(sensor) needs to be int"
            sensors = [sensor]

        if ROI != None:
            assert ROIs == None and sensors == None
            ROIs = [ROI]

        if 'dataset' in seg_kwargs:
            out = 'segment'


        # time cropping
        assert sum([bool(tw), bool(interval), bool(tstart) or bool(tend), (t != None)]) <= 1, \
               "Time window/interval over-specified"
        if tw:
            assert (downsample == None) and (samplingrate == None)
            tstart, tend = tw
            merge = True
        elif interval:
            tstart, tend = interval

        if np.isscalar(tstart) and np.isscalar(tend):
            assert tstart < tend, "Empty Time Interval (tstart >= tend)"
        elif np.isscalar(t):
            assert samplingrate is None, "Prameter conflict: t"
            assert merge is None, "Prameter conflict: t"

        if padding is False:
            if np.isscalar(tstart) and (self.tstart > tstart):
                raise ValueError("requested data starts before segment")
            if np.isscalar(tend) and (self.tend < tend):
                raise ValueError("requested data exceeds segment")


        # samplingrate
        if sum(bool(prop) for prop in [downsample, samplingrate, merge]) > 1:
            raise ValueError("downsample, samplingrate and merge are mutualy exclusive")
        if downsample != None:
            samplingrate = self.samplingrate / downsample



        ###   MODIFY DATA   ################################################

        # start -- take views
        data = self.data

        # exclude subjects
        if exclude or cases:
            assert iscollection(self), "'exclude' arg only valid for stats-segments"
            assert not (cases and exclude), "conflicting arguments"

            # create new slist
            if cases:
                if self.svar:
                    slist = self.svar.values(cases)
                else:
                    slist = cases
            else:
                if self.svar:
                    exclude = self.svar.values(exclude)
                slist = []
                for sid in self.slist:
                    if sid not in exclude:
                        slist.append(sid)

            # select data
            indexes = [self.slist.index(sid) for sid in slist]
            data = data[..., indexes]
            mod_props['slist'] = slist

        # montage
        if ref:
            if ref == True:
                data_ref = data.mean(1)[:, None]
            else:
                if np.isscalar(ref):
                    ref = [ref]
                data_ref = data[:, ref].mean(1)[:, None]
        if ROIs != None:
            channels = [data[:, ROI_sensors].mean(1)[:, None] for ROI_sensors in ROIs]
            data = np.concatenate(channels, 1)
            sensornet = self['sensors']
            mod_props['sensors'] = sensornet.get_subnet_ROIs(ROIs)
        elif sensors != None:
            data = data[:, sensors]
            if len(sensors) < 2:
                rm_props.append('sensors')
            else:
                mod_props['sensors'] = self.sensors.get_subnet(sensors)

        if ref:
            data = data - data_ref

        if baseline != None:
            data = data - data[self.sliceForInterval(baseline)].mean(0)
            mod_props['baseline'] = baseline

        # time interval
        if np.isscalar(tstart) or np.isscalar(tend):
            pad_1 = 0
            pad_2 = 0
            if tstart is None:
                i1 = None
            else:
                mod_props['t0'] = -tstart
                i1 = self.i_for_t(tstart)
                if i1 < 0:
                    pad_1 = -i1
                    i1 = 0
            if tend is None:
                i2 = None
            else:
                i2 = self.i_for_t(tend)
                i_max = self.i_for_t(self.tend)
                if i2 > i_max:
                    pad_2 = i2 - i_max
                    i2 = i_max
            data = data[i1:i2]
            if pad_1:
                if padding == True:
                    pad = data[0]
                else:
                    pad = np.ones(data.shape[1:]) * padding
                data = np.vstack([pad] * pad_1 + [data])
            if pad_2:
                if padding == True:
                    pad = data[-1]
                else:
                    pad = np.ones(data.shape[1:]) * padding
                data = np.vstack([data] + [pad] * pad_2)

        # time point
        elif np.isscalar(t):
            T = self.t
            if t in T:
                i = np.where(T == t)[0][0]
            else:
                i_next = np.where(T > t)[0][0]
                t_next = T[i_next]
                i_prev = np.where(T < t)[0][-1]
                t_prev = T[i_prev]
                if (t_next - t) < (t - t_prev):
                    i = i_next
                    t = t_next
                else:
                    i = i_prev
                    t = t_prev

            data = data[i:i + 1]
            mod_props['t0'] = -t


        # downsample
        if samplingrate != None:
            if self.samplingrate % samplingrate:
                raise ValueError("Samplingrate %s cannot be downsampled to new "
                                 "frequency %s" % (self.samplingrate, samplingrate))
            timestep = int(self.samplingrate / samplingrate)
            newLength = data.shape[0] // timestep
            if data.shape[0] % timestep != 0:
                logging.warning("length %s not divisible by new timestep %s. "
                                "End will be cut off." % (data.shape[0], timestep))
                data = data[ : newLength * timestep]
            newShape = (newLength, timestep,) + data.shape[1:]
            data = data.reshape(newShape).mean(1)
            mod_props['samplingrate'] = samplingrate

        # merge time-dimension
        elif merge:
            data = tw_func(data, axis=0)[None, ...]
            rm_props.append('samplingrate')
            if tstart != None:
                mod_props['t0'] = -tstart

            if tend == None:
                mod_props['tend'] = self.tend
            else:
                mod_props['tend'] = tend


        ###   OUT   ################################################
        if out == 'segment':
            for attr in ['name', 'symbol']:
                if attr not in seg_kwargs:
                    seg_kwargs[attr] = deepcopy(getattr(self, attr))
            return self._create_child(data, mod_props=mod_props, rm_props=rm_props,
                              **seg_kwargs)
        elif out == 'tsv':
            txt = '\t'.join([ s.name for s in [ self.sensors[i] for i in sensors ] ]) + '\n'
            txt += '\n'.join([ '\t'.join(s.astype('S8')) for s in data ])
            return txt
        elif out == 'data':
            return data
        else:
            raise NotImplementedError("can only get data, tsv, or segment (tried to get %s)" % out)

# plotting
    """
    def toAxes_plotStart_length_forSensor_(self, axes, start=None, length=None, sensor=0, label=None, **kwargs):
        "
        !!! interval is SLICE
        plots one sensor (index) for a list f segments
        return plot
        start/length in s !

        "
        logging.warning(" Segment.toAxes_plotStart_length_forSensor_ Not tested with new Shape")
        if self.hasStatistics:
            plot = self.toAxes_plotStatsStart_length_forSensor_(axes, start, length, sensor, label, kwargs)
        else:
            plotData = self.subdata( sensors=[sensor], interval=(start, start+length) )
            #print "data shape:", self.data.shape
            if self.ndim == 1:
                plot = axes.plot(plotData, label=label, **kwargs)
                axes.set_xlim(xmin=0, xmax=len(plotData))
            elif self.ndim == 2:
                #print "im shape: ", plotData.shape
                plot = axes.imshow(plotData, aspect='auto', origin='lower')#, vmin=ravel(self.data).std(), vmax=ravel(self.data).std(), **kwargs)
            # ticklabels
            ticks = linspace( 0, length*self.samplingrate, num=6, endpoint=True )
            axes.xaxis.set_ticks(ticks)
            axes.xaxis.set_ticklabels((ticks/self.samplingrate)+start)
            # events
            for e in self.events:
                if start<e.start<(start+length):
                    logging.debug("event start: %s, duration: %s"%(e.start, e.end-e.start))
                    axes.axvspan((e.start-start)*self.samplingrate, (e.end-start)*self.samplingrate, facecolor=e.facecolor, edgecolor = 'r', alpha=.5)
            return plot
    def plot(self, plotType='e', newFig=False, **kwargs):
        "possible types: 'e', 'im'"
        if newFig:
            fig = P.figure()
            ax = P.axes([.05, .05, .9, .9])
        if self.ndim == 1:
            if plotType=='e':
                eegplot.ax_eArray(self)
            elif plotType=='im':
                if 'colorspace' not in kwargs:
                    kwargs['colorspace']=self.defaultColorspace()
                eegplot.ax_imArray(self)
        elif self.ndim == 'wavelet':
            eegplot.fig_waveletArray(self, **kwargs)
        else:
            raise NotImplementedError("plotting for ndim > 1")
    """


"""
Combinations

"""



class StatsSegment(UTS_Segment):
    """
    Segment which contains a collection of data (e.g., ERP for one condition
    for several subjects).

    Representation::

        time x sensor x ...[data]... x subject

    Attributes which help reconstructing data origin:

    self.slist:
        contains list of subjects
    self.svar:
        is name of slist variable
    self.avars:
        vars composing address
    self.address:
        values on avars

    ??? some plotting functions use .symbol.to_axes(ax) if available

    """
    _seg_type_ = 'collection'
    def __init__(self, properties, data, **kwargs):
        """
        Arguments
        ---------

        properties: <dictionary>
            for the possible properties see
        data: None or <np.array>
            the data for the segment, with the following dimensional structure:
            ``time x sensor x ...[data]... x subject``

        """
        self.N = N = data.shape[-1]
        properties['shape'] = data.shape
        if 'slist' in properties:
            Nl = len(properties['slist'])
            if Nl != N:
                raise ValueError("slist length (%i) != data shape (%i)" % (Nl, N))
            if 'svar' not in properties:
                raise ValueError("'slist' in properties, but not 'svar'")
        else:
            if 'svar' in properties:
                raise ValueError("'svar' in properties, but not 'slist'")
            properties['slist'] = range(data.shape[-1])
            properties['svar'] = None

        UTS_Segment.__init__(self, properties, data=data, **kwargs)

    def get_case(self, case):
        """
        Returns a UTS_Segment for only one case in the collection. The ``case``
        argument should be specified as element of slist.

        """
        if case not in self.slist:
            raise KeyError("This collection has no case named %r; check the "
                           ".slist attribute for all case names." % case)
        i = self.slist.index(case)
        data = self.data[..., i]
        name = self.name + '[%s]' % case
        return self._create_child(data, name=name)

    def initWithSegments_over_(self, segments, over, exclude=None,
                               func=None, mp=False, **kwargs):
        """
        kwargs:
         func=None:     is initially applied to the whole data cube; can be string
                        'phase'--> x/abs(x)
                        else: eval
         mp=True:       use multiprocessing
         v=False        verbose mode

         """
        raise NotImplementedError("TODO: get this method out of here!")
        # interpret func
        if func == None:
            def func(x):
                return x
        elif func == 'phase':
            def func(x):
                return x / abs(x)
        elif type(func) == str:
            func = eval(func)
        # logging.debug( 'func'+str(func) )

        # start
        t0 = time.time()

        self._properties.update(kwargs)
        kwargs['out'] = 'data'

        if exclude:
            over = over != exclude
        else:
            over = asaddress(over)

        segs_sorted = over.sort(segments)

        slist = segs_sorted.keys()
        slist.sort()
        self._properties.update({'slist':slist,
                                'svar':over,
                                'avars': '???',
                                'address': '???'})

        # determine shape
        seg1 = segs_sorted[slist[0]][0]
        data1 = seg1.subdata(**kwargs)
        shape = data1.shape

        # create empty array
        self._data = np.empty(shape + (len(slist),))

        # collect data
        for i, s in enumerate(slist):
            segs = segs_sorted[s]
            assert len(segs) > 0, "No data for %s" % str(s)
            data = np.empty(shape + (len(segs),))
            for j, seg in enumerate(segs):
                segdata = seg.subdata(**kwargs)
                if shape == segdata.shape:
                    data[..., j] = segdata
                else:
                    msg = "'{n}' shape {s1} does not match the shape {s}"
                    msg.format(n=seg.name, s1=segdata.shape, s=shape)
                    raise ValueError(msg)

            self._data[..., i] = data.mean(-1)


        # info
        # segInfo = ' '.join(['%s:%s'%(k, str(len(v)).rjust(3)) for k,v in segments.iteritems()])
        print ' '.join([self.name[:10], 'over', over.name[:10],
                        ": time %.1f s" % (time.time() - t0)])
        return self
    @property
    def svar(self):
        return self.properties['svar']
    @property
    def slist(self):
        return self.properties['slist']
    @property
    def slabels(self):
        return self.svar.labels(self.slist)
    def mean(self, name=False):
        """
        return mean of collection segment.

        kwargs
        ------
        name = False -> don't modify name
             = True -> mean(old name)
             = string -> newname = name.format(name = self.name)

        """
        # name
        if name in [False, None]:
            name = self.name
        else:
            if name == True:
                name = 'mean({name})'
            name = name.format(name=self.name)

        # data
        data = self.data.mean(-1)
        return UTS_Segment(self.properties, data=data, name=name, symbol=self.symbol)
    # # getting derived stuff
    def _create_child(self, data, mod_props=None, rm_props=None, name=None,
                      **kwargs):
        if name is None:
            name = deepcopy(self.name)

        if data.ndim == self.data.ndim:  # collection
            properties = self._get_child_properties(mod_props, rm_props)
            seg = StatsSegment(properties, data, name=name, **kwargs)
        else:  # single data segment
            rm = ['slist', 'svar']
            if rm_props:
                rm.extend(rm_props)
            properties = self._get_child_properties(mod_props, rm)
            seg = UTS_Segment(properties, data=data, name=name, **kwargs)
        return seg




### MARK: ##### STATISTICAL TESTS #####


class StatsTestSegment(UTS_Segment):
    """
    param is the statistical parameter
    .sources:   list of source segments (for later reference)
    .source:    first source segment (for getting values like t0)

    if there is a direction, p and colorspace are automatically adjusted
    """
    _seg_type_ = 'test'
    def __init__(self, properties, param, P, dir=None, statistic='T', **kwargs):
        UTS_Segment.__init__(self, properties, **kwargs)
        self.param = param
        self.P = P
        self.dir = dir
        self.statistic = statistic
    @property
    def data(self):
        if self.dir == None:
            return self.P
        else:
            return self.dirP
    @property
    def dirP(self):
        return (1 - self.P) * self.dir
    def defaultColorspace(self, p=.05, **kwargs):
        # print "segment.defaultColorspace", kwargs
        if self.dir == None:
            return _cs.SigColorspace(p, **kwargs)
        else:
            return _cs.SigColorspace(p, **kwargs)
    def subdata(self, **kwargs):
        if any(kw in kwargs for kw in ('downsample', 'samplingrate', 'ref')):
            raise KeyError("Invalid argument for p-field subdata")
        else:
            return UTS_Segment.subdata(self, **kwargs)
    def _create_child(self):
        raise NotImplementedError("can't get child of test segment")



def issegment(item):
    return hasattr(item, '_seg_type_')

def istest(segment):
    """
    returns True if segment contains a p-field (i.e. it is a StatsTestSegment)

    """
    if issegment(segment):
        return segment._seg_type_ == 'test'
    else:
        return False

def iscollection(segment):
    """
    returns True if segment contains data from a collection of segments (i.e.
    it is a StatsSegment)

    """
    if issegment(segment):
        return segment._seg_type_ == 'collection'
    else:
        return False

