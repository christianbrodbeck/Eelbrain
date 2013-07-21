"""

Collector objects scan datasets to extract statistics from them. They return
statistics as dataframes through the :meth:`get_dataframe` method.

"""
from __future__ import division

import os

import numpy as np

import vars as _vars
import eelbrain.vessels.data as _vsl 

__hide__ = ['ui', 'np', 'logging']



class _timewindow(object):
    """
    Holds the definition of a time window
    
    """
    def __init__(self, tstart=0, tend=None, dur=None, delay=None, time='time'):
        """
        t0 is always event[time]; A time window is specified through 'tstart' 
        and either 'tend' or 'dur' (=duration) 
        
        tend as Var is read relative to t0
        
        kwargs
        ------
        tstart = 0 (relative to t0)
        tend 
        dur (duration)
        delay (constant shift of time points)
        time: event-Variable specifying t0
        
        """
        # check input
        for t in tstart, tend, dur, delay:
            if t is not None:
                assert np.isscalar(t) or _vars.isvar(t)
        self.tstart = tstart
        assert (tend is None) or (dur is None)
        self.tend = tend
        self.dur = dur
        self.delay = delay
        assert isinstance(time, basestring) or _vars.isvar(time)
        self.time = time
    def __repr__(self):
        temp = "_timewindow({tstart}, {kwargs})"
        kwtemp = "{n}={v}"
        kwargs = []
        for kw in ['tend', 'dur', 'delay']:
            arg = getattr(self, kw)
            if arg is not None:
                kwargs.append(kwtemp.format(n=kw, v=arg.__repr__()))
        return temp.format(tstart=self.tstart, kwargs=', '.join(kwargs))
    def _add_delay(self, t, colony):
        if self.delay is None:
            return t
        elif np.isscalar(self.delay):
            return t + self.delay
        elif _vars.isvar(self.delay):
            return t + colony[self.delay]
    def t1(self, colony):
        "return t1 relative to the var-colony"
        t0 = colony[self.time]
        if _vars.isvar(self.tstart):
            out = t0  + colony[self.tstart]
        elif np.isscalar(self.tstart):
            out = t0 + self.tstart
        else:
            out = t0
        return self._add_delay(out, colony)
    def t2(self, colony):
        "return t2 relative to the var-colony"
        t0 = colony[self.time]
        if _vars.isvar(self.tend):
            out =  t0 + colony[self.tend]
        elif np.isscalar(self.tend):
            out =  t0 + self.tend
        else:
            if _vars.isvar(self.dur):
                return self.t1(colony) + colony[self.dur]
            elif np.isscalar(self.dur):
                return self.t1(colony) + self.dur
            else:
                raise NotImplementedError()
        return self._add_delay(out, colony)
    def dt(self, colony=None):
        "return t2-t1"
        if self.tend is None:
            if np.isscalar(self.dur):
                return self.dur
            elif _vars.isvar(self.dur):
                return colony[self.dur]
        else:
            if np.isscalar(self.tend):
                tend = self.tend
            else: # _vars.isvar(self.tend)
                tend = colony[self.tend]
            # subtract tstart
            if np.isscalar(self.tstart):
                return tend - self.tstart
            else:
                return tend - colony[self.tstart]
    def evt(self, colony):
        "return (t1, t2)"
        return self.t1(colony), self.t2(colony)
#        t0 = colony[self.time]
#        # t1
#        if _vars.isvar(self.tstart):
#            tstart = t0  + colony[self.tstart]
#        elif np.isscalar(self.tstart):
#            tstart = t0 + self.tstart
#        else:
#            tstart = t0
#        # t2
#        if _vars.isvar(self.tend):
#            tend = t0 + colony[self.tend]
#        elif np.isscalar(self.tend):
#            tend = t0 + self.tend
#        elif _vars.isvar(self.dur):
#            tend = tstart + colony[self.dur]
#        elif np.isscalar(self.dur):
#            tend = tstart + self.dur
#        else:
#            tend = tstart
#        return tstart, tend


class _collector(object):
    def __init__(self, address, segments, evts=None, var='magnitude',
                  
                 mask=None, segmask=None,
                 name = None,
                 cov=[],
                 
                 # time windows
                 tw=None, tstart=0., tend=None, dur=None,   # time window for data collection
                 delay=None, # shift the tw by a constant (can be Var)
                 bl=None, blstart=None, blend=None, bldur=None,  # time window for baseline
                 bldelay=None,
                 # which data
                 sensor=0, sensors=None,
                 
                 twfunc = np.mean, # function for aggregating timewindows
                 blfunc = None, # function for baseline; if None use twfunc
                 cfunc = np.mean, # collection functions
                 ):
        """
        **Main Parameters**
        
        address : Address 
            address defining cells for data collection 
            (all elements of the address must be provided by the segments in
            ``evts``)
        segments : dataset | list of segments
            segments container providing the source data
        evts : None | dataset | list of event segments 
            If the ``evts`` argument is provided, statistics are collected 
            relative to events in ``evts``. the segments in ``evts`` must be 
            aligned with the data segments in ``segments``, i.e. evts[i] must
            pertain to segments[i]. If evts is ``None``
            (default), data is collected relative to the ``segments``
            directly
        var : variable | str
            variable that is collected (Y); default is magnitude. Can 
            be str (in which case e.variables.get(var) is called. (TODO: unify 
            with sensor))
        mask : Address 
            can be used to restrict the collector to a subset of events
        segmask : Address
            can be used to restrict the collector to a subset
            of whole segments (faster than exclusion through a or mask)
        name : str
            name that is supplied for the collected variable; 
            default is 'Y' 
        cov : variable | list of variables
            variable or list of variables which should be collected as 
            factors/covariates along with the var
        
        
        **time window and baseline parameters**
        
        any time point argument can be a scalar, or a variable ``<var>`` 
        containe in the events, in which case 
        ``t = event['time'] + evt[<var>]``
        
        tstart : scalar
            start of the extraction time window
        tend : scalar
            end of the extraction time window
        tw : tuple(tstart, tend)
            as an alternative to tstart and tend, tw can be supplied as 
            (tstart, tend) tuple
        twfunc : callable
            function for aggregating data across the tw (default is 
            np.mean)
        delay : 
            delay that is added to the event time.
            Useful for example when events in evts indicate cues, but the 
            events of interest occurred at a variable delay after the cues. 
        blstart : 
            like tstart, but for baseline 
        blend :
            like tend, but for baseline 
        bl : 
            like tw, but for baseline
        blfunc : callable
            like twfunc; if None (default), twfunc is used
        bldelay :
            like delay, but for baseline
        cfunc : callable
            function for aggregating data over segments (whenever 
            several segments fall into the same a category). Default is 
            np.mean.
        
        """
        # variables
        e = segments[0].dataset.experiment
        self._time = e.variables.get('time')
        if isinstance(var, basestring):
            var = e.variables.get(var)
        
        if var in segments[0].varlist:
            self._var = var
        else:
            raise ValueError("{v} not in segments".format(v=var.__repr__()))
            # TODO: implement for uts-segments 

        # store primary attributes
        self._info = dict(
                          events = {},
                          events_skipped = {},
                          segments_skipped = [],
                          )
        
        self._warnings = []
        self._experiment = segments.experiment
        
        # store all kwargs
        self.set_segments(segments)
        self.set_evts(evts)
        self.set_address(address)
        self.set_cov(cov)
        self.set_mask(mask)
        self.set_segmask(segmask)
        self.set_name(name)
        self.set_tw_func(twfunc)
        self.set_bl_func(blfunc)
        self.set_collection_func(cfunc)
        
        if np.iterable(tw) and len(tw)==2:
            tstart, tend = tw
        self.set_tw(tstart, tend, dur, delay)

        if np.iterable(bl) and len(bl)==2:
            blstart, blend = bl
        self.set_bl(blstart, blend, bldur, bldelay)
        
        self._sensor = sensors
    
    def update(self, **kwargs):
        """
        Update collector keyword-arguments (see __init__ docstring) for 
        explanations).
        
        .. WARNING:: Some groups of keyword-arguments are coupled, i.e. when you
            update any of them you have to update the other ones too (or 
            defaults apply). This is the case for:
            
            * timewindow: [tstart, tend, dur, delay] 
            * baseline window: [blstart, blend, bldur, bldelay] 
        
        """
        if 'tstart' in kwargs:
            self.set_tw(kwargs.pop('tstart'),
                        tend = kwargs.pop('tend', None), 
                        dur = kwargs.pop('dur', None), 
                        delay = kwargs.pop('delay', None))
        if 'blstart' in kwargs:
            self.set_bl(kwargs.pop('blstart'),
                        tend = kwargs.pop('blend', None), 
                        dur = kwargs.pop('bldur', None), 
                        delay = kwargs.pop('bldelay', None))
        
        # reassign kwargs whose function name differs from 'set_%s' % argname
        for argname, funcname in [('twfunc', 'tw_func'),
                                  ('blfunc', 'bl_func'),
                                  ('cfunc', 'collection_func'),]:
            if argname in kwargs:
                kwargs[funcname] = kwargs.pop(argname)
        
        # call set_%s functions for kwargs
        for key, value in kwargs.iteritems():
            set_func_name = 'set_%s' % key
            if hasattr(self, set_func_name):
                set_func = getattr(self, set_func_name)
                set_func(value)
            else:
                raise KeyError("Invalid Argument: %r" % key)
    
    def set_segments(self, segments):
        if hasattr(segments, 'segments'):
            segments = segments.segments
        self._segments = segments 
    
    def set_evts(self, evts):
        if hasattr(evts, 'segments'):
            evts = evts.segments
        self._evts = evts
    
    def set_address(self, address):
        if isinstance(address, basestring):
            address = self._experiment.variables[address]
        
        if not _vars.isaddress(address):
            try:
                address = _vars.Address(address)
            except ValueError:
                raise ValueError("Need to provide valid address argument")    
        self._address = address
    
    def set_cov(self, covs):
        "list of variables"
        if covs:
            self._covs = _vars.asaddress(covs).keys()
        else:
            self._covs = []
    
    def set_mask(self, mask):
        self._mask = mask
    
    def set_segmask(self, segmask):
        self._segmask = segmask
    
    def set_name(self, name):
        if name is None:
            name = 'Y'
        self._name = name
    
    def set_tw(self, tstart, tend=None, dur=None, delay=None):
        self._tw = _timewindow(tstart=tstart, tend=tend, dur=dur, delay=delay)
    
    def set_bl(self, blstart=None, blend=None, bldur=None, bldelay=None):
        if np.isscalar(blstart):
            self._bl = _timewindow(tstart=blstart, tend=blend, dur=bldur, 
                                  delay=bldelay)
        else:
            self._bl = None
    
    def set_tw_func(self, twfunc):
        if not np.isscalar(twfunc([1,2,3])):
            raise ValueError("twfunc %r does not return scalar"%twfunc)
        else:
            self._twfunc = twfunc
    
    def set_bl_func(self, blfunc):
        if blfunc == None:
            blfunc = self._twfunc
        elif not np.isscalar(blfunc([1,2,3])):
            raise ValueError("blfunc %r does not return scalar"%blfunc)
        self._blfunc = blfunc
    
    def set_collection_func(self, cfunc):
        if not np.isscalar(cfunc([1,2,3])):
            raise ValueError("cfunc %r does not return scalar"%cfunc)
        self._cfunc = cfunc
        
#    Description functions---
    def __repr__(self):
        lead = self.__class__.__name__ + '('
        pre_len = len(lead)
        lead_empty = ' '*pre_len
        out = []
        for name, arg in self.kwargs.iteritems():
            out.append('%s%s: %r' % (lead, name, arg))
            lead = lead_empty
        return '\n'.join(out)
    
    def __str__(self):
        f_names = [f.name for f in self._factors.values() if _vsl.isfactor(f)]
        v_names = [f.name for f in self._factors.values() if _vsl.isvar(f)]
        out = 'Variables:\n' + ', '.join(sorted(v_names))
        out += '\nFactors:\n' + ', '.join(sorted(f_names))
        if hasattr(self, '_stats'):
            out += '\n\nSEGMENTS:\n' + ', '.join(f.name for f in self._stats.values())
        return out
    
    @property
    def kwargs(self):
        out = dict(a = self._address,
                   tw = self._tw,
                   twfunc = self._twfunc,
                   bl = self._bl,
                   blfunc = self._blfunc,
                   cfunc = self._cfunc,
                   segments = '...')
        return out
    
    def print_info(self):
        print "INFO\n----"
        for k, m in self._info.iteritems():
            print "{k}: {m}".format(k=k, m=m)
        if len(self._warnings) > 0:
            print "\nWARNINGS\n--------"
            for i, msg in enumerate(self._warnings):
                print "{msg}".format(i=i, msg=msg)
    
    """
    collection helper functions
    ---------------------------
    
    convoluted method structure that allows subclassing at several stages:
    
    self.get_dataframe
     -> self._collect_data
         -> self._iter_events
         -> self._data_for_seg_evt
     -> self._get_vessel_for_Y
    
    
    """
    def get_dataset(self):
        "get a dataframe containing Y and covariates"
        Y, covs = self._collect_data()
        
        indexes = Y.keys()
        Ydata = [Y[index] for index in indexes]
        Y = self._get_vessel_for_Y(Ydata)
        ds = _vsl.dataset(Y)

        # create _data objects
        for var, valdict in covs.iteritems():
            X = [valdict[index] for index in indexes]
            
            if var.dict_enabled:
                Y = _vsl.factor(X, name=var.name, random=var.random,
                                labels=var.dictionary, colors=var._color_dict)
            else:
                Y = _vsl.var(X, name=var.name)
            
            ds.add(Y)
        
        return ds
    
    def _collect_data(self):
        """
        returns the collected statistics as (Y, covYs) tuple, where:
        Y = {index: [list of data]}
        covYs = {cov: {index: value}} 
        
        iterate through self._iter_events_(). Calls subfunction to collect the
        dependent variable: self._data_for_seg_evt
        """
        # calls: y = self._data_for_seg_evt(seg, evt): to get data for each event
        
        # collect Y data
        # data = {index: [y1, y2, ...], ...}
        Y = {}
        covs = self._address.keys() + self._covs
        covYs = {c: {} for c in covs}
        
        for seg, evt in self._iter_events_():
            index = self._address.index(evt)
            if index:
                # Y
                y = self._data_for_seg_evt(seg, evt)
                Y.setdefault(index, []).append(y)
                # covs
                for cov in covs:
                    value = evt[cov]
                    if covYs[cov].setdefault(index, value) != value:
                        msg = ("Covariate %r has different values in events "
                               "from the same cell" % cov.name)
                        raise NotImplementedError(msg)
        
        return Y, covYs
    
    def _iter_events_(self):
        segments = self._segments
        evts = self._evts
#        info_evts = self._info['events']
#        info_evts_skip = self._info['events_skipped']
        info_segs_skip = self._info['segments_skipped']
        # progress
#        prog = ui.progress(i_max=len(evts),
#                           title = "Collecting...",
#                           message = "Collecting Data from %s segments" % len(evts))
        # iterator
        if evts is None:
            for segment in segments:
                if (not self._segmask) or self._segmask.contains(segment):
                    yield segment, segment
        else:
            for i, evt_seg in enumerate(evts):
    #            t0 = time.time()
                if (not self._segmask) or self._segmask.contains(evt_seg):
                    seg = segments[i]
                    for evt in evt_seg:
                        if (not self._mask) or self._mask.contains(evt):
                            yield seg, evt
                    
                    # info
                else:
                    info_segs_skip.append(evt_seg.name)





class TimewindowCollector(_collector):
    """
    Collector that extracts a summary statistic resulting in one value per 
    event.
    
    """
    def _data_for_seg_evt(self, seg, evt):
        # get data
        t1, t2 = self._tw.evt(evt)
        x = seg.subdata(var = self._var,
                        tstart = t1,
                        tend = t2,
                        out = 'data')   
        if len(x) == 0:
            msg = "Warning: Value in '{n}': {t1}-{t2} contains no event; set to 0"
            self._warnings.append(msg.format(n=seg.name, t1=t1, t2=t2))
            y = 0
        else:
            y = self._twfunc(x)

        # subtract baseline
        if self._bl:
            t1, t2 = self._bl.evt(evt)
            bl = seg.subdata(var = self._var,
                             tstart = t1,
                             tend = t2,
                             out = 'data')
            if len(bl) == 0:
                msg = "Warning: Baseline in '{n}': {t1}-{t2} contains no event; set to 0"
                self._warnings.append(msg.format(n=seg.name, t1=t1, t2=t2))
                y_bl = 0
            else:
                y_bl = self._blfunc(bl)
            y -= y_bl
        return y
    
    def _get_vessel_for_Y(self, Ydata):
        Ydata = [self._cfunc(data) for data in Ydata]
        Y = _vsl.var(Ydata, name=self._name)
        return Y


#class CollectorParasite(_vars.Parasite):
#    """
#    Stores its collector, which can be modified. The hosts are inferred from 
#    the collector's a argument.
#    """
#    def __init__(self, collector, name):
#        hosts = collector.get_address()
#        _vars.Parasite.__init__(self, hosts, name)
#        self.collector = collector


class TimeseriesCollector(_collector):
    def __init__(self, address, segments, evts=None,
                 sr=10, # samplingrate
                 mode = 'lin',
                 winfunc = np.blackman,
                 windur = 1,
                 mask = None,
                 **kwargs):
        """
        **Timeseries-Specific Arguments:**
        
        sr : scalar
            samplingrate
        mode : ``'lin'`` or ``'mw'``
            Determines how data is resampled from the segment: linear ('lin')
            or with a moving window ('mw'). The ``winfunc`` and ``windur`` 
            arguments are relevant only for 'mw'.
        winfunc : callable
            window function (numpy window function; default is ``np.blackman``)
        windur : scalar
            Window duration (in seconds)
        
        """
        self._samplingrate = sr
        
        self._mode = mode
        self._winfunc = winfunc
        self._windur = windur
        
        _collector.__init__(self, address, segments, evts=evts, **kwargs)
        
        # !!! removed "over" arg bc that should be = address now
        # TODO: adapt rest
        
    def _data_for_seg_evt(self, seg, evt):
        # get data
        tstart = self._tw.t1(evt)
        dur = self._tw.dt()
        #print evt
        #print tstart, dur
        y = seg.uts(var=self._var,
                    tstart = tstart,
                    dur = dur,
                    samplingrate = self._samplingrate,
                    uts = self._mode,
                    winfunc = self._winfunc,
                    windur = self._windur,
                    out = 'data')
        if self._bl:
            b = seg.uts(var=self._var,
                        tstart = self._bl.t1(evt),
                        dur = self._bl.dt(),
                        samplingrate = self._samplingrate,
                        uts = self._mode,
                        winfunc = self._winfunc,
                        windur = self._windur,
                        out = 'data')
            y -= self._blfunc(b)
        return y
    
    def _get_vessel_for_Y(self, Ydata):
        Ydata = np.array([self._cfunc(data, axis=0) for data in Ydata])
        T = np.arange(self._tw.tstart, self._tw.tend, 1 / self._samplingrate)
        time = _vsl.var(T, name='time')
        dims = ('case', time)
        info = {'samplingrate': self._samplingrate}
        Y = _vsl.ndvar(Ydata, dims, info=info, name=self._name)
        return Y




def timewindow(address, segments, evts=None, var='magnitude', **kwargs):
    """
    Collect statistics from data segments, either directly or through events
    associated with the data segments.
    
    """
    c = TimewindowCollector(address, segments, evts=evts, var=var, **kwargs)
    return c.get_dataset()

timewindow.__doc__ += TimewindowCollector.__init__.__doc__


def timeseries(address, segments, evts=None, var='magnitude', **kwargs):
    """
    Collect time-series statistics from segments, either directly or 
    through events associated with the data segments.
    
    """
    c = TimeseriesCollector(address, segments, evts=evts, var=var, **kwargs)
    return c.get_dataset()

timeseries.__doc__ += os.linesep.join((TimeseriesCollector.__init__.__doc__,
                                       TimewindowCollector.__init__.__doc__))

