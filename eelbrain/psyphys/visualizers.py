'''
Created on Mar 27, 2010

@author: christian


Visualizer (threading.Thread) 
 - hold data
    - thread updates data -> lock data
 - provide appropriate toax function
    - update is blocked while data is being used
    
    
    - for events: deepcopy event data
    - for data: deepcopy data
    
    
 
View (thread)
 - manages view details
 - calls Visualizer.plot(...)
'''

import logging

import numpy as np

import vars as _vars
import operations as _op


def default(dataset, dur=None, mag=None, a=None):
    """
    returns the standard viewer for the given dataset
    
    kwargs
    ------
    a: address (to select subset of segments in the dataset)
    
    """
    t = type(dataset)
    var_mag = dataset.experiment.variables.get('magnitude')
    var_dur = dataset.experiment.variables.get('duration')
    
    if a is None:
        sds = dataset
    else:
        sds = dataset[a]
    
    if t == _op.physio.HeartBeat:
        var_IBI = dataset.experiment.variables.get('IBI')
#        v1 = V.Physio(dataset.parent)
        new_v = E_Line(sds, var_IBI)
    elif t == _op.physio.SCR:
        new_v = Events(sds, mag=var_mag, dur=.1)
    elif dataset['data_type'] == 'uts':
        new_v = Physio(sds)
    elif dataset['data_type'] == 'event':
        kwargs = dict(dur=dur, mag=mag)
        if not mag and var_mag in dataset.varlist:
            kwargs['mag'] = var_mag
        if not dur:
            if var_dur in dataset.varlist:
                kwargs['dur'] = var_dur
            else:
                kwargs['dur'] = .1
        new_v = Events(sds, **kwargs)
    else:
        raise ValueError("No Visualizer for data_type=%s"%dataset['data_type'])
    return new_v


class _Visualizer(object):
    '''
    from outside:
     - set_segment(i)    - pick the segment with index i from the dataset
     - toax(ax, t1, t2)  - plot
    
    keeps info
    - link to dataset
    - caches data
    - provides toax(ax, t1, t2) function
    
    subclasses should provide the two functions below the <# SUBCLASS> marker
    '''
    def __init__(self, dataset, aa=False, color=None, i=0):
        '''
        Constructor
        
        '''
        self.dataset = dataset
        self.set_aa(aa)
        self.set_color(color)
        self.set_segment(i)
    def __repr__(self):
        temp = "{c}({d})"
        return temp.format(c = self.__class__.__name__,
                           d = repr(self.dataset)) 
    def set_color(self, color):
        """
        None, color, or variable with colors
        """
        self._color = color
    def set_aa(self, aa):
        "Set Antialiasing (True|False)"
        self._aa = bool(aa)
    def set_segment(self, i):
        assert i < len(self.dataset), "invalid index %s"%i
        segment = self._seg = self.dataset[i]
        self.segment_name = segment.name
        self.segment_id = segment._id
        self.get_data_from_segment(segment)
    @property
    def N(self):
        return len(self.dataset)
    # SUBCLASS -- these functions are called by the 
    def get_data_from_segment(self, segment):
        logging.debug("%r Visualizer get data" % self.__class__.__name__)
        self.tstart = segment.tstart
        self.tend = segment.tend
    def toax(self, ax, tstart, tend):
        pass


class Physio(_Visualizer):
    def get_data_from_segment(self, segment):
        _Visualizer.get_data_from_segment(self, segment)
#        self.color = segment['color']
        self.data = segment.data
        self.t = segment.t
        self.samplingrate = segment['samplingrate']
    def toax(self, ax, t0, t1, rc=True):
        logging.debug(" Physio Vis toax(%s, %s), sr=%s"%(t0, t1, self.samplingrate))
        # index from t0 and t1
        if not np.isscalar(t0):
            t0 = self.t[0]
        if not np.isscalar(t1):
            t1 = self.t[-1] + 1
        i = (self.t >= t0) * (self.t < t1)
        
        # data
        t = self.t[i]
        data = self.data[i,:]
        if rc:
            d_min = np.min(data, 0) 
            data = data - d_min
            d_max = np.max(data, 0)
            data /= d_max
            n_chan = data.shape[1]
            data -= np.arange(n_chan)
        
        # color
        if self._color is None:
            color = self._seg.color
        else:
            color = self._color
        if color is None:
            color = 'blue'

        # plot
        ax.plot(t, data, color=color, aa=self._aa)#, zorder=10-i)
#        ax.set_xlim(t0, t1)


class Events(_Visualizer):
    def __init__(self, dataset, mag=None, dur=None, color=None, i=0, address=None):
        """
        kwargs
        ------
        mag: variable coding the magnitude (displayed as block height). By
                default, all blocks will have the same height. 
        dur: shown event duration, variable or scalar
        color: event color (variable)
        i: index of the initial segment 
        address: address indicating the subset of events to draw. By default,
                all events are drawn.

        """
        self._var_time = dataset.experiment.variables.get('time')
        
        if _vars.isvar(mag):
            if id(mag) in [id(v) for v in dataset[0].varlist]:
                self._var_magnitude = mag
            else:
                raise KeyError("segment does not contain magnitude var")
        else:
            self._var_magnitude = None
        if np.isscalar(mag):
            self.magnitude = mag
        else:
            self.magnitude = 1
        
        if dur is None:
            self._var_duration = dataset.experiment.variables.get('duration')
            self.duration = .5
        elif _vars.isvar(dur):
            self._var_duration = dur
            self.duration = .5
        elif np.isscalar(dur):
            self._var_duration = None
            self.duration = dur
        else:
            raise ValueError("duration must be either scalar or variable, "
                             "not %r"%type(dur))
        
        self.set_evt_address(address)
        
        _Visualizer.__init__(self, dataset, color=color, i=i)
        
    def set_evt_address(self, address):
        "Change the subset of events being drawn (Address, slice or None)"
        if _vars.isaddress(address) or isinstance(address, slice):
            self._evt_address = address
        else:
            self._evt_address = None
        
    def get_data_from_segment(self, segment):
        if self._evt_address is not None:
            segment = segment[self._evt_address]
        _Visualizer.get_data_from_segment(self, segment)
    def toax(self, ax, t0, t1, rc=True, dur=.5):
        """

        """
#        print "TOAX: %s -- %s"%(t0, t1)
        # index from t0 and t1
        
        # find standard color and color_var
        color = self._seg.color
        if color is None:
            color = (.1, .1, .9)
        
        if _vars.isvar(self._color):
            color_var = self._color
        else:
            color_var = None
            if self._color is not None:
                color = self._color
        
        
        time = self._var_time
        kwargs = {'alpha':.4}

        # prepare scaling factor
        if rc and self._var_magnitude:
            scaling = max(self._seg.subdata(var=self._var_magnitude))
        else:
            scaling = 1

#        print "TOAX FOR start"
        # loop events
        for evt in self._seg.sub_iter(time, t0, t1):
#            print 'evt'
            #time
            e_start = evt[time]
#            if self._var_duration in evt:
            try:
                e_end = e_start + evt[self._var_duration]
            except:
                e_end = e_start + self.duration
            
            # color
            if color_var is None:
                kwargs['color'] = color
            else:
                c = self._color.get_color_for_colony(evt)
                if c is None:
                    kwargs['color'] = color
                else:
                    kwargs['color'] = c
            
            # height
#            if self._var_magnitude and self._var_magnitude in evt:
            try:
                y = evt[self._var_magnitude] / scaling
            except:
                y = self.magnitude

            span = ax.axvspan(e_start, e_end, ymax=y, **kwargs)
#            if labels:
#                ax.annotate(labels[data[labels]], (start, y/2.))
#        print "TOAX FOR end"

        

class E_Line(Events):
    def get_data_from_segment(self, segment, rc=True):
        Events.get_data_from_segment(self, segment)
        
        self.t = segment.t
        self.data = segment.subdata(var=self._var_magnitude, out='data')
        self.color = segment['color']
    def toax(self, ax, t0=None, t1=None, rc=True):
        if (t0 != None) or (t1 != None):
            if t0 == None:
                t0 = self.tstart
            if t1 == None:
                t1 = self.tend+.0001
            indexes = (self.t > t0) * (self.t < t1)
            
            t = self.t[indexes]
            data = self.data[indexes]
        else:
            t = self.t
            data = self.data
        
        if rc:
#            print "E_Line", data.shape
            data = data - min(data)
            data = data / max(data)
            
        ax.plot(t, data, color=self.color)

