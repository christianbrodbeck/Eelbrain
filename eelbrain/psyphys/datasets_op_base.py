
from __future__ import division

from copy import deepcopy
#import logging

import numpy as np

import param
from datasets_base import Derived_UTS_Dataset


class Extract_Channel(Derived_UTS_Dataset):
    """
    Extract a subset of the channels from the source.
    
    """
    def _addparams_(self, p):
        p.index = param.Param(desc="Index of channels to extract, can be scalar, "
                              "tuple, or slice")
    def _derive_segment_properties_(self, segment):
        parent = segment._p_attr['source']
        properties = parent.properties
        # predict new shape
        n,i = parent.shape
        index = self.p.index.get()
        if isinstance(index, slice):
            start, stop, step = index.start, index.stop, index.step
            if start is None:
                start = 0
            if stop is None:
                stop = i+1
            i = (stop - start)
            if step is not None:
                rest = i % step
                i = int((i - rest) / step)
        elif np.isscalar(index):
            i = 1
        else:
            i = len(index)
        properties['shape'] = (n,i)
        return properties
    def _derive_segment_data_(self, segment):
        parent = segment._p_attr['source']
        index = self.p.index.get()
        if np.isscalar(index):
            index = (index,)
        return parent.data[:, index]
        


class crop_uts(Derived_UTS_Dataset):
    """
    Crop data segments. 
    
    A gui is available for this dataset. Use ``gui(dataset)`` to start it.
    
    """
    def _addparams_(self, p):
        p.tstart = param.PerParentSegment(default=None, can_be_None=True,
                                          desc='Start of extraction')
        p.tend = param.PerParentSegment(default=None, can_be_None=True,
                                        desc='End of extraction')
    def _create_properties_(self):
        p = Derived_UTS_Dataset._create_properties_(self)
        
        # TODO: shape
#        sr = self.parent.samplingrate
#        p['segmented'] = True
#        # shape
#        old_shape = p['shape']
#        length = self.p.tend.in_samples(sr) - self.p.tstart.in_samples(sr)
#        new_shape = (length, ) + old_shape[1:]
#        p['shape'] = new_shape
        
        return p
    def _derive_segment_data_(self, segment):
        id = segment._id
        tstart = self.p.tstart[id]
        tend = self.p.tend[id]
        source = segment._p_attr['source']
        data = source.subdata(tstart=tstart, tend=tend, out='data')
        
        segment.properties['shape'] = data.shape
        return data
    def _derive_segment_properties_(self, segment):
        parent = segment._p_attr['source']
        properties = deepcopy(parent.properties)
        
        id = parent._id
        tstart = self.p.tstart[id]
        tend = self.p.tend[id]
        
        if (tstart is not None) or (tend is not None):
            shape = parent['shape']
            length = shape[0]
            if tend:
                length = parent.i_for_t(tend) 
            if tstart:
                cut = parent.i_for_t(tstart)
                length -= cut
            properties['shape'] = (length,) + shape[1:]
        return properties




class segmentation(Derived_UTS_Dataset):
    def __init__(self, data, events, name="segmentation"):
        assert events['data_type'] == 'event'
        Derived_UTS_Dataset.__init__(self, data, name=name)
        self.parent_events = events
        events.children.append(self)
    def _validate_input_properties_(self, properties):
        assert properties['data_type'] != 'event'
        return True
    def _addparams_(self, p):
        self.p.segment_address = param.Param(del_segs=True, desc="Address by "
                                             "which segments are matched")
        self.p.event_address = param.Param(default=None, del_segs=True, 
                                           can_be_None=True, desc="Address by "
                                           "which to filter events")
        self.p.pre = param.Time(default=.5, desc="Segment start before event "
                                "start.")
        self.p.post = param.Time(default=2.5, desc="Segment end after event "
                                 "start.")
        self.p.padding = param.Choice(options=['constant', 'zero', 'raise'],
                                      desc="values to substitute outside segment")
    def _create_compiled_(self):
        sr = self.parent.samplingrate
        
        c = dict(pre = self.p.pre.in_samples(sr),
                 post= self.p.post.in_samples(sr),
                 timevar = self.experiment.variables.get('time'),
                 )

        segment_address = self.p.segment_address.get(),
        assert segment_address != None
        c['segment_address'] = segment_address
        c['event_address'] = self.p.event_address.get()

        # padding 
        pt = self.p.padding.get_string()
        if pt == 'constant':
            c['padding'] = True
        elif pt == 'zero':
            c['padding'] = 0
        elif pt == 'raise':
            c['padding'] = False
        
        return c
    def _create_properties_(self):
        p = Derived_UTS_Dataset._create_properties_(self)
        c = self.compiled
        sr = self.parent.samplingrate

        p['t0'] = self.p.pre.in_seconds(sr)
        # shape
        p['segmented'] = True
        old_shape = p['shape']
        length = c['pre'] + c['post']
        new_shape = (length, ) + old_shape[1:]
        p['shape'] = new_shape
        
        return p
    def _create_segments_(self):
        c=self.compiled
        # get args
        address = c['segment_address']
        e_filter = c['event_address']
        pre = c['pre']
        post = c['post']
        padding = c['padding']
        time = c['timevar']
        
#        len_err_txt = "Segmentation: Data source too short by {0} samples ({1})"
        new_name = "{n} @ {t:.3f}" 
        
        segs_data = address.dict(self.parent.segments)
        segs_events = address.dict(self.parent_events.segments)
        
#        logging.debug(str(segs_data))
#        logging.debug(str(segs_events))
        for index, dataseg_list in segs_data.iteritems():
            if len(dataseg_list) > 1:
                raise KeyError("segment_address does not uniquely specify data segments")
            dataseg = dataseg_list[0]
            #source_data = dataseg.data
            #properties = deepcopy(dataseg.properties)
            for e_seg in segs_events[index]:
                #logging.debug(" eseg")
                for event in e_seg:
                    if (not e_filter) or (e_filter.isin(event)):
                        t = event[time]
                        
                        kwargs = {'tstart': t - pre, 
                                  'tend': t + post,
                                  't0': pre,
                                  'padding': padding
                                  }
                        p_attr = {'kwargs': kwargs,
                                  'source_seg': dataseg,
                                  }
                        
                        seg = self.Child_Segment(self, dataseg.properties, p_attr,
                                                 name=new_name.format(n=dataseg.name, t=t),
                                                 varsource=dataseg.variables, 
                                                 )
                        for var, val in event.iteritems():
                            seg[var] = val
    def _derive_segment_data_(self, segment):
        kwargs = segment._p_attr['kwargs']
        source = segment._p_attr['source_seg']
        return source.subdata(out='data', **kwargs)
