"""
op.physio
=========

Operations for psychophysiology: heartbeat- and skin conductance response (SCR) 
extraction.

"""

import logging

import numpy as np
import scipy as sp
from matplotlib import mlab
from matplotlib import pyplot as P


from datasets_base import Derived_Event_Dataset
import datasets_mixins as mixins
import param


class HeartBeat(Derived_Event_Dataset, mixins.Reject):
    def _addparams_(self, p):
        p.minIBI = param.Param(default=.1, desc="minimum IBI (in sec)")
        p.maxIBI = param.Param(default=1.5, desc="maximum IBI (in sec)")
        p.threshold = param.Param(default=95, desc="Threshold for beat "
                                  "detection (only applies if advanced==alse)")
        p.advanced = param.Param(default=True, dtype=bool, 
                                 desc=" include minIBI etc. in calc")
    def _validate_input_properties_(self, properties):
        if properties['ndim'] != 1:
            raise ValueError("ibi needs ndim=1 input")
        else:
            return True
    def _create_varlist_(self):
        e = self.experiment
        varlist = [e.variables.get('time'),
                   e.variables.get('IBI'),
                   e.variables.get('Rate')]
        return varlist
    def _derive_segment_data_(self, segment, preview=False):
        source = segment._p_attr['source']
        samplingrate = self.properties['samplingrate']
#        for name in ['threshold', 'minIBI', 'maxIBI', 'advanced']:
#            locals()[name] = self.compiled[name]
        threshold = self.compiled['threshold']
        minIBI = self.compiled['minIBI']
        maxIBI = self.compiled['maxIBI']
        advanced = self.compiled['advanced']
        t = source.t
        
        # prepare data
        f0 = source.data[:,0]
        
        if True:#pretreatment == 1:
            f0 = np.hstack([[0], np.abs(np.diff(f0, n=2)), [0]])

        f0 = self.reject_flatten(segment, f0, t)
        
        threshold = sp.stats.scoreatpercentile(f0, threshold)
        
        ## beat extraction
        beats = [] # [t, IBI]
        f0_len = len(f0)
        if advanced:
            # in samples:
            minstep = int(minIBI * samplingrate)
            window = int(maxIBI * samplingrate)
            last = -minstep
            while last < f0_len - (minstep + window):
                # get window for possible new beats
                start = last + minstep 
                end = start + window
                dtemp = f0[start:end]
                # max inside window
                new = np.argmax(dtemp)
                if threshold:
                    # if minstep and threshold allow insert beat
                    while new > minstep:
                        # test: use percentil
#                        if np.max(dtemp[0:new-minstep]) > np.max(dtemp) * threshold:
                        if np.max(dtemp[0:new-minstep]) > threshold:
                            new = np.argmax(dtemp[0:new-minstep])
                        else:
                            break
                next = last + minstep + new
#                beats.append([t[next], t[next]-t[last]])
                beats.append(t[next])
                last = next
            beats = np.array(beats)
            # interpolate at places where exclusion is too long
            reject_list = self.reject_list_for_segment(segment)
            logging.debug("REJECT: %s"%reject_list)
            for t1, t2 in reject_list:
                if t2-t1 > minIBI:
#                    wrong = (beats > t1) * (beats < t2)
#                    if np.any(wrong):
#                        wrong = np.nonzero(wrong)[0]
#                        beats = np.hstack(beats[:wrong[0]], beats[wrong[-1]+1:])
                    # find time difference 
                    # remove beats inside rejection area
                    i1 = np.sum(beats < t1) - 1 # the last sample before t1
                    i2 = np.sum(beats < t2)     # the first sample after t2
                    ibi = beats[i2] - beats[i1]
                    ibi1 = beats[i1] - beats[i1-1]
                    ibi2 = beats[i2+1] - beats[i2]
                    ibi_mean = np.mean([ibi1, ibi2])
                    logging.debug("t=%s, ibi=%s, mean-ibi=%s"%(t1, ibi, ibi_mean))
                    if ibi > 1.5 * ibi_mean:
                        n = round(ibi / ibi_mean)
                        ibi_new = ibi / n
                        beats = np.hstack([beats[:i1+1], 
                                           [(beats[i1] + i*ibi_new) for i in range(1, n)], 
                                           beats[i2:]])
                    elif i2 != i1 + 1:
                        beats = np.hstack([beats[:i1+1], beats[i2:]])
                        i2 = i1+1
            # calculate ibi
            ibi = beats[1:] - beats[:-1]
            # correct the first beat's ibi
            ibi = np.hstack([[ibi[0]], ibi])
            # derive Rate from IBI
            rate = 60 / ibi
            beats = np.hstack([beats[:,None], ibi[:,None], rate[:,None]])
        else:
            raise NotImplementedError
#            threshold = data_in.max() * threshold
#            last = - minIBI
#            for i in arange(data_in.shape[0]):
#                if data_in[i] > threshold and i > last + minIBI:
#                    beats.append(i)
#                    last = i
        return beats



class SCR(Derived_Event_Dataset, mixins.Reject):
    def _addparams_(self, p):
        p.smooth1 = param.Window(default=(4, 150), #del_segs=True,
                                 desc="Smoothing before taking "
                                 "the first derivative")
        p.smooth2 = param.Window(default=(0, 150), #del_segs=True,
                                 desc="Smoothing of the first derivative "
                                 "before taking the second derivative")
        p.smooth3 = param.Window(default=(4, 150), #del_segs=True,
                                 desc="Smoothing of the second deri"
                                 "vative before finding 0-crossings.")
        p.threshold = param.Param(default=.005, #del_segs=True,
                                  desc="Threshold for SCRs (p-p)")
    def _create_varlist_(self):
        e = self.experiment
        varlist = [e.variables.get('time'),
                   e.variables.get('magnitude')]
        return varlist
    def _derive_segment_data_(self, segment, preview=False):
        # collect processing parameters
        source = segment._p_attr['source']
        sm1 = self.compiled['smooth1']
        sm2 = self.compiled['smooth2']
        sm3 = self.compiled['smooth3']
        threshold = self.compiled['threshold']
        t = source.t
        
        ## get skin conductance (f0)
        f0 = source.data[:,0]
        if sm1 != None:
            f0 = np.convolve(f0, sm1, 'same')
        
        # find the first derivative (discrete difference) 
        d1 = np.diff(f0)
        if sm2 != None:
            d1 = np.convolve(d1, sm2, 'same')
        
        # find zero crossings of the first derivative
        r_start = mlab.cross_from_below(d1, 0)
        r_end = mlab.cross_from_above(d1, 0)
        if r_end[0] < r_start[0]:
            r_end = r_end[1:]
            
        # remove SCRs in rejected TWs
        if self.reject_has_segment(segment):
            sr = float(segment.samplingrate)
            i = self.reject_check(segment, r_start, sr=sr)
            r_start = r_start[i]
            r_end = r_end[i]
            
            i = self.reject_check(segment, r_end, sr=sr)

        # find the second derivative
        d2 = np.diff(d1)
        if sm3 != None:
            d2 = np.convolve(d2, sm3, 'same')
        
        if preview:
            P.figure()
            P.plot(t, f0)
            nf = max(abs(f0))
            d1p = d1[1000:-1000]
            d2p = d2[1000:-1000]
            tp = t[1000:-1000]
            P.plot(tp, nf/2 + (d1p / max(abs(d1p))) * nf)
            P.plot(tp, nf/2 + (d2p / max(abs(d2p))) * nf)
            P.axhline(nf/2)
        
        # collect SCRs with positive p-p values
        SCRs = []
        for start, end in zip(r_start, r_end):
            # find intermediate separation points
            sep = mlab.cross_from_below(d2[start:end], 0) + start + 1
            
            # remove SCRs in rejected TWs
            if self.reject_has_segment(segment):
                i = self.reject_check(segment, sep, sr=sr, tstart=start/sr)
                sep = sep[i]
            
            i = start
            for j in np.hstack((sep, [end])):
                pp = np.sum(d1[i:j])
                if pp > threshold:
                    SCRs.append([t[i], pp])
                i = j
        if preview: # provide preview data to the viewer
            raise NotImplementedError
        else:
            return np.array(SCRs)

