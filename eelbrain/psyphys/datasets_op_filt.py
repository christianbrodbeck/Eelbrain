import logging

import numpy as np
import scipy as sp


from datasets_base import Derived_UTS_Dataset
import param


__all__ = ['filt', 'firwin']



class filt(Derived_UTS_Dataset):
    def _addparams_(self, p):
        p.filter = param.Choice(options=['Elliptic', 'Chebyshev', 
                                         'Butterworth'], default=1)
        p.w_pass = param.Param(default=25, desc="""w_pass & w_stop:
Passband and stopband edge frequencies, in Hz. For example:
    Lowpass:   w_pass = 20,          w_stop = 30
    Highpass:  w_pass = 30,          w_stop = 20
    Bandpass:  w_pass = [20, 50],    w_stop = [10, 60]
    Bandstop:  w_pass = [10, 60],    w_stop = [20, 50]""")
        p.w_stop = param.Param(default=35, desc="see w_pass desc") 
        p.g_pass = param.Param(default=1, desc="The maximum loss in the "
                               "passband (dB).")
        p.g_stop = param.Param(default=50., desc="The minimum attenuation "
                               "in the stopband (dB).")
    def _validate_input_properties_(self, properties):
        if properties['data_type'] == 'event':
            raise ValueError("filter needs uts input")
        else:
            return True
    def _create_compiled_(self):
        samplingrate = self.parent.samplingrate
        # Assemble filter; see 
        # http://www.scipy.org/doc/api_docs/SciPy.signal.filter_design.html
        # wp and ws: normalized from 0 to 1 (1 corresponds to pi radians / sample
        
        name = self.p.filter.get_string()
        Nyq = samplingrate / 2.
        
        wp = self.p.w_pass.get()
        if np.isscalar(wp):
            wp /= Nyq
        else:
            wp = [x / Nyq for x in wp]
        
        ws = self.p.w_stop.get()
        if np.isscalar(ws):
            ws /= Nyq
        else:
            ws = [x / Nyq for x in ws]
        
        gpass = self.p.g_pass.get()
        gstop = self.p.g_stop.get()
        
#        print "wp: {0}, ws: {1}, gpass: {2}, gstop: {3}".format(wp, ws, gpass, gstop)
        try:
            # btype
            # FIXME: what about bandpass??
            if wp < ws:
                btype = 'lowpass'
            elif wp > ws:
                btype = 'highpass'
            else:
                raise ValueError("Invalid Filter Spec")
            #create filter
            if name == 'Butterworth':
                ord, Wn = sp.signal.filter_design.buttord(wp, ws, gpass, gstop)
                B, A = sp.signal.filter_design.butter(ord, Wn, btype=btype)
            elif name == 'Chebyshev':
                ord, Wn = sp.signal.filter_design.cheb2ord(wp, ws, gpass, gstop)
                B, A = sp.signal.filter_design.cheby2(ord, 1, Wn, btype=btype)
            elif name == 'Elliptic':
                ord, Wn = sp.signal.filter_design.ellipord(wp, ws, gpass, gstop)
                B, A = sp.signal.filter_design.ellip(ord, 0, 0, Wn, btype=btype)
            else:
                raise ValueError("Unknown Filter (%s) Specified" % name)
        except Exception, exc:
            msg = ["Filter could not be created; scipy Error:", str(Exception),
                   str(exc)]
            print '\n'.join(msg)
            B, A = sp.signal.filter_design.butter(4, 1.5, btype='low')
        # store stuff
        return {'A': A, 'B': B}
    def _derive_segment_data_(self, segment):
        # get stuff
        source = segment._p_attr['source']
        source_data = source.data
        A = self.compiled['A']
        B = self.compiled['B']
        # filter
        out = sp.signal.lfilter(B, A, source_data, axis=0)
        return out



class firwin(Derived_UTS_Dataset):
    """
    Simple filtering based on:
    
    http://www.scipy.org/Cookbook/FIRFilter
    http://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html
    
    """
    def _validate_input_properties_(self, properties):
        if properties['data_type'] == 'event':
            raise ValueError("need uts input")
        else:
            return True
    
    def _addparams_(self, p):
        p.cutoff = param.Param(default=1.0, desc="in Hertz")
        p.numtaps = param.Param(default=1001, desc="numtaps")
        p.pass_zero = param.Param(default=True, desc="(see scipy.signal.firwin"
                                  " documentation)")
    
    def _derive_segment_data_(self, segment, preview=False):
        # collect processing parameters
        source = segment._p_attr['source']
        N = self.compiled['numtaps']
        cutoff_Hz = self.compiled['cutoff']
        pass_zero = self.compiled['pass_zero']
        
        # Cutoff frequency of filter: expressed in the same units as nyq 
        nyq_rate = source.samplingrate / 2.
        cutoff = cutoff_Hz / nyq_rate
        
        taps = sp.signal.firwin(N, cutoff, pass_zero=pass_zero)
        
        data = sp.signal.lfilter(taps, 1.0, source.data, axis=0)
        
        # compensate for the phase delay of the filtered signal.
        delay = int(0.5 * (N - 1))
        data[:-delay] = data[delay:]
        data[-delay:] = data[-1]
        return data

       