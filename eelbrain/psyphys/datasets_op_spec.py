"""
Datasets are initialized with a parent dataset and can get and modify data form 
that parent


Requiremens:
 d.segments: returns list of segments
 d.ndim:     returns number of dimensions; 
             description if not equally spaced (e.g. 'wavelet')
"""
from __future__ import division

import logging

import numpy as np
import scipy as sp

import param
from datasets_base import Derived_UTS_Dataset


class spectrum(Derived_UTS_Dataset):
#    type_name = "Moving Window Fourier Transform"
    data_type = 'spec'
    def _addparams_(self, p):
        p.window = param.Param(default=np.hanning, #dtype=str,
                               desc="Window function (function which "
                               "takes length as argument and returns "
                               "window, such as numpy window functions.")
        p.window_length = param.Time(default=.2, can_be_var=False,
                                     desc="Window length")
        p.offset = param.Time(default=.1,
                              desc="distance of consecutive DFT windows")
        p.frequencies = param.Param(default=np.arange(1,100),
                                    desc="array of frequencies, specified "
                                    "as division of window length")
    def _validate_input_properties_(self, properties):
        if properties['ndim'] != 1:
            raise ValueError("fft nees ndim=1 input")
        else:
            return True
    def _create_compiled_(self):
        c = {}
        samplingrate = self.parent.samplingrate

        n = self.p.window_length.in_samples(samplingrate)
        window_func = self.p.window.get()
        window = window_func(n)[:,None]
        window /= window.sum()
        c['window'] = window
        
        F = np.array(self.p.frequencies.get())
        assert np.all(F > 0)
        c['F'] = F
        c['n'] = n
        c['offset'] = self.p.offset.in_samples(samplingrate)
        
        return c
    def _create_properties_(self):
        p = Derived_UTS_Dataset._create_properties_(self)
        c = self.compiled
        F = c['F']
        n = c['n']

        p['data_type'] = 'spec'
        
        samplingrate = self.parent.samplingrate
        offset_insamples = c['offset']
        p['samplingrate'] = samplingrate / offset_insamples
        
        p['t0'] = p['t0'] - (self.p.window_length.in_seconds(samplingrate) / 2)
        
        # data shape
        p['ndim'] = 2
        shape = p['shape'][:]
        if shape[0] == None:
            length = None
        else:
            length = (shape[0] - n) // offset_insamples
        p['shape'] = (length, ) + shape[1:] + F.shape
        
        freq = F * (samplingrate / n)
        p['frequencies'] = freq
        
#        info = ["{n} (d{s}) _compile:".format(n=self.name, s=self.id),
#                "  > _window {w}, {s}".format(w=self.p.window.get(), 
#                                              s=window.shape)]
#        logging.info('\n'.join(info))
        return p
    def _derive_segment_data_(self, segment, preview=False):
        c = self.compiled
        source = segment._p_attr['source']
        data = source.data
        # get params
        F = c['F']
        n = c['n']
        offset = c['offset']
        window = c['window']
        # prepare
        Tc = np.linspace(0, 2*np.pi, n)#[:,None]
        c = np.vstack([np.sin(Tc*f) + np.cos(Tc*f)*1j for f in F])
        tmax = data.shape[0]
        # do
        Tstart = np.arange(0, tmax - n, offset)
        out = []
        for t in Tstart:
            x = data[t:t+n]
            y = x * window
            out.append(abs(np.dot(c, y)).T)
        out = np.array(out)
        segment['shape'] = out.shape
        return out


#### OLD SHIT ######## OLD SHIT ######## OLD SHIT ######## OLD SHIT ######## OLD


'''

class Average(DerivedDataset):
    """contains special export method that defaults to self.address"""
    typeName    = "Average"
    nameExtension = '.av'
    parameters = []
    def setup(self):
        print """***
    Setup UI is not available for Averaging. Use the Address class
    to construct an address over which to average. Example:    
    >>> address = Address({subjectVar:None, conditionVar:None})
    >>> d.setAddress( address )"""
    def __init__(self, *args, **kwargs):
        self.address = None
        print "* must specify address through d.setAddress(address)"
        DerivedDataset.__init__(self, *args, **kwargs)
    def setAddress(self, address):
        if type(address) != Address:
            raise TypeError()
        self.address = address
        self.segments = self.getSegmentsFromParent()
        print "-> got %s segments"%len(self.segments)
    def getSegmentsFromParent(self):
        if type(self.address) != Address:
            print "!!! must specify %s address"%self.name
            return []
        source = self.address.dict( self.parent.segments )
        dictKey = self.address.keys()
        segments = []
        for index in source:
            new = DerivedSegment( source[index], self, onetoone=False )
            new.variables = self.variables.getNewColony()
            for k,v in zip(dictKey, index):
                new.variables[k]=v
            segments.append( new )
        return segments
    def _p(self, segment):
        data = [ s.data[newaxis] for s in segment.source ]
        data = vstack( data ).mean(0)
        self.addCache_toSegment_(data, segment)
        return data
    def export(self, address=None, **kwargs):
        if address==None:
            address=self.address
        DerivedDataset.export(self, address, **kwargs)




class Downsample(DerivedDataset):
    typeName   = "Downsample"
    setupText  = "Use None "
    nameExtension='.downsampled'
    parameters = [  Param( name="samplingrate",
                                        desc="New Samplingrate in Hz",
                                        default=100,
                                        type_func=float ),
                    Param( name="baseline",
                                        desc="New baseline in s; tuple. Use None for open end.",
                                        default=None,
                                        type_func=tuple ),
                    Param( name="interval",
                                        desc="Get sub-section of the data;\
                                              tuple in seconds, Use None for open end.",
                                        default=None ,
                                        type_func=tuple )   ]
    #def __init__(self, parent):
    #    print 'init'
    #    DerivedDataset.__init__(self, parent)
    @property
    def samplingrate(self):
        return self._p_samplingrate
    @property
    def segmentLength(self):
        if self.parent.segmentLength == None:
            return None
        else:
            return self.parent.segmentLength * self.samplingrate / self.parent.samplingrate
    def _p(self, segment):
        data = segment.source.subdata(  interval=None, 
                                        baseline=self._p_baseline, 
                                        samplingrate=self._p_samplingrate, 
                                        out='data' )
        self.addCache_toSegment_(data, segment)
        return data
    def _previewToFig(self, fig, i=0, sensors=[0]):
        ax = fig.add_subplot(211)
        ax.plot( self.segments[i].source.subdata(sensors=sensors, out='data') )
        ax = fig.add_subplot(212)
        ax.plot( self.segments[i].subdata(sensors=sensors, out='data') )


class Lowbutter(DerivedDataset):
    typeName="Low-Pass Butterworth"
    nameExtension='.filtered'
    parameters = [  Param( name="lpcf",
                                        desc="lowpass corner freq",
                                        default=8.5 ),
                    Param( name="lpsf",
                                        desc="lowpass stop freq",
                                        default=7.5 ),
                    Param( name="gpass",
                                        desc="corner freq attenuation",
                                        default=3 ),
                    Param( name="gstop",
                                        desc="stop freq attenuation",
                                        default=15 ),
                    Param( name="filtfilt",
                                        desc="avoid phase shift (slow!)",
                                        default=False ) ]
    def _p(self, segment):
        """ FROM PBRAIN UTILS """
        Nyq = self.samplingrate/2.
        wp = self._p_lpcf/Nyq
        ws = self._p_lpsf/Nyq
        
        ord, Wn = signal.buttord(wp, ws, self._p_gpass, self._p_gstop)
        b, a = signal.butter(ord, Wn, btype='lowpass')
        #filter
        data = segment.source.data
        if self._p_filtfilt:
            data = hstack(( [ signal.filtfilt(b,a,data[:,i])[newaxis].T for i in range(data.shape[1]) ] ))
        else:
            data = signal.lfilter(b,a, data , axis=0)

        self.addCache_toSegment_(data, segment)
        return data
    def _previewToFig(self, fig, i=0, sensors=[0]):
        # get data
        rawData = self.segments[i].source.data[:,sensors]
        filtData= self.segments[i].data[:, sensors]
        # plot
        fig.suptitle("Butterworth")
        ax=fig.add_subplot(311); ax.set_title("Input")
        ax.plot(rawData)
        ax=fig.add_subplot(312); ax.set_title("Output")
        ax.plot(filtData)
        # freq resp
        ax=fig.add_subplot(325); ax.set_title("Frequency Response")
        w, h = signal.freqz(b,a) # FIXME: freqz?
        ax.plot(h)
        ax=fig.add_subplot(326); ax.set_title("Filter")
        ax.plot(a)
        ax.plot(b)



class Wavelet(DerivedDataset):
    typeName = "Wavelet Decomposition"
    desiredNdim= [1]
    #canProcessMultipleSensors=True
    parameters = [  Param( name="frequencies",
                                        desc = "npy array",
                                        default = np.r_[1.5:5:15j]**np.e), #r_[1.5:3.5:15j]**e
                    Param( name="cycles",
                                        desc="number of times the sine/cosine repeats in the wavelet",
                                        default=3,
                                        type_func=int ),
                    Param( name="edges",
                                        desc='how to treat edges: 0:empty; 1:fill half',
                                        default=1,
                                        type_func=int),
                    Param( name="window",
                                        desc='Window with which the Wavelet will be multiplied (None=flat)',
                                        default="hanning"),
                ]
    def validateParentDataset(self, parent):
        assert parent.ndim == 1
        assert parent.segmented
    @property
    def customProperties(self):
        self.getParams()
        return {'frequencies':self.frequencies, # list of frequencies that are computed
                'N':self.N, # list of n of sample points per ferquency
                'F':self.F, # frequencies but each frequency times N
                'S':self.S, # list of arrays indicating sample# to start wavelet 
                'T':self.T, # list of arrays indicating t of samples (= middle of wavelet) in s
               } 
    @property
    def ndim(self):
        return "wavelet"
    @property
    def frequencies(self):
        return self._p_frequencies
    @property
    def F(self):
        return hstack([ [f]*n for f,n in zip(self._p_frequencies, self.N) ])
    def default(self):
        DerivedDataset.default(self)
        self.getParams()
    def invalidateCache(self):
        DerivedDataset.invalidateCache(self)
        self.getParams()
    def getParams(self):
        samplingrate = self.samplingrate
        length_samples = self.parent.segmentLength
        length_seconds = float(length_samples) / samplingrate
        assert length_seconds * min(self._p_frequencies) >= self._p_cycles+1
        # get N, S
        N=[]    # list of n-samplepoints per frequency
        S=[]    # list of arrays indicating sample# to start wavelet 
        T=[]    # --> T: #list of arrays indicating t of samples (= middle of wavelet) in s
        for f in self._p_frequencies:
            windowLength_samples = len(Wavelets.psi(f, samplingrate, self._p_cycles)[0])
            windowLength_sec = float(windowLength_samples) / samplingrate
            tstep_sec = 1./f
            if self._p_edges==0:
                st = r_[0 : length_seconds-windowLength_sec : tstep_sec]
            else:
                padding = self._p_cycles / 2 * tstep_sec
                st = r_[-padding : length_seconds-windowLength_sec+padding : tstep_sec]
            S.append( (st*samplingrate).astype(int) )
            T.append( st + windowLength_sec/2. )
            N.append( len(st) )
        self.N = N
        self.S = S
        self.T = T
    def _p(self, segment):
        data_in = segment.source.data
        data = Wavelets.dec2d(  data_in, 
                                self._p_frequencies, 
                                self.S, 
                                self.samplingrate, 
                                self._p_cycles, 
                                self._p_edges,
                                window = self._p_window,
                              )
        self.addCache_toSegment_(data, segment)
        return data
    def toParam_value_(self, param, value):
        if param.name=='edges':
            if value != '0':
                value='1'
        if param.name=='window':
            try:
                eval('Wavelets.'+value)
                self._p_window=value
            except exception, exc:
                print "faild."
        else:
            DerivedDataset.toParam_value_(self, param, value)
    def _previewToFig(self, fig, i=0, sensor=0):
        seg = self.segments[i]
        ax = fig.add_subplot(3,2,1)
        windowFunc = eval('Wavelets.'+self._p_window)
        for wav, c in zip(Wavelets.psi(2, 200, self._p_cycles, windowFunc=windowFunc), ['b','r']):
            ax.plot(wav, c=c)
        # TODO: sin testSignal
        ax = fig.add_subplot(3,1,2)
        ax.plot( seg.source.data[:,sensor] )
        ax = fig.add_subplot(3,1,3)
        minT=0; maxT=self.segmentLength #= -self.t0; maxT=self.duration
        minF=min(self.frequencies); maxF=max(self.frequencies)
        eegplot.ax_wavelet(seg, sensor=sensor, extent = ( minT, maxT, minF, maxF ))











class ICA(DerivedDataset):
    """The following settings must be made before data can be used:
    - setup():   select the variable that is used to sort segments
    - selectComponents():   extract the ICs and select which to retain
    """

    typeName = "ICA"
    desiredNdim = [1]
    def __init__(self, *args, **kwargs):
        DerivedDataset.__init__(self, *args, **kwargs)
        self.setup()
        self.trainedNodes = {}          # { v:icaNode, ... } 
        self.removeComponents = {}      # { v:[componentsToRemove], ... }
    @property
    def varValues(self):
        return dict([ (i, self.var[i]) for i in self.var.uniqueValuesForColonies_(self.segments)  ])
    def setup(self):
        for i,v in enumerate( self.variables ):
            print " %s)\t %s"%(i, v.name)
        while True:
            i=raw_input( "choose Variable: ")
            try:
                self.var = self.variables[ int(i) ]
                print "--> %s selected"%self.var.name
                break
            except Exception, exc:
                print "invalid var. Use number. Exc:  %s"%exc
    def selectComponents(self):
        self.cacheICAs()
        for v in self.trainedNodes:
            self.selectComponentsFor(v)
    def selectComponentsFor(self, v):
        if not v in self.trainedNodes:
            self.cacheICAs(values=[v])
        fig = P.gcf()
        projmatrix = self.trainedNodes[v].get_projmatrix().T
        temp = ravel(projmatrix)
        colorspace = eegplot.symmetricPolarColorspace(max(abs(temp)))
        nPlots = len(projmatrix)
        nRows = int( sqrt(nPlots) )
        nCols =  nPlots/nRows + bool(nPlots%nRows)
        axes = []
        crossOuts = {}
        for i, component in enumerate( projmatrix ):
            ax = fig.add_subplot(nRows, nCols, i+1)
            ax.ID=i
            axes.append(ax)
            eegplot.ax_topomap_data( component, self.sensors)#, colorspace=colorspace )
            ax.set_title( "Component %s"%i )
            if i in self.removeComponents[v]:
                line = Line2D( ax.get_xlim(), ax.get_ylim(), linewidth=2.5, color='r'  )
                crossOuts[i]= ax.add_line(line)
        def onclick(event):
            ax = event.inaxes
            ID = ax.ID
            print '>button ax.ID %s'%(ID)
            if ID in self.removeComponents[v]:
                self.removeComponents[v].remove(ID)
                crossOuts[ID].remove()
                del crossOuts[ID]
            else:
                self.removeComponents[v].add(ID)
                line = Line2D( ax.get_xlim(), ax.get_ylim(), linewidth=2.5, color='r'  )
                crossOuts[ID] = ax.add_line(line)
            print "toggele %s"%ax.ID
            P.draw()
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        P.show()
        fig.canvas.mpl_disconnect(cid)
        print 'done; rem components %s'%(str(self.removeComponents))
        
    def invalidateCache(self):
        self.trainedNodes = {}
        self.removeComponents = {}
        DerivedDataset.invalidateCache(self)
    def cacheICAs(self, values=None):
        if values==None:
            values =    set( self.var.uniqueValuesForColonies_([self.segments]) )   -  set( self.trainedNodes.keys() )
        logging.debug( " ICA caching ICAs for values %s"%(str(values)) )
        for v in values:
            ica = mdp.nodes.CuBICANode()
            segments = [ s.source.data for s in self.segments if s[self.var]==v ]
            data = vstack(segments)
            logging.debug(" ICA: training %s with %s segments. Total length %s."%(v, len(segments), str(data.shape)))
            ica.train( data )
            self.trainedNodes[v]=ica
            self.removeComponents[v]=set()
        return
            # TODO: intractive selection
    
    #def p(self, segment):
    #    segment.eventsCache      = segment.source.events
    #    if not segment[self.ICAVar] in self.trained:






'''