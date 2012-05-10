"""

Available importers
-------------------

txt     tab separated value text files (one line equals one time sample, each
        column equals one channel)
mat     matlab files exported by ???
wav     wave sound files
audacity  extracts cues from audacity files (produces event dataset). Can be
        used in combination with importer.wav to segment wave files according
        to cues set in Audacity. 

eeg
---

eph    Epoched eph
egi    Electrical Geodesics epoched RAW

sensors    import sensor nets for eeg 




Importers load data from different files and create Slave_Datasets with valid
properties.


Importers should be Importer subclasses and provide the following
functionality:

- initializing with Experiment instance:
    >>> e = Experiment()
    >>> i = importer.egi(e)

- information: __str__ method which displays importer settings
    >>> print i

- import files:
    >>> importer.get()


e.g.:

...





Idea for restructuring
----------------------

current Importer
- importer has settings for extracting data and creates Slave_Datasets

NewImporter
 - importer reads files and feeds data to extractors
 - extractors extract relevant data  
   - properties: extractor is child of importer -> derives properties and modifies
   - data
     importer: data = self.import(path)
               for extractor in self.extractors:
                   extractor.extract(data)

    the beauty of it: importer can restrict itself to file management and io;
    extractor takes care of stuff like topography 

"""

from __future__ import division

import os, time
import logging
import multiprocessing
import threading

import numpy as np
import scipy as sp
import scipy.io.wavfile
import scipy.io
import matplotlib.pylab as P

from eelbrain import ui
from eelbrain import fmtxt

from eelbrain.vessels import sensors
import param
import fileio
import segments as _seg
from datasets_op_events import _evts_from_data
from datasets_base import ExperimentItem, Slave_Dataset, Slave_Event_Dataset


__hide__ = ['Importer', 'uts_importer', 'eeg_importer',
            'scipy', 'fmtxt',
            'Slave_Dataset', 'Slave_Event_Dataset', 
            'bioread', 'division', 
            'UTS_Segment', 'Event_Segment', 'ExperimentItem', ]         


class Importer(ExperimentItem):
    """
    _Importer Base Class_
    
    Procedure
    =========
    create the importer:
    
    i = importer.Importer(experiment)
    
    i.p.source.set():  to specify source folder or files
    
    print i.p
    i.p... (modify parameters)
    
    i.get()
    
    """
    properties = {'data_type': "import",
                  } # tells __init__ to filter files for extension
    _default_ext = None 
    store = None
    def __init__(self, experiment, source=None, name='importer'):
        """
        **Arguments**
        
        experiment : 
            parent experiment instance
        
        source : str | None
            optional, path to source folder (sets self.p.source)
        
        name : str
            name for the importer
        
        """
        ExperimentItem.__init__(self, experiment, name)

        self.fnvars = []
                
        # set up sample file
        self._example_path = None
        self._example_data = None
        self._imported_files = {} # id -> date
        
        # add source path
        if source is not None:
            self.p.source.set(source)
    
    def _addparams_(self, p):
        
        self.p._section_("File Source")
        self.p.source = param.FileList(ext=self._default_ext, desc="Files that "
                                       "contain data to be imported")
        logging.debug(str(self.p.source))
        self.p.vars_from_names = param.VarsFromNames(del_cache=False,
                                                     namesource=self.p.source)
        
        self.p._section_("Data Properties")
        self.p.t0 = param.Param(del_cache=False, desc="location of t0 from data"
                                                      " start, in seconds")
    def get_input_properties(self):
        """
        Is called by i._datasets() as a basis for properties of the imported
        datasets. Should retrieve from i.p those properties that are equal for
        all imported datasets.
        
        """
        properties = {
                      't0': self.p.t0.get(),
                      }

        return properties
    def delete_properties(self):
        pass
    def delete_segments(self):
        pass
    def delete_cache(self, ids=None, _ids=None):
        "called by p.source when files change; calls self.read_example_file"
        filelist = self.p.source.get_as_filelist()
        if len(filelist) == 0:
            return
        path = filelist[0]
        if path == self._example_path:
            return
        self.read_example_file(path)
    ###SUBCLASS###SUBCLASS###SUBCLASS###SUBCLASS###SUBCLASS###SUBCLASS#######
    def read_example_file(self, path):
        """
        is called automatically whenever a new file is on the top of the import
        file list.
        
        """
        pass
    ###SUBCLASS###SUBCLASS###SUBCLASS###SUBCLASS###SUBCLASS###SUBCLASS#######
    def __repr__(self):
        return "importer." + ExperimentItem.__repr__(self)
    def __str__(self):
        "Provide a summary of the import settings"
        title = "importer.{cn}:".format(cn=self.__class__.__name__)
        out = [title,""]
        # properties
        out += self.p.__repr__().split(os.linesep)
        out.append("")
        out += ["1) set properties in importer.p.*",
                "2) use importer.get() to fill data segments. Any change to properties",
                "   that changes the data requires a call to importer.et()"]
        return os.linesep.join(out)
    ######## GET ##########################################
    def get(self, save=True):
        """
        checks property definitions and then imports and adds a new datast 
        to experiment. 
        
        :arg bool save: save the experiment after import is complete
        
        
        calls relevant for subclassing (in order)
        -----------------------------------------
        
        self._check      should print warnings or raise when property definitions
                        fail.

        for x in self._datasets()       iter over properties dict for the 
                                        Slave_Datasets (in order!)
        
        self._variables(self.experiment)    list of segment variables to add to 
                                            segments; can be: 
                                            - VarCommander, 
                                            - str, 
                                            - (name, dtype) tuple (*args for new 
                                              VarCommander)
        
        for (segment, filename) in self._segments(self.children)
                               out  - segment: segment object
                                    - filename: filename, used to construct 
                                      filename variabels
                                      
                                in  - self.children provides the datasets that 
                                      have been created based on the call to
                                      self._datasets()
        """
        # check some preconditions
        logging.debug("%s.get()"%self.__repr__())
        self._check()
        t_start = time.time()
        # kill all existing Slave_Datasets
        for c in self.children:
            raise NotImplementedError("Can't re-run import")
            c.delete()
        # free space
        if hasattr(self, '_example_data'):
            del self._example_data

            
        # create children
        for properties in self._datasets():
            if 'name' in properties:
                name = properties['name']
            else:
                name = 'importer.' + self.__class__.__name__
            if properties['data_type'] == 'event':
                new_d = Slave_Event_Dataset(self, name=name)
                new_d._set_varlist_(self.get_varlist())
            else:
                new_d = Slave_Dataset(self, name=name)
            new_d._set_properties_(properties)
                #!!! varlist must not be deepcopied!
        
        # adding segments
        for segment, filename in self._segments(self.children):
            values = self.p.vars_from_names.split(filename)
            for k, v in values.iteritems():
                segment[k] = v
        logging.debug("_segments loop finished")
        
        # finish
        for c in self.children:
            #if 'memmap_mgr' in c.__dict__:
            c._store.write_cache()
        
        logging.info(" Import finished. Took %s s"%(time.time() - t_start))
        
        if save:
            self.experiment.save()
    def get_varlist(self):
        time = self.experiment.variables.get('time')
        duration = self.experiment.variables.get('duration')
        magnitude = self.experiment.variables.get('magnitude')
        varlist = [time, duration, magnitude]
        return varlist
###SUBCLASS###SUBCLASS###SUBCLASS###SUBCLASS###SUBCLASS###SUBCLASS###SUBCLASS###
    def _check(self):
        """
        called before importing starts; can raise an error if something is not
        set up correctly 
        
        """
        pass
    def _datasets(self):
        """
        iter over properties dict for the Slave_Datasets (in order!)
        
        Default: 
        creates a single dataset based on the importer._i_properties dictionary.
        
        """
        properties = self.get_input_properties()
        return [properties]
    def _segments(self, datasets):
        raise NotImplementedError



class audacity(Importer):
    """
    Audacity Tag Importer
    ---------------------
    
    Event importers provide additional properties:
     - 'data_type' = 'evente'
     - '
     
    data is stored as nparray
     time x sensor x variable-array
     
    _Properties_:
    event_variables:    
        In this case name and dtype of the audacity label var. dtype can be None
    
    
    """
    properties = {'data_type': "import-aup",
                  'target_name': "Audacity Events",
                  }
    _default_ext = 'aup'
    def _addparams_(self, p):
        self.p._section_("Audacity Events")
#        self.p.label2dict = param.Param(default=True)
        labelvar = self.experiment.variables.get('event')
        self.p.label_var = param.VarCommander(default=labelvar, desc="var that "
                           "will contain the audacity event label")
        self.p.force_duration = param.Param(default=False, desc="False, or "
                                            "duration in seconds")
    def get_input_properties(self):
        properties = Importer.get_input_properties()
        properties['name'] = 'Audacity Events'
        return properties
    def _check(self):
        Importer._check(self)
        if not self._i_properties['label2dict']:
            raise NotImplementedError
        if self._i_properties['force_duration']:
            if not np.isscalar(self._i_properties['force_duration']):
                raise ValueError(" force_duration property must be scalar")
    def _datasets(self):
        properties = self.get_input_properties()
        properties['data_type'] = 'event'
        return [properties]
    def get_varlist(self):
        time = self.experiment.variables.get('time')
        duration = self.experiment.variables.get('duration')
        labelvar = self.p.label_var.get()
        varlist = [time, duration, labelvar]
        return varlist
    def _segments(self, datasets):
        dataset = datasets[0]
        force_duration = self.p.force_duration.get()
        from xml.dom.minidom import parse
        # get VarCommanders from experiment
        labelvar = self.p.label_var.get()

        files = self.p.source.get()
        for path in files:
            filename = os.path.basename(path)
            
            # id
            id, ext = os.path.splitext(filename)
            while id in self._imported_files:
                id += '_' # FIXME:
            self._imported_files[id] = os.path.getmtime(path)
                
            audacity_file = parse(path)
            e_list = []
            for label in audacity_file.getElementsByTagName('label'):
                # get values
                t = float(label.getAttribute('t'))
                t1 = float(label.getAttribute('t1'))
                title = label.getAttribute('title')
                # check/transform values
                if force_duration:
                    d = force_duration
                else:
                    d = t1 - t
                title_code = labelvar.code_for_label(title)
                # append to list
                e_list.append([t, d, title_code])
            e_list = np.array(e_list)
            #vtable = VarTable(varlist, e_list)
            seg_duration = e_list[-1,0] + e_list[-1,1]
            properties = dict(duration = seg_duration,
                              shape = e_list.shape)
            new_seg = _seg.Event_Segment(properties, dataset=dataset, data=e_list,
                                         name=filename, id=filename)
            yield new_seg, filename






class uts_importer(Importer):
    """
    importer subclass for importing uts-files containing one or more channel.
    
    self.channels=[(Name, slice), ...] or =None
    self.events=[(channel, threshold), ...]
    
    
    
    ###SUBCLASS###
    --------------
  - data, properties = self.get_file(path)   
    should return sample data and properties derivable from the first filename
    
    _i_properties with values other than None are assumed to be constant, and
    file properties are checked on import 
    
    """
    properties = {'data_type': "import",
                  }
    def _addparams_(self, p):
        Importer._addparams_(self, p)
        
        p._section_("UTS")
        p.channels = param.DataChannels(del_cache=False)
        p.samplingrate = param.Param(del_cache=False, default=200)
        
        p._section_("Epoching")
        desc = ("length in samples of one epoch. If ==None, the whole file is "
                "imported as one epoch. Otherwise, the file is split, and the "
                "variable i.p.epoch_trial is used as index")
        p.epoch_length = param.Param(default=None, can_be_None=True, desc=desc)
        p.epoch_var = param.VarCommander(desc="epoch index (see epoch_length)")

    def get_input_properties(self):
        properties = Importer.get_input_properties(self)
        properties['samplingrate'] = self.p.samplingrate.get()
        return properties
    def read_example_file(self, path):
        self._example_path = path
        if path == None:
            self.p.channels._set_n_channels(0)
            self._example_data = None
            self._example_prop = None
        else:
            data, properties = self.get_file(path)
            assert data.ndim == 2, "get_file returned invalid data"
            t, n = data.shape
            # save example
            properties['dtype'] = data.dtype
            self._example_data = data
            self._example_prop = properties
            
            # use example properties to set p
            self.p.channels._set_n_channels(n)
            if 'samplingrate' in properties:
                new_sr = properties['samplingrate']
                old_sr = self.p.samplingrate.get()
                if old_sr != new_sr and bool(new_sr):
                    self.p.samplingrate(new_sr)
                    ui.message("Samplingrate updated %s Hz -> %s Hz"%(old_sr, new_sr))

    def _datasets(self):
        # get common properties (epoch)
        epoch_length = self.p.epoch_length.get()
        segmented = bool(epoch_length)

        if not segmented:
            epoch_length = None
        
        
        # loop through channels to import
        for i, settings in self.p.channels.iterchannels():
            if settings not in [None, False]:
                # check index
                if isinstance(i, int):
                    n_chan = 1
                    index = [i]
                elif isinstance(i, slice):
                    n_chan = i.stop - i.start
                    index = i
                else:
                    n_chan = len(i)
                    index = i
                
                # properties
                name, ds_type, arg1, arg2 = settings
                properties = self.get_input_properties()
                properties['segmented'] = segmented
                properties['name'] = name
                properties['source_index'] = index
                properties['n_sensors'] = n_chan
                
                assert ds_type in ['uts', 'evt', 'topo']
                if ds_type == 'evt':
                    assert n_chan == 1, "cannot convert more than 1 channel to events"
                    # the channel shoud be converted to events
                    properties['data_type'] = 'event'
                    properties['evt_threshold'] = arg1
                    properties['evt_targets'] = arg2
                else:
                    properties['shape'] = (epoch_length, n_chan)
                    properties['ndim'] = 1
                    if ds_type == 'uts':
                        # it is a uts data channel
                        properties['data_type'] = 'uts'
                    elif ds_type == 'topo':
                        # it is topographic uts data 
                        properties['data_type'] = 'utstopo'
                        properties['sensors'] = arg1                    
                    
                yield properties

    def _segments(self, datasets):
        """
        useage::
        
            >>> for (new_seg, filename) in self._segments(self.children)
        
        :arg datasets: the datasets that have been created based on the call to
            self._datasets()
        
        :returns: new_seg: segment object; filename: filename, used to construct 
            filename variabels

        """
        logging.debug("preparing to IMPORT")
        # prepare properties and varlist if there are event channels 
        i_properties = self.get_input_properties()
        
        ###   prepare -- epoching   ###   ###   ###   
        epoch_length = self.p.epoch_length.get()
        epoch_var = self.p.epoch_var.get()
        
        # start import loop
        files = self.p.source.get_as_filelist()
        
        # progress bar
        n = len(files)
        prog_i = 0
        prog_ttl = "Importing {n} Files".format(n=n)
        prog_msg = "{i} of {n} files done...".format(n=n, i='{i}')
        prog = ui.progress(i_max=n,
                           title=prog_ttl,
                           message=prog_msg.format(i=0))
        
        for path in files:
            filename = os.path.basename(path)
            logging.debug("%i of %i: %s" % (prog_i, n, filename))
            data, properties = self.get_file(path)
            
            # id
            id, ext = os.path.splitext(filename)
            while id in self._imported_files:
                id += '_' # FIXME:
            self._imported_files[id] = os.path.getmtime(path)

            
            # assert that input conforms to dataset properties; if not, skip.
            errors = []
            superfluous = []
            t0 = i_properties['t0']
            for k, v in properties.iteritems():
                if k in i_properties:
                    if i_properties[k] != v:
                        txt = " Property [{p}] mismatch: {vf} (file) != {vd} dataset"
                        errors.append(txt.format(p=k, vf=v, 
                                                 vd=i_properties[k]))
                    else:
                        superfluous.append(k)
            for k in superfluous:
                properties.pop(k)
            
            if errors:
                txt = '\n'.join(["! Skipped: {f}".format(f=filename)] + errors)
                logging.warning(txt)
            else:  # Add to datasets
                n_samples, n_sensors = data.shape
                if epoch_length:
                    # check shape 
                    if n_samples % epoch_length != 0:
                        raise ValueError("n_samples not divisible by epoch_"
                                         "length for '%s'"%filename)
                    
                    epoch_indexes = np.arange(0, n_samples, epoch_length)
                    i_length = epoch_length
                else:
                    epoch_indexes = [0]
                    i_length = n_samples
                
                for i, i_start in enumerate(epoch_indexes):
                    
                    for dataset in datasets:
                        index = dataset['source_index']
                        i_stop = i_start + i_length
                        c_data = data[i_start:i_stop, index]
                        
                        # segment name
                        if epoch_length:
                            seg_name = "%s, %s-%s" % (filename, i_start, i_stop)
                            id = '%s_%s_%s' % (filename, i_start, i_stop)
                        else:
                            seg_name = filename
                        
                        if dataset.properties['data_type'] == 'event':
                            samplingrate = dataset['samplingrate']
                            seg_duration = c_data.shape[0] / samplingrate
                            
                            e_list = _evts_from_data(c_data, samplingrate, t0=t0,
                                                     threshold=dataset.properties['evt_threshold'],
                                                     targets=dataset.properties['evt_targets'])
                            properties.update(duration = seg_duration,
                                              shape = e_list.shape)
                            new_seg = _seg.Event_Segment(properties=properties, 
                                                         data=e_list, dataset=dataset, 
                                                         name=seg_name, id=id)
                        else: # UTS
                            properties.update(shape = c_data.shape)
                            new_seg = _seg.UTS_Segment(properties=properties, dataset=dataset,
                                                       data=c_data, name=seg_name, id=id)
                        
                        # assign epoch id
                        if epoch_length:
                            if epoch_var is not None:
                                new_seg[epoch_var] = i
                        
                        yield new_seg, filename
                    
            
            # progress monitoring
            prog_i += 1
            if prog:
                prog.advance(prog_msg.format(i=prog_i))
        # terminate progress bar
        if prog:
            prog.terminate()
    
    def plot(self, fig_num=None):#, tmax=1000):
        if self._example_path == None:
            ui.message("No files specified")
            return
        
        data = self._example_data#[:tmax]
        
        # range correct data
        t, n = data.shape
        data = data - np.min(data, 0)
        data /= np.max(data, 0) * 1.1
        data += np.arange(n-1, -1, -1) # channel 1 on the top
        
        # collect properties
        name = os.path.basename(self._example_path)
        samplingrate = self.p.samplingrate.get()
        T = np.arange(0, t)
        if samplingrate:
            T /= samplingrate
        
        # start plot
        P.figure(fig_num)
        P.subplots_adjust(.2, .05, .99, .9)
        P.suptitle("Example: %s"%fmtxt.texify(name))
        ax = P.axes()
        lines = [ax.plot(data[:,i], c='.75', antialiased=False)[0] for i in xrange(n)]
        y_ticks = np.arange(n-1, -1, -1) + .45
        y_ticklabels = range(n)        
        # get channels data
        for index, settings in self.p.channels.iterchannels():
#            ax = P.subplot(n, 1, i+1, sharex=ax)
            name, ds_type, arg1, arg2 = settings
            
#            y_ticklabels.append(name)
            if isinstance(index, int):
                ls = [lines[index]]
                y_ticklabels[index] = name
#                y_ticks.append(i)
            else:
                if isinstance(index, slice):
                    start, stop = index.start, index.stop
                    ls = lines[index]
                    indexes = xrange(start+1, stop+1)
                else:
                    start = index[0]
                    indexes = index[1:]
                    ls = [lines[i] for i in index]
                y_ticklabels[start] = name
                for i in indexes:
                    y_ticklabels[i] = ''
#                tick_y = start + (stop-start)//2
#                y_ticks.append(tick_y)
            
            if ds_type == 'uts':
                c = 'k'
            elif ds_type == 'topo':
                c = 'r'
            elif ds_type == 'evt':
                c = 'b'
            
            for l in ls:
                l.set_color(c)
        
        if hasattr(ax, 'tick_params'):
            ax.tick_params(axis='y', length=0)
        
        ax.set_xlim(0, len(data))
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticklabels)
#        ax.set_xlim(T[0], T[-1])
#        ax.set_xlim(0, tmax)
        P.show()
        
        # plotting before merging with eeg importer
#            if settings is False:
#                ax.plot(T, data[:,i], '.75')
#                ax.set_ylabel("ch %s (drop)"%i, color='.5')
#            else:
#                name, evt, evt_threshold, evt_step = settings
#                if evt == None:
#                    ax.plot(T, data[:,i], 'r')
#                    ax.set_ylabel(name, color='r')
#                elif evt == 'evt':
#                    ax.plot(T, data[:,i], 'b')
#                    if np.isscalar(evt_threshold):
#                        ax.axhline(evt_threshold, c='#FF9900')
#                    elif np.iterable(evt_threshold):
#                        for th in evt_threshold:
#                            ax.axhline(th, c='#FF9900')
#                    ax.set_ylabel(name + " (EVT)", color='b')
#                else:
#                    ui.msg("invalid channel specification: '%s'"%evt)

#        event_channels = [i for name, i in self.events]
#        logging.debug(" {0} + {1}".format(self.channels, self.events))
#        channel_names = dict([(i, name) for name, i in self.channels+self.events])
#        for i in range(n):

    ### SUBCLASS ###### SUBCLASS ###### SUBCLASS ###### SUBCLASS ###### SUBCLASS
    def get_file(self, path):
        """
        return data, properties
        
        data: [t x data] array
        properties: dict with properties that can be derived from file
        
        """
        raise NotImplementedError
        
        
class txt(uts_importer):
    "Txt Importer"
    _default_ext = 'txt'
    def get_file(self, path):
        "return data and dict with properties that can be derived from file"
        data = np.loadtxt(path)
        return data, {}

try:
    import bioread
    
    
    class acq(uts_importer):
        """
    Biopac Acq Importer
    -------------------
    TODO: make use of channel names
           
        """
        _default_ext = 'acq'
        def __init__(self, experiment, source=None, name='importer', samplingrate=250):
            """
            Currently the acq importer can only handle channels of a single
            samplingrate. The samplingrate argument specifies which 
            channels are available.
            
            """
            self._samplingrate = samplingrate
            super(uts_importer, self).__init__(experiment, source=source, name=name)
        
        def get_file(self, path):
            ACQ = bioread.read_file(str(path)) # can't handle unicode
            
            # filter channels to those of matching samplingrate
            channels = [c for c in ACQ.channels if c.samples_per_second == self._samplingrate]
            
            # read data
            n_channels = len(channels)
            if n_channels == 0:
                srs = np.unique([c.samples_per_second for c in ACQ.channels])
                err  =("No channels found. Change self._samplingrate to one of "
                       "%s?" % srs)
                raise ValueError(err)
            
            c0 = channels[0]
            length = c0.point_count
            if n_channels > 1:
                assert all(c.point_count == length for c in channels[1:])
            data = np.empty((length, n_channels))
            for i,c in enumerate(channels):
                data[:,i] = c.data
            # properties
            properties = dict(samplingrate = self._samplingrate,
                              )
            return data, properties
except:
    print "to enable acq import, install bioread module"


class wav(uts_importer):
    """
    .wav audio File Importer
    
    difference to binary importer is that audio importers provide sampling rate
    along with data.
    
    """
    properties = {'data_type': "import-wav",
                  'target_name': "Wav",
                  }
    _default_ext = 'wav'
    def get_file(self, path):
        samplingrate, data = sp.io.wavfile.read(path)
        properties = {'samplingrate':samplingrate}
        return data[:,None], properties



class mat(uts_importer):
    """
    UTS-importer for Matlab Files
    
    """
    properties = {'data_type': "import-mat",
                  'target_name': "MatLab",
                  }
    _default_ext = 'mat'
    def get_file(self, path):
        m_file = scipy.io.loadmat(path)
        data_blocks = []
        i = 1
        while 'data_block%s'%i in m_file.keys():
            data_blocks.append( m_file['data_block%s'%i].T )
            i += 1
        data = np.concatenate(data_blocks)
        properties = {}
        logging.debug(" {p}: {s}".format(p=path, s=data.shape))
        return data, properties



# MARK: DEFUNCT Importer method
'''
    def guessvars(self):
        """
        Divide filenames into parts that are constant across file and parts that
        are varying. Create a variable for each varying part.
        
        """
        raise NotImplementedError()
        # from egi raw import func:
        flex = systemOps.guessFilenameStructure(filenames)
        tout = ''
        for i,f in enumerate(flex):
            tout = tout.ljust(f[0])+str(i+1)*(f[1]-f[0])
        print filenames[0]+'\n'+tout    
        filenamevars = []
        for i,f in enumerate(flex):
            name = raw_input("Variable %s Name:"%(i+1))
            filenamevars.append( (self.fnvars.getNewCommanderWithName_dtype_(name, 'dict'), f) ) # [name, [start, stop]] 

'''
