"""
Base class for Datasets. Datasets contain and manage Segments.


Dataset Classes
---------------

Experimentitem: node in a hierarchical experiment structure.
    .p: properties that can be changed by set user
    - subclass importer.Importer

Dataset: ExperimentItem with segments containing data and data properties
    - subclass importer.Slave_Dataset  

DerivedDataset: Dataset with mechanism to create segments based on properties 
        self.p




dataset.properties returns properties shared by all segments in the dataset.
Cf. properties listed in segments.py.


Dataset Types
------------- 
Importer: uses some basic properties like the params
Slave: Dataset that is controlled by parent (e.g. Importer)



dataset class attributes
------------------------

"""

import logging, time
from copy import deepcopy

import numpy as np

from eelbrain import ui
from eelbrain import fmtxt

import vars as biovars
import param
from segments import StatsSegment, UTS_Segment, Event_Segment
import _cache



def is_experiment_item(item):
    return isinstance(item, ExperimentItem)

#import dataset_mixins as mixins
        

class _data_subset(object):
    """
    Simulacrum of a dataset with only a subset of the original segments. 
    Initialized through::
    
        >>> dataset[address]
    
    """
    def __init__(self, dataset, address):
        self.source = dataset
        if type(address) is biovars.Address:
            self.address = address
        else:
            raise ValueError("type(address) == %s != Address"%str(type(address)))
    @property
    def segments(self):
        return self.address.filter(self.source.segments)
    def __len__(self):
        return len(self.segments)
    def __iter__(self):
        for s in self.segments:
            yield s
    def __getitem__(self, i):
        if np.isscalar(i):
            return self.segments[int(i)]
        else:
            return self.source[i]
    def __getattr__(self, name):
        try:
            return object.__getattribute__(self, name)
        except:
            return getattr(self.source, name)


class ExperimentItem(object):
    """
    Experiment Hierarchy Base Class: Provides base functions which are needed 
    to integrate an object into the experiment hierarchy
    
    """
    def __init__(self, parent, name='{c}_{i}'):
        """
        Add an ExperimentItem to the experiment. The name must be a valid and 
        unused attribute for the parent experiment. 
        
        The name can contain the following formatting keys:
        * '{c}' the class name 
        * '{i}' is an integer which is incremented (starting at 0) to yield a 
          unique name
        
        .. CAUTION:: 
            Automated names are discouraged because they can lead to trouble
            later. For example, when an additional dataset is inserted into
            a chain in an importer script, all the names down from there can
            change, and old analysis scripts will now use the wrong datasets.
            
        """
        # check parent properties
        if hasattr(parent, 'properties'):
            validation = self._validate_input_properties_(parent.properties)
            if validation is not True:
                raise ValueError("Invalid Parent: %s" % str(validation))
        
        ## experiment hierarchy
        if isinstance(parent, ExperimentItem): #hasattr(parent, 'experiment'):
            self.experiment = parent.experiment
            self.parent = parent
        else: # = Experiment
            self.experiment = parent
            self.parent = None
            
        self.id, self.name = self.experiment._register_item(self, name)
        parent.children.append(self)
        
        self.children = []
        # parameter
        self.p = param.Host(self)
        self._addparams_(self.p) # -> add custom params
        self.p._section_("General")
        self.p.color = param.Color()
        # initialization for subclasses
        self._init_()
    def __repr__(self):
        temp = '{c}({pname}, "{name}", <{dtype}>)'
        if hasattr(self.parent, 'experiment'):
            pname = '<e>.%s'%self.parent.name
        else:
            pname = '<e>'
        return temp.format(#id = self.id,
                           c = self.__class__.__name__,
                           pname = pname,
                           name = self.name,
                           dtype = self.properties['data_type'])
    def _get_tree_repr_(self, indent=''):
        n = len(self.children)
#        if n == 0:
#            return [indent + "|" + self.name]
#        else:
        lines = [indent + "|" + self.name]
        for i, c in enumerate(self.children):
            loc_indent = indent + '| '*(n-1-i)
            lines += c._get_tree_repr_(indent=loc_indent)
        if n == 0:
            lines.append(indent)
        return lines
    def rename(self, name):
        self.experiment.rename_item(self, name)
    def delete(self):
        for c in getattr(self, 'children', []):
            c.delete()
        self.p._delete()
        self.experiment._del_item(self)
        if hasattr(self, 'memmap_mgr'):
            self.memmap_mgr.delete()
        if hasattr(self.parent, 'children'):
            self.parent.children.remove(self)
    def close(self):
        for c in self.children:
            c.close()
    ## SUBCLASS
    def _validate_input_properties_(self, properties):
        "should return True or string describing the problem"
        return True
    def _init_(self):
        pass
    def _addparams_(self, p):
        pass
    """
    
    Data Interface
    --------------
    
     - compiled: 
     - properties: properties of the data
     - segments: ordered list of all segments
    
    Eventsets additionally provide:
    
     - varlist: list of the variables in the data (see mixin class below)
    
    
    """
    @property
    def compiled(self):
        if not hasattr(self, '_compiled'):
            self._compiled = self._create_compiled_()
        return self._compiled
    @property
    def properties(self):
        if not hasattr(self, '_properties'):
            self._properties = self._create_properties_()
        return self._properties
    @property
    def segments(self):
        "self._segments is created by Segment.attach_to_dataset() method."
        if not hasattr(self, '_segments'):
            self._create_segments_()
        return self._segments
    @property
    def segments_by_id(self):
        if not hasattr(self, '_segments_by_id'):
            self._segments_by_id = dict((s._id, s) for s in self.segments)
        return self._segments_by_id
    ###    craete (must be implemented by subclass)    ###  ###  ###  ###  ###
    def _create_compiled_(self):
        raise NotImplementedError
    def _create_properties_(self):
        raise NotImplementedError
    def _create_segments_(self):
        raise NotImplementedError
    ###    delete    ###  ###  ###  ###  ###  ###  ###  ###  ###  ###  ###  ###
    def _del_private_attrs(self, *attrs):
        for attr in attrs:
            name = '_'+attr
            if hasattr(self, name):
                delattr(self, name)
    def delete_properties(self):
        logging.debug(" Dataset %s DEL properties"%self.name)
        
        for c in self.children:
            c.delete_properties()
        self._del_private_attrs('properties', 'compiled', 'varlist')
    
    def delete_segments(self, IDs=None):
        logging.debug(" Dataset %s DEL segments"%self.name)
        
        if IDs:
            raise NotImplementedError("the problem is cache management...")
        
        for c in self.children:
            c.delete_segments(IDs=IDs)
    
        # remove the VarColonies from the VarMothership
        if hasattr(self, '_segments'):
            
#            if not IDs: # all
#                IDs = xrange(len(self._segments))
            
            for s in self._segments:
                s.variables.kill()
            del self._segments
            self.delete_cache()
            
            if hasattr(self, '_segments_by_id'):
                del self._segments_by_id



class Eventset:
    "Mixin"
    @property
    def varlist(self):
        """
        varlist must be managed separately from properties because it must not
        be deepcopied
        
        """
        if not hasattr(self, '_varlist'):
            self._varlist = self._create_varlist_()
        return self._varlist
    
    # must be implemented by subclass
    def _create_varlist_(self):
        raise NotImplementedError



class Dataset(ExperimentItem):
    """ 
    A Dataset contains data (segments) of one particular type (e.g. EEG, or 
    GSR) and stores technical information (like electrode locations)

    Methods:
     .collectstats()    create several statssegments
     .statsForAddress_over_(...)   returns a statistics segment
     .compare(cellvar, val1, val2, over)  plots the comparison
    
    Properties:
    
    """
    store = _cache.Memmap
    def _init_(self):
        if self.store:
            self._store = self.store(self.experiment)
        self._id2id = {}
        
        # repository for statistics
        self.stats = {}
#        self.anovas = {}
        # default kwargs for plotting functions (?)
#        self.kwargs = {'topo':{'interval':(-.1,1),
#                               'plotSensors':[8, 10, 21, 23, 32, 44, 51, 57, 61, 
#                                              74, 91, 95, 107, 121, 123, 128]},
#                       'tv':{},
#                       'statistics':{'parametric':True}}
    """
    representation
    """
    def __str__(self):
        # descr
        repr = self.__repr__()
        txt = [repr, '-'*len(repr), '']
        if self['data_type'] == 'event':
            varlist = ', '.join(v.name for v in self.varlist)
            txt.append("VARS:  "+varlist)
        # processing parameter
        if hasattr(self, 'p'):
            txt.append(str(self.p))
        # properties
        txt.append('\nProperties:')
        template = "  {k}: {v}"
        for k, v in self.properties.iteritems():
            txt.append(template.format(k=k, v=v))
        return '\n'.join(txt)
    def table_segvars(self, *vars, **kwargs):
        """
        *vars: variables
        *kwargs:
            labels=True: display labels (instead of values) for nominal 
            variables
            digits
        """
        _kwargs = dict(labels=True,
                       digits=4)
        _kwargs.update(kwargs)
        labels = _kwargs['labels']
        digits = _kwargs['digits']
        
        n = len(vars)
        table = fmtxt.Table('ll'+'r'*n)
        table.cell()
        table.cell('name')
        for var in vars:
            table.cell(var.name)
        table.midrule()
        for i, s in enumerate(self):
            table.cell(i, digits=0)
            table.cell(s.name)
            for var in vars:
                v = var[s]
                if labels:
                    v = var.label(v)
                table.cell(v, digits=digits)
        return table
    """
    properties access
    """
    def __len__(self):
        return len(self.segments)
    def __iter__(self):
        for s in self.segments:
            yield s
    @property
    def t0(self):
        return self.properties['t0']
    @property
    def ndim(self):
        return self.properties['ndim']
    @property
    def samplingrate(self):
        return self.properties['samplingrate']
    @property
    def sensors(self):
        return self.properties['sensors']
    def __getitem__(self, name):
        if biovars.isaddress(name):
            return _data_subset(self, name)
#            return name.filter(self.segments)
        elif isinstance(name, basestring):
            if name in self.properties:
                return self.properties[name]
            elif hasattr(self, 'compiled'):
                if name in self.compiled:
                    return self.compiled[name]
            raise KeyError("Property %r neither in properties nor compiled" % name)

        else:
            return self.segments[name]
    """
    
    datastore functions
    -------------------"""
    def _set_data(self, id, data):
        self._store[id] = data
    def _get_data(self, id):
        "return the data for the segment with ID"
        if id in self._store:
            out = self._store[id]
            if out is None:
                # if cache cannot be returned reset the cache and retrieve 
                # item again
                self.delete_cache()
                return self._get_data(id)
            else:
                return out 
        else:
            segment = self.segments[id]
            assert id == segment.id
            data = self._derive_segment_data_(segment)
            self._store[id] = data
            if segment.shape != data.shape:
                segment.properties['shape'] = data.shape
            return data
    def delete_cache(self, ids=None, _ids=None):
        "ids: list of ``segment._id``s"
        logging.debug(" Dataset {n} DEL cache {i}".format(n=self.name, i=ids))
        if _ids is not None:
            if ids is None:
                ids = []
            ids += [self._id2id[_id] for _id in _ids]
        
        for c in self.children:
            c.delete_cache(ids=ids)
        del self._store[ids]
        
        if hasattr(self, '_segments'):
            for segment in self._segments:
                if hasattr(segment, '_properties'):
                    del segment._properties
    def close(self):
        ExperimentItem.close(self)
        self._store.close()
    """
    
    statistics functions
    --------------------"""
    def statsForAddress(self, address, over, 
#                        exclude=None,
                        color=None, 
#                        _connector=None, 
                        **kwargs):
        """
        initializes a statsSegment with the mean for each value of 'over' variable for address
        kwargs: func: function that is applied to each segment.data before averaging
                    e.g. use abs to get Amplitude from wavelet data
                   -'phase' to get phase ( x/abs(x) - results in a complex number)
                **subdata kwargs
        """
        assert ('segmented' in self.properties) and self.properties['segmented']

        segments = address.filter(self.segments)
        
        if len(segments) == 0:
            return None
        properties = deepcopy(self.properties)
        if color:
            properties.update(color=color)
        s = StatsSegment(properties=properties, name=address.name)
        s.initWithSegments_over_(segments, over, **kwargs)
        
        return s
    
#        if exclude: # needs to be caught because does not work with non-stat subdata
#            ex_address = biovars.Address({over:(False, exclude)})
#            address = address * ex_address
#        segments = address.dict_over_(self.segments, over)
#        if _connector == None:
#            return s
#        else:
#            _connector.send(s)
#            _connector.close()
    def collectstats(self, cells, over, mask=None, mp=False, pool=None, 
                     save='ask', **kwargs):
        """
        cells: address; statistics are extracted for each valid combination of
               values in this address
        over: variable specifying the cases within the cells (usually subject)
        
        - adds statistics segment for all values of cells in address to self.stats
        - forwards statsForAddress_over_ kwargs
        
        kwargs:
        mp  False
            'outer' Spawn one process for each StatsSegment 
            True    do statsSegments serially but use multiple processes in
                    StatsSegment.initWithSegments_over_
                    
        **kwargs: 
                    
        """
        t0=time.time()
        msg = "--> Collecting Statistics for {0} over {1}, mp={2}"
        print msg.format(cells.name, over.name, str(mp))
        
        if len(self.stats) > 0:
            msg = "self.stats currently contains the following statistics: %s"%str(self.stats.keys())
            if ui.ask(title="Clear existing statistics?",
                      message=msg):
                self.stats = {}
        
        cells = biovars.asaddress(cells)
        if mask == None:
            all_segments = self.segments
        else:
            all_segments = mask.filter(self.segments)
        segs_sorted = cells.sort(all_segments)
        
        # progress bar
#        n = len(segs_sorted)
#        i = 0
#        prog_ttl = "Collecting statistics for {n} cells".format(n=n)
#        prog_msg = "{name} ({i} of {n}) done...".format(name='{name}', n=n, i='{i}')
#        prog = ui.progress(i_max=n,
#                           title=prog_ttl,
#                           message=prog_msg.format(name='', i=0))
        
        label_dic = cells.short_labels()
        
        for address, segs in segs_sorted.iteritems():
            label = label_dic[address]
#            color = cells.color_for_value(key)
            s = StatsSegment(self.properties, name=label)#, color=color)
            s.initWithSegments_over_(segs, over, **kwargs)

            # make sure label is a valid name
            if label[0].isalpha():
                valid_label = label
            else:
                valid_label = 'c' + label
            
            self.stats[valid_label] = s
            
            # close all files
            self._store.close()
        
#        # progress bar
#            if prog:
#                i += 1
#                prog.advance(prog_msg.format(name=valid_label, i=i))
#        if prog:
#            prog.terminate()


#        if mp=='outer':
#            for key, label in cells.dictionary.iteritems():
#                self.stats[label] = self.stats[label].recv()
        print '\n'.join(["---",
                         "Finished after %s"%(time.time()-t0),
                         "Dataset now contains stats: %s"%self.stats.keys()])
        
        # save the experiment
        if save == 'ask':
            msg = "Save Experiment with newly added statistics segments?"
            save = ui.ask(title="Save Experiment?", message=msg)
        if save:
            self.experiment.save()
            
## OLD #### OLD #### OLD #### OLD #### OLD #### OLD #### OLD #### OLD #### OLD
    # exporting
    def _export(self, address, format='txt', sensors=slice(None)):
        """
        .. WARNING::
            METHOD NOT MAINTAINED
        
        Possible values for format:
        'txt':  export tsv files to folder (ask)
        'names': print a list of filenames ordered for 2 vars in address
        
        """
        segDict = address.dict( self.segments )
        dictKey = address.keys()
        if format == 'txt':
            folder = ui.ask_dir('target dir: ')
            if folder == False:
                raise IOError("no valid target directory")
            for index, segments in segDict.iteritems():
                data = np.vstack([s.data for s in segments])
                name = '_'.join([ var.name+'-'+var[v] for var,v in zip(dictKey, 
                                                                       index) ])
                filename = folder + name +'.txt'
                np.savetxt(filename, data, delimiter='\t')
        elif format == 'names':
            if len(dictKey)!=2:
                raise NotImplementedError
            d0=[]; d1=[]; names={}
            for index in segDict.keys():
                if index[0] not in d0:
                    d0.append(index[0])
                if index[1] not in d1:
                    d1.append(index[1])
                if index not in names:
                    names[index]= '_'.join([var.name+'-'+var[v] for var,v in \
                                            zip(dictKey, index)]+['.txt'])
            txt=['\t']
            for i1 in d1:
                txt.append( dictKey[1][i1]+'\t' )
            for i0 in d0:
                txt.append('\n'+dictKey[0][i0]+'\t')
                for i1 in d1:
                    txt.append(names[(i0,i1)]+'\t')
            print ''.join(txt)
    # Plotting without pre caching stats segments -- not very efficient 
#    def compare(self, cellvar, val1, val2, over, **kwargs):
#        s1=self.statsForAddress_over_({cellvar:val1}, over)
#        s2=self.statsForAddress_over_({cellvar:val2}, over)
#        s1.compare(s2, **kwargs)
#    def compareF(self, frequency, cellvar, val1, val2, over, **kwargs):
#        s1=self.amplitudeInFrequency_ForAddress_Over_(frequency, {cellvar:val1}, over)
#        s2=self.amplitudeInFrequency_ForAddress_Over_(frequency, {cellvar:val2}, over)
#        s1.compare(s2, **kwargs)
## OLD #### OLD #### OLD #### OLD #### OLD #### OLD #### OLD #### OLD #### OLD


class Slave_Dataset(Dataset):
    store = _cache.Memmap
    """
    importer overwrites self.properties to invalidate @property ???
    
    """
    __slave_msg = ("Slave_Datasets need to be fed by their parents. Call parent's"
                   " .push method to create segments.")
    def _set_properties_(self, properties):
        self._properties = deepcopy(properties)
    # blocked functions
    def _create_segments_(self):
        raise AttributeError(self.__slave_msg)
    def _create_compiled_(self):
        raise AttributeError(self.__slave_msg)
    def _create_properties_(self):
        raise AttributeError(self.__slave_msg)
    def _derive_segment_data_(self, segment, preview=False):
        raise NotImplementedError(self.__slave_msg)
    def delete_properties(self):
        pass
    

class Slave_Event_Dataset(Slave_Dataset, Eventset):
    # blocked functions
    def _create_varlist_(self):
        raise AttributeError(self.__slave_msg)
    def _set_varlist_(self, varlist):
        self._varlist = varlist


class DerivedDataset(Dataset):
    """

    should implement the following functions
    ----------------------------------------
    - D._addparams_          popoulate self.p with Params (self.p._add(...))
    - D._validateInputProperties(properties)    called to check whether a data-
                                set with properties could be used as input. 
                                Return True or False, or raise an Error
    - D._compile_()  set D._properties based on the current p settings. Can also
                    create elements that will be the same for each segment and 
                    store them in D._compiled (which should be accessed as 
                    D.compiled to allow lazy creation). 
                    _properties NEED to contain (formemmap mgr):
                        - 'shape'
                        - 'dtype'
    - data = D._derive_segment_data_(segment)  called by Dataset._get_data; should 
                                    return the processed data and can modify
                                    the segment._p_attr dict


    can overwrite the following
    ---------------------------
    - D._create_segments_()    by default creates one segment for each segment in
                                    parent and adds 'parent' to segment._p_attr 
                                    dict
                                    also creates: D.cached, D.memmap_mgr, 
    - D.delete_segments()   removes the D._segments list so it has to be 
                            recreated on its next call 
    - D.init_cache(seglist=None)  seglist can contain ids or segs; 
                                    None=all
    
    
    Dataset superclass provides a memmap mgr interface:
    - Dataset.delete_cache(shape, dtype)      (is called by _create_segments_)
    - Dataset._set_data(id, data)       len must be constant. if len changes,
                                        the memmap mgr needs to be reinitialized
    - Dataset._get_data(id)
    
    Bound_UTS_Segments call  D._get_data(self.id), which is specified in the 
    Dataset class to access memmap_mgr, however memmap mgr must be created in 
    the subclass because of the dtype specification
    Dataset.
    
    
    
    when data is requested from a segment, the segment calls 
    dataset._get_data(self.id)
    
    

    kwargs
    ------
    
    
    OLD
    ---
    DESIGN REQUIREMENTS:
    - needs to dynamically react to changes in its parent
        - change in number of segments
    - segments need to be retrievable without forcing access to data (e.g. for 
      selecting based on vars)
    - segment needs continuity from one getitem to the next so it can access the
      cache  
        
-->
    - self.. retrieves the rawSegment list    
    


    properties management:
    .properties    dict 
    .compiled      dict
    
    subclass
    --------
    ._compile_() is called when .properties and .compiled are created, and can
                be used to add additional properties to the self._properties 
                and self._compiled dicts.
    """
    Child_Segment = UTS_Segment
    def _create_compiled_(self):
        """
        The base compile function called by access to D.properties and
        D.compiled; calls self._compile_ which can assign parameters
        by modifying _compiled and _properties dictionaries
        
        """  
        if hasattr(self, 'p'):
            compiled = self.p._compiled()
        else:
            compiled = {}
        return compiled
    def _create_properties_(self):
        "baseclass function just returns a deepcopy of the parent's properties"
        properties = deepcopy(self.parent.properties)
        return properties
    #####   SUBCLASS   #####   SUBCLASS   #####   SUBCLASS   #####   SUBCLASS   #####    
    def _create_segments_(self):
        """
        Simply creates one segment for each parent segment
        
        Note that Bound Segments append themselves to dataset._segments
        automatically
        
        """
        for s in self.parent.segments:
            p_attr = {'source': s}
            new_seg = self.Child_Segment(name=s.name, p_attr=p_attr,
                                         varsource=s.variables, 
                                         dataset=self, id=s._id)
            self._finish_new_segment(new_seg, s)
        
    def _add_segments(self, parent_segments):
        raise NotImplementedError
    
    def _derive_segment_properties_(self, segment):
        parent = segment._p_attr['source']
        return parent.properties
    def _finish_new_segment(self, new_segment, parent_segment):
        "can be used to set properties that need to be set before data is retrieved"
        pass
    def _derive_segment_data_(self, segment, preview=False):
        "? preview = function called with data as arg?"
        source = segment._p_attr['source']
        data = source.data
        return data



class Derived_UTS_Dataset(DerivedDataset):
    Child_Segment = UTS_Segment
    store = _cache.Memmap


class Derived_Event_Dataset(DerivedDataset, Eventset):
    Child_Segment = Event_Segment
    store = _cache.Local
    def _create_properties_(self):
        p = DerivedDataset._create_properties_(self)
        p['data_type'] = 'event'
        return p
    def _derive_segment_properties_(self, segment):
        parent = segment._p_attr['source']
        properties = parent.properties
        properties.update(duration = parent.duration,
                          shape = segment.data.shape)
        return properties

