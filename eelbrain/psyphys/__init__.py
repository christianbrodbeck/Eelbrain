'''
The overall model
=================

Hierarchical structure: 

 - A :class:`Segment` contains a continuous record of data, as well as 
   properties (e.g., samplingrate)
 - A :class:`Dataset` manages a collection of 
   :class:`Segments`
 - An :class:`Experiment` manages a hierarchical tree of 
   :class:`Datasets`



Properties
----------

Bound segments store only properties which differ from their dataset; Properties
which are identical for all segments in a dataset are stored in 
``dataset.properties``.


``data_type``: str 
    the type of data contained in the segments. Used by internal 
    mechanisms to determine icons etc. 
    ``'uts'``: data sampled with a constant sampling rate. dimension 0 is 
    time. (e.g. voltage by time recordings)
    ``'utstopo'``: uts with topography
    ``'event'``: events (VarTable) 
    should have 'varlist' attr (not in properties bc. VarCommanders
    must not be deep copied        

``ndim``: int
    number of dimensions in data (apart from time/sensor) --> normal uts 
    data has 1; time-frequency data has 2

``n_sensors``: int
    the number of sensors (first dimension after time, shape[1])

``samplingrate``: scalar
    samplingrate in Hz

``sensors``: sensor_net
    sensor_net instance specifying the sensor coordinates

``shape``: tuple
    data.shape - so that data shape can be retrieved without loading 
    the data (time x sensor x ...; for dataset, None means that this
    dimension differs between segments.

``t0``: float
    time in seconds from segment start to t0 (other time points like tstart, 
    tend etc. are calculated from t0).

``unit``: str
    the unit of measurement (used for plotting as y-label)

``ylim`` : float
    for plotting: default limit for the y-axis  

``ylim_mean`` : float
    default new ylim when taking a ``.mean()``; if not specified, ``ylim`` is 
    inherited.  


Segments
--------

 - Provide access to the data
 - unique ID based on index in the segment list. Any dataset that manipulates
   segments (epoching, averaging) needs to make sure IDs propagate in the right way!!
   ??? is an ID based on segment order sufficient?  YES (so far)


Datasets
-------- 
used for maintaining transformation parameters

 - have a .pull method which should fill all segments with data and can use
   multiprocessing
 - push method might be more useful when one data chunk is distributed over
   several recipients (e.g., importer to extractors)
   
 - for dynamically managing Segments: [FIXME: PROBLEM:] cache management: 

    - by source file?
    - ._add_segments(IDs)
    - .delete_segments(IDs)


importer:
 
- knows its source folder, 
- keeps a list of imported files, so it can sync
    
BIGfiles:

- push data chunks -> Extractor

OR

- extract cues before data and use cues to scan file?



SlaveDatasets
-------------

wth do you need a SlaveDataset? Because one file can contain several data
types (e.g., EEG data and cues)

importer can import several 

    - as soon as files are added, the content of the importer should reflect that
    - data is imported on a lazy basis
    - segments constitute an ordered list
        


'''
# Created on Oct 20, 2010
# @author: christian

import importer
import operations as op
import collect
import visualizers
#import segments
import mat

from bionetics import Experiment, isexperiment, _extension
#from datasets_base import is_experiment_item
