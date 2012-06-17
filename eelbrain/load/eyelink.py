'''
Created on Jun 9, 2012

@author: christian
'''
import fnmatch
import os

import numpy as np

from eelbrain import ui
from eelbrain.utils import subp
from eelbrain.vessels.data import dataset, var

__all__ = ['Edf']



class Edf(object):
    """
    Class for reading an eyelink .edf file and extracting acceptability
    based on contamination with ocular artifacts (saccades and blinks)
    
    **Methods:**
    
    add_by_Id:
        Add acceptability to a dataset based on matching event-Ids
    add_by_T:
        Add acceptability to a dataset based on time values (in the edf file's 
        timing)
    get_accept:
        returns acceptability for a list of time points
    get_triggers:
        returns all trigger events found in the .edf file (Id and time)
    
    """
    def __init__(self, path=None):
        """
        loads events in an edf file and can add them to datasets
        
        path : str(path) | None
            Path to the .edf file. The If path contains '*', the files matching
            the pattern are concatenated. If None, a file-open dialogue will be
            displayed.

        """
        if path is None:
            path = ui.ask_file("Load an eyelink .edf file", "Pick the edf file",
                               ext=[('edf', 'eyelink data format')])
        
        # find all paths from which to read
        self.path = path
        if '*' in path:
            head, tail = os.path.split(path)
            if '*' in head:
                err = ("Invalid path: %r. All edf files need to be in the same "
                       "directory." % path)
                raise ValueError(err)
            
            fnames = sorted(fnmatch.filter(os.listdir(head), tail))
            self.paths = [os.path.join(head, fname) for fname in fnames]
        else:
            self.paths = [path]
        
        triggers = []
        artifacts = []
        for path in self.paths:
            edf = subp.edf_file(path)
            triggers += edf.triggers
            artifacts += edf.artifacts
        
        dtype = [('T', np.uint32), ('Id', np.uint8)]
        self.triggers = np.array(triggers, dtype=dtype)
        dtype = np.dtype([('event', np.str_, 5), ('start', np.uint32), ('stop', np.uint32)])
        self.artifacts = np.array(artifacts, dtype=dtype)
    
    def __repr__(self):
        return "Edf(%r)" % self.path
    
    def _assert_Id_match(self, Id):
        ID_edf = self.triggers['Id']
        if len(Id) != len(ID_edf):
            lens = (len(Id), len(ID_edf))
            mm = min(lens)
            for i in xrange(mm):
                if Id[i] != ID_edf[i]:
                    mm = i
                    break
            
            args = lens + (mm,)
            err = ("dataset containes different number of events from edf file "
                   "(%i vs %i); first mismatch at %i." % args)
            raise ValueError(err)
        
        check = (Id == ID_edf)
        if not all(check):
            err = "Event ID mismatch: %s" % np.where(check==False)[0]
            raise ValueError(err)
    
    def add_by_Id(self, ds, tstart=-0.1, tstop=0.6, Id='eventID',
               target='accept', reject=False, accept=None):
        """
        Mark each epoch in the ds for acceptability based on overlap with 
        blinks and saccades. ds needs to contain exactly the same triggers
        as the edf file. For adding acceptability to a decimated ds, use 
        Edf.add_T_by_Id() and then Edf.add_by_T().
        
        dataset : dataset
            dataset that contains the data to work with.
        start : scalar
            start of the time window relevant for rejection. 
        stop : scalar
            stop of the time window relevant for rejection.
        reject : 
            value that is assigned to epochs that should be rejected based on 
            the eye-tracker data.
        accept :
            value that is assigned to epochs that can be accepted based on 
            the eye-tracker data.
        
        """        
        self._assert_Id_match(ds[Id])

        if isinstance(target, str):
            if target not in ds:
                ds[target] = var(np.ones(ds.N, dtype=np.bool_))
            target = ds[target]
        
        target.x *= self.get_accept(tstart=tstart, tstop=tstop)
    
    def add_by_T(self, ds, tstart=-0.1, tstop=0.6, T='t_edf', target='accept'):
        "adds acceptability to a dataset based on edf-time values"
        ds[target] = self.get_accept(T=ds[T], tstart=tstart, tstop=tstop)
    
    def add_T_by_Id(self, ds, Id='eventID', t_edf='t_edf'):
        """
        Asserts that trigger events in the dataset match trigger events in the 
        edf file, and adds edf trigger time to the ds. This can be used for
        Edf.add_by_T(ds) after ds hads been decimated.
         
        """
        self._assert_Id_match(ds[Id])
        ds[t_edf] = var(self.triggers['T'])
    
    def get_accept(self, T=None, tstart=-0.1, tstop=0.6):
        """
        returns a boolean var indicating for each epoch whether it should be 
        accepted or not based on ocular artifacts in the edf file.
        
        T : array-like | None
            List of time points (in the edf file's time coordinates). If None, 
            the edf's trigger events are used.
        tstart : scalar
            start of the epoch relative to the event (in seconds)
        tstop : scalar
            end of the epoch relative to the even (in seconds)
        
        """
        if T is None:
            T = self.triggers['T']
        
        # conert to ms
        start = int(tstart * 1000)
        stop = int(tstop * 1000)
        
        self._debug = []
        
        # get data for triggers
        N = len(T)
        accept = np.empty(N, np.bool_)
        for i, t in enumerate(T):
            starts_before_tstop = self.artifacts['start'] < t + stop
            stops_after_tstart = self.artifacts['stop'] > t + start
            overlap = np.all((starts_before_tstop, stops_after_tstart), axis=0)
            accept[i] = not np.any(overlap)
            
            self._debug.append(overlap)
        
        return accept
    
    def get_triggers(self, Id='Id', T='t_edf'):
        ds = dataset()
        ds[Id] = var(self.triggers['Id'])
        ds[T] = var(self.triggers['T'])
        return ds


