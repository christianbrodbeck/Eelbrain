"""
Every cache manager class provides the same interface for string and retrieving
cache data. They can thus be used interchangeably depending on a Dataset;s
requirements.


Interface
=========

The current interface uses the following methods:

__init__(experiment)
    construct with the parent experiment

__setitem__(ID, data), __getitem__(ID), __delitem__(IDs)
    data is accessed like a dictionary, with an ID, a list of IDs or None
    (=all)

write_cache()
    write all the cached data to disk

close()
    close the associated file


The Memmap Implementation
-------------------------

the _memmap_mgr class can be used to manage a harddrive data cache of a sequence 
of arrays who agree on their data.shape[1:]

the _memmap_mgr creates _mm_file instance dynamically to create actual files on the 
hard drive.


Ideas
=====

file path handling
^^^^^^^^^^^^^^^^^^

purpose: get rid of store lugging around a reference to the experiment
new methods set_cache_path(path).

"""

import os, logging

import numpy as np



class _mm_file:
    """
    manages one np.memmap file
    
    Attributes
    ----------
    mod_time:   timestamp the file was last modify (to check upon opening 
                whether the file is still valid)
    """
    def __init__(self, name, shape, dtype, experiment):
        logging.debug(" <mm_file>: '{n}' __init__()".format(n=name))
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.experiment = experiment
        self.open(mode='create')
    @property
    def path(self):
        filepath = os.path.join(self.experiment._cache_path, 
                                self.name + '.npy')
        return filepath
    def open(self, mode='read'):
        "returns True if successful, False if the file is outdated"
        if not hasattr(self, 'file'):
            path = self.path
            if mode == 'read':
                mod_time = os.path.getmtime(path)
                if mod_time != self.mod_time:
                    msg = " <mm_file>: {n} cache file '{p}' was outdated"
                    logging.info(msg.format(n=self.name, p=path))
                    return False
                fmode = 'r+'
            elif mode == 'create':
                fmode = 'w+'
                self.mod_time = None
            else: #if mode == 'write':
                raise NotImplementedError
            
            logging.debug(" <mm_file>: {n} open({m}), shape={s}".format(n=self.name, 
                          m=mode, s=str(self.shape)))
            self.file = np.memmap(path, dtype=self.dtype, shape=self.shape,
                                  mode=fmode)
        return True
    def close(self):
        if hasattr(self, 'file'):
            logging.debug(" <mm_file>: '{n}' close()".format(n=self.name))
            self.file.flush()
            del self.file
            if self.mod_time == None:
                self.mod_time = os.path.getmtime(self.path)
    def write(self, start, end, data):
        self.file[start:end] = data
    def get(self, start, end):
        if self.open(mode='read'):
            return self.file[start:end]
        else:
            return None
    def delete(self):
        self.close()
        try:
            os.remove(self.path)
        except:
            logging.warning(" Could not remove cache file {0}".format(self.path))

    
class _memmap_mgr:
    max_size = 2.**29 # 30 fails
    def __init__(self, shape, dtype, experiment):
        self.shape = shape
        self.dtype = dtype
        self.experiment = experiment
        self.ID = experiment._get_memmap_id()
        ## determine block length
        if type(dtype) is np.dtype:
            dsize = dtype.itemsize
        else:
            dsize = dtype().itemsize
        x = self.max_size / dsize
        for i in shape:
            x /= i
        self.max_len = int(x)
        logging.debug(" FILE maxlen: %s" % str(self.max_len))
        ## other
        self.data = {}
        self.cache = {}
        self.cache_len = 0
        self.reserved = {}
        self.addresses = {} # id:(file_ID, start, end)
        self.files = []
    def get(self, ID):#, index):
        if len(self.cache):
            self.write_cache()
        file_ID, start, end = self.addresses[ID]
        file = self.files[file_ID]
        return file.get(start, end)
        """
        if type(index) is tuple:
            i = index[0]
        else:
            i = index
        if type(i) == int:
        data = self.files[file_ID]
        """
    def set(self, ID, data):
        "ID needs to be unique!"
        data_len = data.shape[0]
        # assert that data has the right shape 
        if data.shape[1:] != self.shape:
            txt = "memmap_mgr dropped data because of shape mismatch ({ss} / {os})"
            logging.error(txt.format(ss=str(self.shape), os=str(data.shape)))
        # if ID exists, assert length is correct and write
        elif ID in self.addresses:
            f_ID, start, end = self.addresses[ID]
            assert data_len == end - start, "data does not fit shape"
            file = self.files[f_ID]
            file.write(start, end, data)
        # else append to cache
        else:
            if self.cache_len + data_len > self.max_len:
                self.write_cache()
            self.cache[ID] = data
            self.cache_len += data_len
    def write_cache(self):
        if len(self.cache) > 0:
            # create file
            f_ID = len(self.files)
            f_len = sum([data.shape[0] for data in self.cache.values()])
#            print f_len, self.cache_len
            f_name = '_'.join([str(self.ID), str(f_ID)])
            file = _mm_file(f_name, (f_len,)+self.shape, self.dtype, self.experiment)
            self.files.append(file)
            # write the cache to the file
            pos = 0
            for ID, data in self.cache.iteritems():
                start = pos
                pos += data.shape[0]
                end = pos
#                logging.debug(" {s}-{e}, len {l}, {sha}".format(s=start, e=end, 
#                                                                l=end-start,
#                                                                sha=str(data.shape)))
                file.write(start, end, data)
                self.addresses[ID] = (f_ID, start, end)
            file.close()
            self.cache = {}
            self.cache_len = 0
    def close(self):
        for f in self.files:
            f.close()
    def delete(self):
        for f in self.files:
            f.delete()



class Memmap(object):
    """
    TODO: make resizable; use 
    
    http://mail.scipy.org/pipermail/numpy-discussion/2010-June/050742.html
    
    Hi again,
    To answer to the second part of my question, here follows an example
    demonstrating how to "resize" a memmap:
    
    >>> fp = numpy.memmap('test.dat', shape=(10,), mode='w+')
    >>> fp._mmap.resize(11)
    >>> cp = numpy.ndarray.__new__(numpy.memmap, (fp._mmap.size(),), dtype=fp.dtype, buffer=fp._mmap, offset=0, order='C')
    >>> cp[-1] = 99
    >>> cp[1] = 33
    >>> cp
        memmap([ 0, 33,  0,  0,  0,  0,  0,  0,  0,  0, 99], dtype=uint8)
    >>> fp
        memmap([ 0, 33,  0,  0,  0,  0,  0,  0,  0,  0], dtype=uint8)
    >>> del fp, cp
    >>> fp = numpy.memmap('test.dat', mode='r')
    >>> fp
        memmap([ 0, 33,  0,  0,  0,  0,  0,  0,  0,  0, 99], dtype=uint8)
    
    Would there be any interest in turning the above code to numpy.memmap
    method, say, to resized(newshape)? For example, for resolving the original
    problem, one could have
    
    fp = numpy.memmap('test.dat', shape=(10,), mode='w+')
    fp = fp.resized(11)
    
    Regards,
    Pearu

    
    """
    def __init__(self, experiment):
        self.cached = set()
        self.experiment = experiment
    @property
    def empty(self):
        return not hasattr(self, '_memmap_mgr')
    @property
    def memmap_mgr(self):
        if not hasattr(self, '_memmap_mgr'):
            self._memmap_mgr = _memmap_mgr(self.shape, self.dtype, self.experiment)
        return self._memmap_mgr
    def __contains__(self, ID):
        return ID in self.cached
    def __setitem__(self, ID, data):
        if self.empty:
            self.shape = data.shape[1:]
            self.dtype = data.dtype
        else:
            assert data.shape[1:] == self.shape
            assert data.dtype == self.dtype
        self.memmap_mgr.set(ID, data)
        self.cached.add(ID)
    def write_cache(self):
        if hasattr(self, '_memmap_mgr'):
            self._memmap_mgr.write_cache()
    def __getitem__(self, ID):#, index):
        if ID in self.cached:
            return self.memmap_mgr.get(ID)
        else:
            raise KeyError("Item %r not cached!" % ID)
    def __delitem__(self, IDs):
        "IDs: list"
        if IDs:
            for ID in IDs:
                if ID in self.cached:
                    self.cached.remove(ID)
                    # memmap mgr -> requires that new shape is == old shape
        else:
            self.cached = set()
            if hasattr(self, 'shape'):
                del self.shape
            if hasattr(self, 'dtype'):
                del self.dtype
            if hasattr(self, '_memmap_mgr'):
                self._memmap_mgr.delete()
                del self._memmap_mgr
    def close(self):
        if hasattr(self, '_memmap_mgr'):
            self._memmap_mgr.close()

    def __del__(self):
        logging.debug("MemmapStore.__del__() called")
        self._clear_cache(None)
    
        



class Local(dict):
    def __init__(self, experiment):
        dict.__init__(self)
    def __delitem__(self, IDs):
        if IDs is None:
            IDs = self.keys()
        for ID in IDs:
            if ID in self:
                dict.__delitem__(self, ID)
    def write_cache(self):
        pass
    def close(self):
        pass


#"""
#UNFINISHED implementation of storage based on pytables
#"""
#try:
#    import tables
#    
#    class TablesStore(object):
#        def __init__(self, experiment, cache_limit=2**30):
#            self._cache = {}   # the cache (keep data in RAM)
#            self._shape = None # the data shape
#            self._table = None # the table object
#        
#        def __contains__(self, ID):
#            a = ID in self._cache
#            b = ID in self._table_locs
#            return a or b
#        
#        def __setitem__(self, ID, data):
#            if self._shape:
#                assert self._shape == data.shape[1:]
#            else:
#                self._shape = data.shape[1:]
#            
#            self._cache[ID] = data
#            
#            cache_size = sum(np.prod(data.shape) for ID, data in self._cache)
#            if cache_size > self._cache_limit:
#                self.write_cache()
#            
#        def __getitem__(self, ID):
#            if ID in self._cache:
#                return self._cache[ID]
#            elif ID in self._table_locs:
#                a, b = self._table_locs[ID]
#                return self._table[a:b]
#        
#except:
#    logging.info("import tables failed; TablesStore not available")
