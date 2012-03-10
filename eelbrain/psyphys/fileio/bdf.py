'''
Read biosemi data format (bdf) files

After http://www.biosemi.com/faq/file_format.htm


Created on Dec 27, 2010
@author: Christian Brodbeck
'''
from __future__ import division

import numpy as np

class bdf:
    def __init__(self, filename):
        "reads the bdf header"
        # read header
        f = open(filename)
        self.hdr = dict(
                        hdr = f.read(8),
                        s_id = f.read(80),
                        rec_id = f.read(80),
                        rec_date = f.read(8),
                        rec_time = f.read(8),
                        n_bytes_in_hdr = int(f.read(8)),
                        data_format_version = f.read(44),
                        
                        )
        
        # Duration of a data record, in seconds
        self.records_N = int(f.read(8))
        self.record_duration = float(f.read(8))
        
        # Channels
        N = self.chan_N = int(f.read(4)) # n channels
        self.chan_names = [f.read(16) for i in xrange(N)]
        self.chan_types = [f.read(80) for i in xrange(N)]
        self.chan_unit = [f.read(8) for i in xrange(N)]
        
        # read channel gain information
        self.chan_pmin = pmin = np.array([int(f.read(8)) for i in xrange(N)])
        self.chan_pmax = pmax = np.array([int(f.read(8)) for i in xrange(N)])
        self.chan_dmin = dmin = np.array([int(f.read(8)) for i in xrange(N)])
        self.chan_dmax = dmax = np.array([int(f.read(8)) for i in xrange(N)])
        self.gain = (pmax-pmin) / (dmax-dmin)
        
        
        self.chan_prefiltering = [f.read(80) for i in xrange(N)]
        
        self.chan_samples_per_record = [float(f.read(8)) for i in xrange(N)]
        
        trash = f.read
        
        
        self._data_start = f.tell() + N * 32
        self.file = f
    def get_hdr(self):
        "compile header dictionary with useful information"
        samplingrate = self.get_samplingrate()
        hdr = dict(samplingrate = samplingrate,
                   n_channels = self.chan_N,
                   length = samplingrate*self.record_duration*self.records_N
                   )
        return hdr
    def get_samplingrate(self):
        "raises an error if channels are sampled at different rates"
        u = np.unique(self.chan_samples_per_record)
        if len(u) > 1:
            raise NotImplementedError("unequal samplingrate per channel")
        else:
            return u[0] 
    def read_fast(self):
        """
        Read the bdf file with just one call to np.fromfile
        
        """
        samplingrate = self.get_samplingrate()
                
        # read file
        self.file.seek(self._data_start)
        int_rows = np.fromfile(self.file, dtype=np.uint8, count=-1)
        int_rows = int_rows.reshape((-1, 3)).astype(np.int32)
        
        # -> 32 bit numpy array, after
        # http://code.google.com/p/psychicml/source/browse/psychic/bdfreader.py
        ints = int_rows[:, 0] + (int_rows[:, 1] << 8) + (int_rows[:, 2] << 16)
        ints[ints >= (1 << 23)] -= (1 << 24)
        
        # (data_blocks, channels, t) --> data spreads along innermost axis
        samples_in_record = int(self.record_duration * samplingrate)
        shape = (self.records_N, self.chan_N, samples_in_record)
        ints = ints.reshape(shape)
        
        # shape of target array
        t = samples_in_record * self.records_N
        shape = (t, self.chan_N)
        out = np.empty(shape)
        
        # blocks 
        for chan in xrange(self.chan_N):
            out[:,chan] = np.ravel(ints[:,chan,:])
        
        return out * self.gain
        
    def __del__(self):
        self.file.close()
        