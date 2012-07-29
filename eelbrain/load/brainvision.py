'''
Loaders for brain vision data format. Currently only vhdr header file can be 
read.


Created on Jul 27, 2012

@author: Christian M Brodbeck
'''
import os
import re

from eelbrain import ui


class vhdr(dict):
    def __init__(self, path=None):
        """
        Reads a .vhdr file and returns its content as a dictionary
        
        The .markerfile and .datafile attributes provide absolute paths to those
        files.
        
        """
        if path is None:
            path = ui.ask_file("Pick a Brain Vision EEG Header File", 
                                    "Pick a Brain Vision EEG Header File",
                                    ext=[('vhdr', 'Brain Vision Header File')])
            if not path:
                raise RuntimeError("User Canceled")

        section_re = re.compile('^\[([ \w]+)\]')
        kv_re = re.compile('^(\w+)=(.+)', re.MULTILINE)
        
        hdr_section = {}
        for line in open(path):
            m = section_re.match(line)
            if m:
                name = m.group(1)
                hdr_section = self[name] = {}
            else:
                m = kv_re.match(line)
                if m:
                    key = m.group(1)
                    value = m.group(2).strip()
                    hdr_section[key] = value
        
        self.path = path
        info = self['Common Infos']
        self.dirname, self.basename = os.path.split(path)
        
        datafile = info['DataFile']
        if os.path.isabs(datafile):
            self.datafile = datafile
        else:
            self.datafile = os.path.join(self.dirname, datafile)
        
        if 'MarkerFile' in info:
            markerfile = info['MarkerFile']
            if os.path.isabs(markerfile):
                self.markerfile = markerfile
            else:
                self.markerfile = os.path.join(self.dirname, markerfile)
        else:
            self.markerfile = None
    
    def __repr__(self):
        return "vhdr(%r)" % self.path

