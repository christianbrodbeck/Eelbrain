'''
Loaders for brain vision data format. Currently only vhdr header files can be
read.


Created on Jul 27, 2012

@author: Christian M Brodbeck
'''
import os
import re

import numpy as np

from ... import ui
from ..data_obj import Dataset, Factor, Var


__all__ = ['events', 'vhdr']


section_re = re.compile('^\[([ \w]+)\]')
#             <Type>,<Description>,<Position>,<Points>,<Channel Number>,<Date>
marker_re = re.compile('^Mk(\d+)=([ \w]+),(\w)(\d+),(\d+),(\d+),(\d+)', re.MULTILINE)

vhdr_hdr = 'Brain Vision Data Exchange Header File'
_vhdr_wildcard = ('Brain Vision Header File (*.vhdr)', '*.vhdr')


def events(vhdr_path=None):
    """Load events from a brainvision vhdr file.
    """
    if vhdr_path is None:
        filetypes = []
        vhdr_path = ui.ask_file("Pick a Brain Vision EEG Header File",
                                "Pick a Brain Vision EEG Header File",
                                [_vhdr_wildcard])
        if not vhdr_path:
            return

    hdr = vhdr(vhdr_path)
    if hdr.markerfile is None:
        raise IOError("No marker file referenced in %r" % vhdr_path)
    elif hdr.DataType == 'FREQUENCYDOMAIN':
        raise NotImplementedError

    txt = open(hdr.markerfile).read()
    m = marker_re.findall(txt)
    m = np.array(m)

    name, _ = os.path.split(os.path.basename(hdr.path))
    ds = Dataset(name=name)
    ds['Mk'] = Var(np.array(m[:, 0], dtype=int))
    ds['event_type'] = Factor(m[:, 1])
    ds['trigger'] = Var(np.array(m[:, 3], dtype=int))
    ds['i_start'] = Var(np.array(m[:, 4], dtype=int))
    ds['points'] = Var(np.array(m[:, 5], dtype=int))
    ds['channel'] = Var(np.array(m[:, 6], dtype=int))

    ds.info['hdr'] = hdr
    return ds


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
                               [_vhdr_wildcard])
            if not path:
                raise RuntimeError("User Canceled")

        kv_re = re.compile('^(\w+)=(.+)', re.MULTILINE)

        hdr_section = {}
        with open(path) as FILE:
            if not FILE.readline().startswith(vhdr_hdr):
                err = ("Not a brain vision vhdr file: %r (needs to start with"
                       " %r" % (path, vhdr_hdr))
                raise IOError(err)
            for line in FILE:
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

        # paths
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

        # other info
        self.DataFormat = info.get('DataFormat', 'ASCII')
        self.DataOrientation = info.get('DataOrientation', 'MULTIPLEXED')
        self.DataType = info.get('DataType', 'TIMEDOMAIN')
        self.NumberOfChannels = int(info['NumberOfChannels'])
        self.samplingrate = 1e6 / float(info['SamplingInterval'])

    def __repr__(self):
        return "vhdr(%r)" % self.path

