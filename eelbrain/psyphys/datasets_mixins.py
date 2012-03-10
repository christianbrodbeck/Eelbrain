"""
Mixins for Datasets

"""
import os
import cPickle as pickle
from copy import deepcopy

import numpy as np

from eelbrain import ui


class Reject(object):
    def REJECTION_EXPORT(self, fn=None):
        if fn is None:
            msg = "Export rejection dictionary; WARNING: segment id is used as key, so the dictionary can only beused with the same segment sequence (=import procedure)"
            fn = ui.ask_saveas(message=msg,
                               ext = [('pickled', "pickled file")])
        if isinstance(fn, basestring):
            if fn[-8:] != '.pickled':
                fn = fn + '.pickled'
            # create segment-name - id table to assure same segment matching
            with open(fn, 'w') as f:
                out = deepcopy(self.reject_dict)
                out['safety'] = dict((s.id, s.name) for s in self.segments)
                pickle.dump(out, f)
    def import_rejection(self, fn=None):
        self.REJECTION_IMPORT(fn=fn)
    def REJECTION_IMPORT(self, fn=None):
        if fn is None:
            fn = ui.ask_file(wildcard = "pickled file (*.pickled)|*.pickled")
        if os.path.exists(fn):
            with open(fn, 'r') as f:
                rej = pickle.load(f)
            safety = rej.pop('safety')
            for s in self.segments:
                if s.name != safety[s.id]:
                    raise IOError("segment mismatch!")
            remove = []
            for id, d in rej.iteritems():
                try:
                    int(id)
                    if len(d) > 0:
                        self.delete_cache(id)
                except:
                    remove.append(id)
                    print "Dict contained invalid item: {k}: {v}".format(k=id, v=d)
            [rej.pop(id) for id in remove]
            self._reject_dict = rej
    @ property
    def reject_dict(self):
        """
        stores rejected intervals in t (seconds)
        """
        if not hasattr(self, '_reject_dict'):
            self._reject_dict = {}
        return self._reject_dict
    def reject_list_for_segment(self, segment):
        if not np.isscalar(segment):
            segment = segment.id
        if segment not in self.reject_dict:
            self.reject_dict[segment] = []
        return self.reject_dict[segment]
    def reject_tw(self, segment, t0, t1):
        if t1 < t0:
            t0, t1 = t1, t0
        self.reject_list_for_segment(segment).append([t0, t1])               
    def reject_remove(self, segment, t):
        reject_list = self.reject_list_for_segment(segment)
        x = np.array(reject_list)
        isin = (x[:,0]<t) * (x[:,1]>t)
        indexes = np.where(isin)
        for i in reversed(indexes):
            reject_list.pop(i)
    # check
    def reject_has_segment(self, segment):
        reject_list = self.reject_list_for_segment(segment)
        return len(reject_list) > 0
    def reject_check(self, segment, t, sr=None, tstart=0):
        """
        returns False if t is rejected, True otherwise
        
        sr=samplingrate (if t indicate samples rather than seconds)
        """
        reject_list = self.reject_list_for_segment(segment)
        x = np.array(reject_list)
        x0 = x[:, 0]
        x1 = x[:, 1]
        if np.isscalar(t):
            if np.any( (x0<t) * (x1>t) ):
                return False
            else:
                return True
        else:
            if t.ndim == 1:
                t = t[:, None]
            if sr:
                t = t / float(sr)
            if tstart:
                t = t + tstart
            indexes = -np.any((x0<t) * (t<x1), axis=1)
            return indexes
    def reject_flatten(self, segment, data, t):
        "flattens data in all TWs that are excluded"
        reject_list = self.reject_list_for_segment(segment)
        if reject_list:
            x = np.array(reject_list)
            x0 = x[:, 0]
            x1 = x[:, 1]
            t = t[:,None]
            indexes = np.any( (x0<=t) * (t<x1), axis=1)
            if indexes.ndim == 1:
                data[indexes] = 0
        return data
    # plotting
    def reject_toax(self, ax, segment, t0, t1, **axvspan_kwargs):
        "plots rejection TWs to ax"
        kwargs = {'ymax': 1,
                  'ec': 'r', 
                  'fc': 'r',
                  'hatch': '/'}
        kwargs.update(axvspan_kwargs)
        reject_list = self.reject_list_for_segment(segment)
        if reject_list:
            x = np.array(reject_list)
            x0 = x[:, 0]
            x1 = x[:, 1]
            indexes = (x1>t0) * (x0<t1)
            items = []
            for t0, t1 in x[indexes]:
                i = ax.axvspan(t0, t1, **kwargs)
                items.append(i)
            return items
        