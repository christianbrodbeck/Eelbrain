'''
Created on Jun 12, 2012

@author: Christian M Brodbeck
'''
import numpy as np

from eelbrain import ui
from eelbrain.vessels.data import var, dataset



__all__ = ['meg160_triggers', 'besa_edt']


def meg160_triggers(ds, dest=None):
    """
    Saves the time of all events in ds, plus 5 padding trials, to a txt file
    that can be used for epoching in MEG-160
    
    
    
    """
    T = ds['i_start'] / ds.info['samplingrate']
    
    # destination
    if dest is None:
        dest = ui.ask_saveas("Save MEG-160 Triggers", "Please pick a "
                             "destination for the MEG-160 triggers", 
                             ('txt', 'trigger-list'))
        if not dest:
            return
    
    # export trigger list
    a = np.ones(len(T) + 10) * 2
    a[5:-5] = T.x
    triggers = var(a, name='triggers')
    triggers.export(dest)



def besa_evt(ds, dest=None, sub=None, tstart=-0.1, tstop=0.6, pad=0.1):
    """
    ds needs to be the same dataset (i.e., same length) from which the MEG-160 
    triggers were created
    
    tstart, tstop and pad need to be the same values used in MEG-160 epoching
     
    """
    # destination
    if dest is None:
        dest = ui.ask_saveas("Save Besa Events", "Please pick a "
                             "destination for the Besa Events", 
                             ('evt', 'Besa Events'))
        if not dest:
            return
    
    # save trigger times in ds
    tstart2 = tstart - pad
    tstop2 = tstop + pad
    epoch_len = tstop2 - tstart2
    start = epoch_len * 5 - tstart2
    stop = epoch_len * (5 + ds.N)
    
    # BESA events
    evts = dataset()
    evts['Tsec'] = var(np.arange(start, stop, epoch_len))
    evts['Code'] = var(np.ones(ds.N))
    evts['TriNo'] = var(ds['eventID'].x)
    
    if sub is not None:
        if isinstance(sub, str):
            sub = ds[sub]
        evts = evts.subset(sub)
    
    evts.export(dest)
