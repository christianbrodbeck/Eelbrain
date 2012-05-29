"""

Provides shortcuts for constructing displays from the shell.

"""

import wx

from eelbrain.psyphys import operations as op
from eelbrain.psyphys import visualizers as _V

# Viewers
import vw_physio as _vw_physio
import vw_list as _vw_list
import vw_guis as _vw_guis 
import experiment_frame as _eframe


__all__ = ['gui', 'list', 'physio']



def frame_experiment(e):
    "Creates a WxPython frame for the experiment"
    parent = wx.GetApp().shell
    frame = _eframe.ExperimentFrame(e, parent)
    return frame


def _shell():
    return wx.GetApp().shell


def _auto(datasets, v_type=0, y=5, debug=True, dur=None, mag=None, a=None, **kwargs):
    """
    v_type: 0  physio
            1  list
    
    a: address (to select subset of segments in the datasets)
    
    """
    if debug:
#        reload(_vw_physio)
#        reload(V)
        reload(_vw_list)
    
    v = []
    for ds in datasets:
        new_v = _V.default(ds, a=a, dur=dur, mag=mag)
        v.append(new_v)
    
    if v_type == 0:
        w, h = wx.GetDisplaySize()
        return _vw_physio.PhysioViewerFrame(_shell(), v, size=(w, 450), **kwargs)
    elif v_type == 1:
        return _vw_list.ListViewerFrame(v, y=y, **kwargs)
    else:
        raise ValueError("v_type %s not implemented"%v_type)


def list(*datasets, **kwargs):
    """
    Create a list viewer for one or more datasets. Kwargs:
    
    y : int
        Number of plots per page. Default is 5.
    
      
    
    """
    return _auto(datasets, v_type=1, **kwargs)

def physio(*datasets, **kwargs):
    "Create a physio viewer for one or more datasets"
    return _auto(datasets, v_type=0, **kwargs)


def gui(dataset, **kwargs):
    """
    Finds and initiates the right gui for the dataset.
    
    """
    if isinstance(dataset, op.base.crop_uts):
        return _vw_guis.CropGUI(_shell(), dataset, **kwargs)
    elif isinstance(dataset, op.physio.HeartBeat):
        return physio(dataset.parent, dataset, **kwargs)
    elif isinstance(dataset, op.physio.SCR):
        return physio(dataset.parent, dataset, **kwargs)
    else:
        return list(dataset, **kwargs)
#        raise NotImplementedError("No GUI for %r" % type(dataset))
