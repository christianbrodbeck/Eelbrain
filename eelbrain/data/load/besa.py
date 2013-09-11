'''
Loading data from the besa-mn pipeline
'''
# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
from __future__ import division

import numpy as np
from scipy.io import loadmat

from ... import ui
from ..data_obj import Dataset, Factor, NDVar, UTS


def roi_results(path=None, varname=None):
    """
    Load the meg data from a saved besa-mn ROI results object

    Parameters
    ----------
    path : str | None
        Path to the *.m file containing the saved results. If None, a file
        can be selected using a system file dialog.
    varname : str | None
        If the .m file contains more than one variable, the name of the
        variable containing the results can be specified as string.

    Returns
    -------
    besa_mn_data : Dataset
        A Dataset containing the following variables: 'meg' the meg ROI
        waveforms; 'condition' the condition label; 'subject' a subject
        identifier.
    """
    if path is None:
        path = ui.ask_file("Pick a Matlab file with Besa-MN results",
                           "Pick a Matlab File",
                           ext=[('mat', 'Matlab data file')])
        if not path:
            return

    m = loadmat(path)
    if varname is None:
        keys = [k for k in m.keys() if not k.startswith('_')]
        if len(keys) == 0:
            raise ValueError("Noe data in .mat file")
        elif len(keys) == 1:
            varname = keys[0]
        else:
            err = ("More than one variables in .mat file. Use the varname "
                   "parameter.")
            raise ValueError(err)
    res = m[varname]

    ds = Dataset()

    n_c, n_s, n_t = res['waves'][0][0].shape
    x = res['waves'][0][0].reshape(n_c * n_s, n_t)
    times = np.ravel(res['latencies'][0][0])
    time = UTS(times[0] / 1000, .001, len(times))
    ds['meg'] = NDVar(x, dims=('case', time))

    conds = [c[0] for c in res['conditions'][0][0][0]]
    ds['condition'] = Factor(conds, rep=n_s)

    ds['subject'] = Factor(map(str, xrange(n_s)), tile=n_c, random=True)

    return ds
