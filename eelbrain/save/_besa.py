'''
Export events for use in the Besa pipeline. Use :func:`meg160_triggers` to
export a trigger list for MEG160, then reject unwanted events, and finally
use :func:`besa_evt` to export a corresponding .evt file.

'''
import numpy as np

from eelbrain import ui
from eelbrain.vessels.data import var, dataset

import _txt



__all__ = ['meg160_triggers', 'besa_evt']


def meg160_triggers(ds, dest=None, pad=1):
    """Export a list of event times used for epoching in MEG-160

    For use together with :func:`besa_evt`. ``save.meg160_triggers(ds)`` adds
    a variable called 'besa_index' which keeps track of the event sequence so
    that the .evt file can be saved correctly after event rejection.

    Parameters
    ----------
    ds : dataset
        Dataset containing all the desired events.
    dest : None | str
        Path where to save the triggers (if None, a dialog will be displayed).
    pad : int
        Number of epochs to pad with at the beginning and end of the file.
    """
    sfreq = ds.info['raw'].info['sfreq']
    times = ds['i_start'] / sfreq

    # destination
    if dest is None:
        dest = ui.ask_saveas("Save MEG-160 Triggers", "Please pick a "
                             "destination for the MEG-160 triggers",
                             ('txt', 'trigger-list'))
        if not dest:
            return

    # make trigger list with padding
    a = np.ones(len(times) + pad * 2) * times[0]
    a[pad:-pad] = times.x
    triggers = var(a, name='triggers')

    # export trigger list
    _txt.txt(triggers, dest=dest)

    # index the datafile to keep track of rejections
    ds.index('besa_index', pad)


def besa_evt(ds, tstart= -0.1, tstop=0.6, pad=0.1, dest=None):
    """Export an evt file for use with Besa

    For use together with :func:`meg160_triggers`

    Parameters
    ----------
    ds : dataset
        Dataset containing the events.
    tstart, tstop : scalar [sec]
        start and stop of the actual epoch.
    pad : scalar [sec]
        Time interval added before and after every epoch for padding.
    dest : None | str
        Destination file name. If None, a save as dialog will be shown.

    Notes
    -----
    ds needs to be the same dataset (i.e., same length) from which the MEG-160
    triggers were created.

    tstart, tstop and pad need to be the same values used for epoching in
    MEG-160.
    """
    idx = ds['besa_index']
    N = idx.x.max() + 1

    # save trigger times in ds
    tstart2 = tstart - pad
    tstop2 = tstop + pad
    epoch_len = tstop2 - tstart2
    start = -tstart2
    stop = epoch_len * N

    # BESA events
    evts = dataset()
    evts['Tsec'] = var(np.arange(start, stop, epoch_len))
    evts['Code'] = var(np.ones(N))

    # remove rejected trials
    evts = evts.subset(idx)

    evts['TriNo'] = var(ds['eventID'].x)

    # destination
    if dest is None:
        dest = ui.ask_saveas("Save Besa Events", "Please pick a "
                             "destination for the Besa events",
                             ('evt', 'Besa Events'))
        if not dest:
            return

    evts.export(dest)
