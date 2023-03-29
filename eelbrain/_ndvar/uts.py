# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from math import ceil, floor

import numpy

from .._data_obj import NDVar, UTS
from .._utils.numpy_utils import index


def pad(
        ndvar: NDVar,
        tstart: float = None,
        tstop: float = None,
        nsamples: int = None,
        set_tmin: bool = False,
        name: str = None,
) -> NDVar:
    """Pad (or crop) an NDVar in time

    Parameters
    ----------
    ndvar
        NDVar to pad.
    tstart
        New tstart.
    tstop
        New tstop.
    nsamples
        New number of samples.
    set_tmin
        Reset ``tmin`` to be exactly equal to ``tstart``.
        By default, the new ``tmin`` is inferred from the time-step, i.e. the
        sample at which ``time == 0`` is left exactly unchanged. With
        ``set_tmin``, the time point at which ``time == 0`` might shift slightly
        if the amount of time added is not an exact multiple of the time step.
    name
        Name for the new NDVar.
    """
    axis = ndvar.get_axis('time')
    time: UTS = ndvar.dims[axis]
    if name is None:
        name = ndvar.name
    # start
    if tstart is None:
        if nsamples is not None:
            raise NotImplementedError("nsamples without tstart")
        n_add_start = 0
    elif tstart < time.tmin:
        n_add_start = int(ceil((time.tmin - tstart) / time.tstep))
    elif tstart > time.tmin:
        n_add_start = -time._array_index(tstart)
    else:
        n_add_start = 0

    # end
    if nsamples is None and tstop is None:
        n_add_end = 0
    elif nsamples is None:
        n_add_end = int(floor((tstop - time.tstop) / time.tstep + 1e-5))
    elif tstop is None:
        n_add_end = nsamples - n_add_start - time.nsamples
    else:
        raise TypeError("Can only specify one of tstart and nsamples")
    # need to pad?
    if not n_add_start and not n_add_end:
        return ndvar
    # construct padded data
    xs = [ndvar.x]
    shape = ndvar.x.shape
    # start
    if n_add_start > 0:
        shape_start = shape[:axis] + (n_add_start,) + shape[axis + 1:]
        xs.insert(0, numpy.zeros(shape_start))
    elif n_add_start < 0:
        xs[0] = xs[0][index(slice(-n_add_start, None), axis)]
    # end
    if n_add_end > 0:
        shape_end = shape[:axis] + (n_add_end,) + shape[axis + 1:]
        xs += (numpy.zeros(shape_end),)
    elif n_add_end < 0:
        xs[-1] = xs[-1][index(slice(None, n_add_end), axis)]
    x = numpy.concatenate(xs, axis)
    if n_add_start == 0:
        new_tmin = time.tmin
    elif set_tmin:
        new_tmin = tstart
    else:
        new_tmin = time.tmin - (time.tstep * n_add_start)
    new_time = UTS(new_tmin, time.tstep, x.shape[axis])
    dims = (*ndvar.dims[:axis], new_time, *ndvar.dims[axis + 1:])
    return NDVar(x, dims, name, ndvar.info)
