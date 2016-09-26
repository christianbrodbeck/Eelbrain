"""NDVar operations"""
import numpy as np

from ._data_obj import NDVar, UTS


def concatenate(ndvars, dim='time', name=None):
    """Concatenate multiple NDVars

    Parameters
    ----------
    ndvars : NDVar | sequence of NDVar
        NDVars to be concatenated. Can also be a single NDVar with ``case``
        dimension to concatenate the different cases.
    dim : str
        Dimension along which to concatenate (only 'time' and 'case' are
        implemented).
    name : str (optional)
        Name the NDVar holding the result.

    Returns
    -------
    ndvar : NDVar
        NDVar with concatenated data. For ``dim='time'``, the output time axis
        starts at t=0.
    """
    ndvar = ndvars[0]
    axis = ndvar.get_axis(dim)
    dim_names = ndvar.get_dimnames((None,) * axis + (dim,) +
                                   (None,) * (ndvar.ndim - axis - 1))
    x = np.concatenate([v.get_data(dim_names) for v in ndvars], axis)
    if dim == 'time':
        out_dim = UTS(0, ndvar.time.tstep, x.shape[axis])
    elif dim == 'case':
        out_dim = 'case'
    else:
        raise NotImplementedError("dim=%s is not implemented; only 'time' and "
                                  "'case' are implemented" % repr(dim))
    dims = ndvar.dims[:axis] + (out_dim,) + ndvar.dims[axis + 1:]
    return NDVar(x, dims, {}, name or ndvar.name)
