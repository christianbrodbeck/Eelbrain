import numpy as np
from numpy import newaxis

from .. import _colorspaces as cs
from .._data_obj import NDVar, UTS, dataobj_repr


class RevCorrData(object):
    """Restructure input NDVars into arrays for reverse correlation
    
    Attributes
    ----------
    y : array  (n_y, n_times)
        Dependent variable.
    x : array  (n_x, n_times)
        Predictors.
    """
    def __init__(self, y, x, error, scale_data):
        # scale_data param
        if isinstance(scale_data, bool):
            scale_in_place = False
        elif isinstance(scale_data, str):
            if scale_data == 'inplace':
                scale_in_place = True
            else:
                raise ValueError("scale_data=%r" % (scale_data,))
        else:
            raise TypeError("scale_data=%r, need bool or str" % (scale_data,))

        # check y and x
        if isinstance(x, NDVar):
            x_name = x.name
            x = (x,)
            multiple_x = False
        else:
            x = tuple(x)
            assert all(isinstance(x_, NDVar) for x_ in x)
            x_name = tuple(x_.name for x_ in x)
            multiple_x = True
        time_dim = y.get_dim('time')
        if any(x_.get_dim('time') != time_dim for x_ in x):
            raise ValueError("Not all NDVars have the same time dimension")

        # y_data:  ydim x time array
        if y.ndim == 1:
            ydims = ()
            y_data = y.x[None, :]
        else:
            dimnames = y.get_dimnames(last='time')
            ydims = y.get_dims(dimnames[:-1])
            y_data = y.get_data(dimnames).reshape((-1, len(y.time)))

        # x_data:  predictor x time array
        x_data = []
        x_meta = []
        x_names = []
        n_x = 0
        for x_ in x:
            if x_.ndim == 1:
                xdim = None
                data = x_.x[newaxis, :]
                index = n_x
                x_names.append(dataobj_repr(x_))
            elif x_.ndim == 2:
                xdim = x_.dims[not x_.get_axis('time')]
                data = x_.get_data((xdim.name, 'time'))
                index = slice(n_x, n_x + len(data))
                x_repr = dataobj_repr(x_)
                for v in xdim:
                    x_names.append("%s-%s" % (x_repr, v))
            else:
                raise NotImplementedError("x with more than 2 dimensions")
            x_data.append(data)
            x_meta.append((x_.name, xdim, index))
            n_x += len(data)

        if len(x_data) == 1:
            x_data = x_data[0]
            x_is_copy = False
        else:
            x_data = np.vstack(x_data)
            x_is_copy = True

        if scale_data:
            if not scale_in_place:
                y_data = y_data.copy()
                if not x_is_copy:
                    x_data = x_data.copy()
                    x_is_copy = True

            y_mean = y_data.mean(1)
            x_mean = x_data.mean(1)
            y_data -= y_mean[:, newaxis]
            x_data -= x_mean[:, newaxis]
            if error == 'l1':
                y_scale = np.abs(y_data).mean(1)
                x_scale = np.abs(x_data).mean(1)
            elif error == 'l2':
                y_scale = y_data.std(1)
                x_scale = x_data.std(1)
            else:
                raise RuntimeError("error=%r" % (error,))
            y_data /= y_scale[:, newaxis]
            x_data /= x_scale[:, newaxis]
            # for data-check
            y_check = y_scale
            x_check = x_scale
        else:
            y_mean = x_mean = y_scale = x_scale = None
            y_check = y_data.var(1)
            x_check = x_data.var(1)
        # check for flat data
        zero_var = [y.name or 'y'] if np.any(y_check == 0) else []
        zero_var.extend(x_name[i] for i, v in enumerate(x_check) if v == 0)
        if zero_var:
            raise ValueError("Flat data: " + ', '.join(zero_var))
        # check for NaN
        has_nan = [y.name] if np.isnan(y_check.sum()) else []
        has_nan.extend(x_name[i] for i, v in enumerate(x_check) if np.isnan(v))
        if has_nan:
            raise ValueError("Data with NaN: " + ', '.join(has_nan))

        self.time = time_dim
        self._scale_data = bool(scale_data)
        # y
        self.y = y_data
        self.y_mean = y_mean
        self.y_scale = y_scale
        self.y_name = y.name
        self._y_info = y.info
        self.ydims = ydims
        self.yshape = tuple(map(len, ydims))
        # x
        self.x = x_data
        self.x_mean = x_mean
        self.x_scale = x_scale
        self.x_name = x_name
        self._x_meta = x_meta
        self._multiple_x = multiple_x
        self._x_is_copy = x_is_copy

    def data_scale_ndvars(self):
        if self._scale_data:
            # y
            if self.ydims:
                y_mean = NDVar(self.y_mean.reshape(self.yshape), self.ydims, self._y_info.copy(), self.y_name)
                y_scale = NDVar(self.y_scale.reshape(self.yshape), self.ydims, self._y_info.copy(), self.y_name)
            else:
                y_mean = self.y_mean[0]
                y_scale = self.y_scale[0]
            # x
            x_mean = []
            x_scale = []
            for name, dim, index in self._x_meta:
                if dim is None:
                    x_mean.append(self.x_mean[index])
                    x_scale.append(self.x_scale[index])
                else:
                    dims = (dim,)
                    x_mean.append(NDVar(self.x_mean[index], dims, {}, name))
                    x_scale.append(NDVar(self.x_scale[index], dims, {}, name))
            if self._multiple_x:
                x_mean = tuple(x_mean)
                x_scale = tuple(x_scale)
            else:
                x_mean = x_mean[0]
                x_scale = x_scale[0]
        else:
            y_mean = y_scale = x_mean = x_scale = None
        return y_mean, y_scale, x_mean, x_scale

    def package_kernel(self, h, tstart):
        """Package kernel as NDVar
        
        Parameters
        ----------
        h : array  (n_y, n_x, n_times)
            Kernel data.
        """
        h_time = UTS(tstart, self.time.tstep, h.shape[-1])
        hs = []
        for name, dim, index in self._x_meta:
            x = h[:, index, :]
            if dim is None:
                dims = (h_time,)
            else:
                dims = (dim, h_time)
            if self.ydims:
                dims = self.ydims + dims
                if len(self.ydims) > 1:
                    x = x.reshape(self.yshape + x.shape[1:])
            else:
                x = x[0]
            hs.append(NDVar(x, dims, self._y_info.copy(), name))

        if self._multiple_x:
            return tuple(hs)
        else:
            return hs[0]

    def package_statistic(self, stat, meas, name):
        if not self.ydims:
            return stat[0]
        elif len(self.ydims) > 1:
            stat = stat.reshape(self.yshape)
        return NDVar(stat, self.ydims, cs.stat_info(meas), name)

    def package_value(self, value, name):
        if not self.ydims:
            return value[0]
        elif len(self.ydims) > 1:
            value = value.reshape(self.yshape)
        return NDVar(value, self.ydims, self._y_info.copy(), name)
