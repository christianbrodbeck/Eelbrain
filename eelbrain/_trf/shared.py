from collections import Iterator
from functools import reduce
from math import ceil, floor
from operator import mul

import numpy as np
from numpy import newaxis

from .. import _info
from .._data_obj import NDVar, UTS, dataobj_repr, ascategorial, asndvar


class RevCorrData(object):
    """Restructure input NDVars into arrays for reverse correlation
    
    Attributes
    ----------
    y : NDVar
        Dependent variable.
    x : NDVar | sequence of NDVar
        Predictors.
    segments : np.ndarray
        ``(n_segments, 2)`` array of segment ``[start, stop]`` indices.
    cv_segments : Sequence
        Sequence of ``(all_segments, train, test)`` tuples, where each is a
        2d-array of ``[start, stop]`` indices.
    cv_indexes : Sequence
        Only available for segmented data. For each partition, the index into
        :attr:`.segments` used as test set.
    """
    def __init__(self, y, x, error, scale_data, ds=None):
        y = asndvar(y, ds=ds)
        if isinstance(x, (tuple, list, Iterator)):
            x = (asndvar(x_, ds=ds) for x_ in x)
        else:
            x = asndvar(x, ds=ds)

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
        n_times = len(time_dim)

        # determine cases (used as segments)
        n_cases = segments = None
        for x_ in x:
            # determine cases
            if n_cases is None:
                if x_.has_case:
                    n_cases = len(x_)
                    # check y
                    if not y.has_case:
                        raise ValueError('y=%r: x has case dimension but y has not case' % (y,))
                    elif len(y) != n_cases:
                        raise ValueError('y=%r: has different number of cases from x (%i)' % (y, n_cases))
                    # prepare segment index
                    seg_i = np.arange(0, n_cases * n_times + 1, n_times, np.int64)[:, newaxis]
                    segments = np.hstack((seg_i[:-1], seg_i[1:]))
                else:
                    n_cases = 0
            elif n_cases:
                if len(x_) != n_cases:
                    raise ValueError(f'x={x}: not all components have same number of cases')
            else:
                assert not x_.has_case, 'some but not all x have case'
        case_to_segments = n_cases > 0

        # y_data:  ydim x time array
        if y.ndim == 1:
            y_dimnames = ('time',)
            ydims = ()
        elif case_to_segments:
            y_dimnames = y.get_dimnames(last=('case', 'time'))
            ydims = y.get_dims(y_dimnames[:-2])
        else:
            y_dimnames = y.get_dimnames(last='time')
            ydims = y.get_dims(y_dimnames[:-1])
        shape = (
            reduce(mul, map(len, ydims), 1),
            n_cases * n_times if case_to_segments else n_times)
        y_data = y.get_data(y_dimnames).reshape(shape)

        # x_data:  predictor x time array
        x_data = []
        x_meta = []
        x_names = []
        n_x = 0
        for x_ in x:
            ndim = x_.ndim - bool(n_cases)
            if ndim == 1:
                xdim = None
                dimnames = ('case' if n_cases else np.newaxis, 'time')
                data = x_.get_data(dimnames)
                index = n_x
                x_names.append(dataobj_repr(x_))
            elif ndim == 2:
                dimnames = x_.get_dimnames(last='time')
                xdim = x_.get_dim(dimnames[-2])
                if n_cases:
                    dimnames = (xdim.name, 'case', 'time')
                data = x_.get_data(dimnames)
                index = slice(n_x, n_x + len(data))
                x_repr = dataobj_repr(x_)
                for v in xdim:
                    x_names.append("%s-%s" % (x_repr, v))
            else:
                raise NotImplementedError("x with more than 2 dimensions")
            if n_cases:
                data = data.reshape((-1, n_cases * n_times))
            x_data.append(data)
            x_meta.append((x_.name, xdim, index))
            n_x += len(data)

        if len(x_data) == 1:
            x_data = x_data[0]
            x_is_copy = False
        else:
            x_data = np.concatenate(x_data)
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
            # zero-padding for convolution
            x_pads = -x_mean / x_scale
        else:
            y_mean = x_mean = y_scale = x_scale = None
            y_check = y_data.var(1)
            x_check = x_data.var(1)
            x_pads = np.zeros(n_x)
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
        self.segments = segments
        self.cv_segments = self.cv_indexes = self.n_segments = self.model = None
        self._scale_data = bool(scale_data)
        self.shortest_segment_n_times = n_times
        # y
        self.y = y_data
        self.y_mean = y_mean
        self.y_scale = y_scale
        self.y_name = y.name
        self._y_info = y.info
        self.ydims = ydims
        self.yshape = tuple(map(len, ydims))
        self.full_y_dims = y.get_dims(y_dimnames)
        # x
        self.x = x_data
        self.x_mean = x_mean
        self.x_scale = x_scale
        self.x_name = x_name
        self._x_meta = x_meta
        self._multiple_x = multiple_x
        self._x_is_copy = x_is_copy
        self.x_pads = x_pads

    def initialize_cross_validation(self, n_segments=None, model=None, ds=None):
        if n_segments is not None and n_segments <= 1:
            raise ValueError(f"n_segments={n_segments}")
        cv_segments = []  # list of (segments, train, test)
        n_times = len(self.time)
        if self.segments is None:
            if model is not None:
                raise TypeError('model=%r: model cannot be specified in unsegmented data' % (dataobj_repr(model),))
            if n_segments is None:
                n_segments = 10
            seg_n_times = int(floor(n_times / n_segments))
            # first
            for i in range(n_segments):
                test = ((seg_n_times * i, seg_n_times * (i + 1)),)
                if i == 0:  # first
                    train = ((seg_n_times, n_times),)
                elif i == n_segments - 1:  # last
                    train = ((0, n_times - seg_n_times),)
                else:
                    train = ((0, seg_n_times * i),
                             (seg_n_times * (i + 1), n_times))
                cv_segments.append((np.vstack((train, test)), train, test))
            cv_segments = (tuple(np.array(s, np.int64) for s in cv) for cv in cv_segments)
        else:
            n_total = len(self.segments)
            irange = np.arange(n_total)
            if model is None:
                cell_indexes = [irange]
            else:
                model = ascategorial(model, ds=ds, n=n_total)
                cell_indexes = [irange[model == cell] for cell in model.cells]
            cell_size = min(len(i) for i in cell_indexes)
            if n_segments is None:
                n_segments = min(cell_size, 10)
            elif n_segments > cell_size:
                raise ValueError('n_segments=%r with cell size %i' % (n_segments, cell_size))
            indexes = []
            for i in range(n_segments):
                index = np.zeros(n_total, bool)
                for cell_index in cell_indexes:
                    index[cell_index[i::n_segments]] = True
                indexes.append(index)
            cv_segments = ((self.segments, self.segments[np.invert(i)], self.segments[i]) for i in indexes)
            self.cv_indexes = tuple(indexes)
        self.n_segments = n_segments
        self.cv_segments = tuple(cv_segments)
        self.model = dataobj_repr(model)

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
        return NDVar(stat, self.ydims, _info.for_stat_map(meas), name)

    def package_value(self, value, name):
        if not self.ydims:
            return value[0]
        elif len(self.ydims) > 1:
            value = value.reshape(self.yshape)
        return NDVar(value, self.ydims, self._y_info.copy(), name)

    def package_y_like(self, data, name):
        shape = tuple(map(len, self.full_y_dims))
        return NDVar(data.reshape(shape), self.full_y_dims, {}, name)
