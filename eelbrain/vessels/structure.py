'''
Vessels that impose structure on data

Created on Feb 24, 2012

@author: christian
'''

import numpy as np
import scipy.stats

import data as _data
import eelbrain.utils.statfuncs as _statfuncs



class celltable(object):
    """
    Divides Y into cells defined by X.

    Attributes
    ----------
    .Y, .X,
        Y and X after sub was applied
    .sub, .match:
        input arguments
    .cells : list of str
        list of all cells in X
    .data : dict(cell -> data)
        data in each cell
    .data_indexes : dict(cell -> index-array)
        for each cell, a boolean-array specifying the index for that cell in ``X``

    **If ``match`` is specified**:

    .within : dict(cell1, cell2 -> bool)
        dictionary that specifies for each cell pair whether the corresponding
        comparison is a repeated-measures or an independent measures
        comparison (only available when the input argument ``match`` is
        specified.
    .all_within : bool
        whether all comparison are repeated-measures comparisons or not
    .groups : dict(cell -> group)
        A slice of the match argument describing the group members for each cell.

    """
    def __init__(self, Y, X, match=None, sub=None, match_func=np.mean, ds=None):
        """
        Parameters
        ----------

        Y : var, ndvar
            dependent measurement
        X : categorial
            factor or interaction
        match :
            factor on which cases are matched (i.e. subject for a repeated
            measures comparisons). If several data points with the same
            case fall into one cell of X, they are combined using
            match_func. If match is not None, celltable.groups contains the
            {Xcell -> [match values of data points], ...} mapping corres-
            ponding to self.data
        sub : bool array
            Bool array of length N specifying which cases to include
        match_func : callable
            see match
        ds : dataset
            If a dataset is specified, input items (Y / X / match / sub) can
            be str instead of data-objects, in which case they will be
            retrieved from the dataset.


        Examples
        --------

        Split a repeated-measure variable Y into cells defined by the
        interaction of A and B::

            >>> c = celltable(Y, A % B, match=subject)

        """
        if isinstance(Y, basestring):
            Y = ds.eval(Y)
        if isinstance(X, basestring):
            X = ds.eval(X)
        if isinstance(match, basestring):
            match = ds[match]
        if isinstance(sub, basestring):
            sub = ds.eval(sub)

        if _data.iscategorial(Y) or _data.isndvar(Y):
            if sub is not None:
                Y = Y[sub]
        else:
            Y = _data.asvar(Y, sub)

        if X is not None:
            X = _data.ascategorial(X, sub)

        if match:
            match = _data.asfactor(match, sub)
            assert len(match) == len(Y)
            self.groups = {}

        # save args
        self.X = X
        self.Y = Y
        self.sub = sub
        self.match = match

        # extract cells and cell data
        self.data = {}
        self.data_indexes = {}
        if X is None:
            self.data[None] = Y
            self.data_indexes[None] = np.ones(len(Y), dtype=bool)
            self.cells = [None]
            return

        self.cells = X.cells

        for cell in self.cells:
            self.data_indexes[cell] = cell_index = (X == cell)
            newdata = Y[cell_index]
            if match:
                group = match[cell_index]
                values = group.cells

                # sort
                if len(values) < len(group):
                    newdata = newdata.compress(group, func=match_func)
                    group = _data.factor(values, name=group.name)
                else:
                    group_ids = [group == v for v in values]
                    sort_arg = np.sum(group_ids * np.arange(len(values)), axis=0)
                    newdata = newdata[sort_arg]
                    group = group[sort_arg]

                self.groups[cell] = group

            self.data[cell] = newdata

        if match:
            # determine which cells compare values for dependent values on
            # match_variable
#            n_cells = len(self.indexes)
#            self.within = np.empty((n_cells, n_cells), dtype=bool)
            self.within = {}
            for cell1 in self.cells:
                for cell2 in self.cells:
                    if cell1 == cell2:
                        pass
                    else:
                        v = self.groups[cell1] == self.groups[cell2]
                        if v is not False:
                            v = all(v)
                        self.within[cell1, cell2] = v
                        self.within[cell2, cell1] = v
            self.all_within = np.all(self.within.values())
        else:
            self.all_within = False

    def __repr__(self):
        args = [self.Y.name, self.X.name]
        rpr = "celltable(%s)"
        if self.match is not None:
            args.append("match=%s" % self.match.name)
        if self.sub is not None:
            if _data.isvar(self.sub):
                args.append('sub=%s' % self.sub.name)
            else:
                indexes = ' '.join(str(i) for i in self.sub[:4])
                args.append("sub=[%s...]" % indexes)
        return rpr % (', '.join(args))

    def __len__(self):
        return len(self.cells)

    def cell_label(self, cell, delim=' '):
        """
        Returns a label for a cell. Interaction cells (represented as tuple
        of strings) are joined by ``delim``.

        """
        return _data.cellname(cell, delim=delim)

    def cell_labels(self, delim=' '):
        """
        Returns a list of all cell names as strings.

        delim : str
            delimiter to join interaction cell names

        """
        return [_data.cellname(cell, delim) for cell in self.cells]

    def get_data(self, out=list):
        if out is dict:
            return self.data
        elif out is list:
            return [self.data[cell] for cell in self.cells]

    def get_statistic(self, func=np.mean, a=1, **kwargs):
        """
        Returns a list with a * func(data) for each data cell.

        Parameters
        ----------

        func : callable | str
            statistics function that is applied to the data. Can be string,
            such as '[X]sem', '[X]std', or '[X]ci', e.g. '2sem'.
        a : scalar
            Multiplier (if not provided in ``function`` string).
        kwargs :
            Are submitted to the statistic function.


        Notes
        ----

        :py:meth:`get_statistic_dict`


        See also
        --------

        celltable.get_statistic_dict : return statistics in a dict

        """
        if isinstance(func, basestring):
            if func.endswith('ci'):
                if len(func) > 2:
                    a = float(func[:-2])
                elif a == 1:
                    a = .95
                func = _statfuncs.CIhw
            elif func.endswith('sem'):
                if len(func) > 3:
                    a = float(func[:-3])
                func = scipy.stats.sem
            elif func.endswith('std'):
                if len(func) > 3:
                    a = float(func[:-3])
                func = np.std
                if 'ddof' not in kwargs:
                    kwargs['ddof'] = 1
            else:
                raise ValueError('unrecognized statistic: %r' % func)

        Y = [a * func(self.data[cell].x, **kwargs) for cell in self.cells]
        return Y

    def get_statistic_dict(self, func=np.mean, a=1, **kwargs):
        """
        Same as :py:meth:`~celltable.get_statistic`, except that he result is returned in
        a {cell: value} dictionary.

        """
        return zip(self.cells, self.get_statistic(func=func, a=a, **kwargs))
