'''
Vessels that impose structure on data

Created on Feb 24, 2012

@author: christian
'''

import numpy as np
import scipy.stats

import data as _data
import eelbrain.utils.statfuncs as _statfuncs



class celltable:
    """
    Attributes
    ----------
    
    indexes
        list of indexes
    
    cells
        dict(index -> label)
    
    data
        dict(index -> cell_data) 
    
    group
        dict(index -> match_values)
    
    within
        pairwise square; True if all match_values are equal, False 
        otherwise (i.e. whether a dependent measures test is appropri-
        ate or not)
    
    all_within
        True if np.all(self.within)
        
    """
    def __init__(self, Y, X, match=None, sub=None, match_func=np.mean):
        """
        divides Y into cells defined by X
        
        Y       dependent measurement
        X       factor or interaction
        match   factor on which cases are matched (i.e. subject for a repeated 
                measures comparisons). If several data points with the same 
                case fall into one cell of X, they are combined using 
                match_func. If match is not None, celltable.groups contains the
                {Xcell -> [match values of data points], ...} mapping corres-
                ponding to self.data
        sub     Bool Array of length N specifying which cases to include
        match_func:  see match
        
        
        e.g.
        >>> c = S.celltable(Y, A%B, match=subject)
        
        """
        if _data.isfactor(Y):
            if sub is not None:
                Y = Y[sub]
        else:
            Y = _data.asvar(Y, sub)
        
        X = _data.ascategorial(X, sub)
        assert X.N == Y.N
        
        if match:
            match = _data.asfactor(match, sub)
            assert match.N == Y.N
            self.groups = {}
        
        # save args
        self.X = X
        self.Y = Y
        self.sub = sub
        self.match = match

        # extract cells and cell data
        self.data = {}
        self.data_indexes = {}
        self.cells = X.cells
        self.indexes = sorted(X.cells.keys())
        for cell in self.indexes:
            sub = X==cell
            self.data_indexes[cell] = sub
            newdata = Y.x[sub]
            if match:
                # get match ids
                group = match.x[sub]
                occurring_ids = np.unique(group)
                
                # sort
                if len(occurring_ids) < len(group):
                    newdata = np.array([match_func(newdata[group==ID]) 
                                        for ID in occurring_ids])
                    group = occurring_ids
                else:
                    sort_arg = np.argsort(group)
                    group = group[sort_arg]
                    newdata = newdata[sort_arg]
                
                self.groups[cell] = group
            self.data[cell] = newdata
        
        if match:
            # determine which cells compare values for dependent values on 
            # match_variable
#            n_cells = len(self.indexes)
#            self.within = np.empty((n_cells, n_cells), dtype=bool)
            self.within = {}
            for cell1 in self.indexes:
                for cell2 in self.indexes:
                    if cell1==cell2:
                        self.within[cell1,cell2] = True
                    else:
                        v = self.groups[cell1] == self.groups[cell2]
                        if v is not False:
                            v = all(v)
                        self.within[cell1,cell2] = v
                        self.within[cell2,cell1] = v
            self.all_within = np.all(self.within.values())
        else:
            self.within = self.all_within = False
    
    def __repr__(self):
        args = [self.Y.name, self.X.name]
        rpr = "celltable(%s)"
        if self.match != None:
            args.append("match=%s"%self.match.name)
        if self.sub != None:
            indexes = ' '.join(str(i) for i in self.sub[:4])
            args.append("match=[%s...]"  % indexes)
        return rpr % (', '.join(args))
    
    def get_data(self, out=list):
        if out is dict:
            return self.data
        elif out is list:
            return [self.data[i] for i in self.indexes]
    
    def get_statistic(self, function=np.mean, out=dict, a=1, **kwargs):
        """
        :returns: function applied to all data cells.
        
        :arg function: can be string, '[X]sem', '[X]std', or '[X]ci' with X being 
            float, e.g. '2sem'
        :arg out: can be dict or list.
        :arg a: multiplier (if not provided in ``function`` string)
        
        :arg kwargs: are submitted to the statistic function 
        
        """
        if isinstance(function, basestring):
            if function.endswith('ci'):
                if len(function) > 2:
                    a = float(function[:-2])
                elif a == 1:
                    a = .95
                function = _statfuncs.CIhw
            elif function.endswith('sem'):
                if len(function) > 3:
                    a = float(function[:-3])
                function = scipy.stats.sem
            elif function.endswith('std'):
                if len(function) > 3:
                    a = float(function[:-3])
                function = np.std
                if 'ddof' not in kwargs:
                    kwargs['ddof'] = 1
            else:
                raise ValueError('unrecognized statistic: %s'%function)
        
        if out in [list, np.array]:
            as_list = [a * function(self.data[i], **kwargs) for i in self.indexes]
            if out is list:
                return as_list
            else:
                return np.array(as_list)
        elif out is dict:
            return dict((i, a * function(self.data[i], **kwargs)) for i in self.indexes)
        else:
            raise ValueError("out not in [list, dict]")
