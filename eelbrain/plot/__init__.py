"""



Desired Functionality::

    ? >>> plot((X__, __X))
    
    >>> plot_compare(X__, __X, match='subject')
    >>> ttest(X__, __X)
    >>> ttest(X__, __X, match='subject')
    
    1)
    -> X__ needs vars
    -> dataset with 'default_ndvar' property
    
    2)
    -> condition_pointer object stores 
        - pointer to dataset
        - dependent variable name 
        - index (condition=='X__')


Test object stores references to source ConditionPointers and result epochs: 

    >>> test = ttest(X__, __X, match='subject')
    # use test attributes to choose plots  
    >>> plot(test.all)    # [seg1, seg2, seg1-seg2, T&p]
    >>> plot(test.diff)   # [seg1-seg2, T&p]
    >>> plot(test.sig)    # [T&p]



Another option::

    >>> plot_compare(ds, 'X__', '__X', match='subject'
    >>> ttest(ds, 'condition', 'X__', '__X', match='subject')
    





Data-specific Plotting
======================

to produce multiple plots with the same settings, one could use a kwarg 
dicts::

    >>> tvtopo_layout{'sensors':[2,3,4], ...}
    >>> fig_tvtopo(segments, **tvtopo_layout)

Plot function argument names show the kind of input (`epoch` vs `epochs`
indicates whether a single epoch or a list of epochs should be provided; 
`epoch` vs `ndvar` indicates the object type (if an ndvar is provided for a
`epoch` argument, it is converted through the as_epoch method)


Modules
-------

sensor:
    plotting sensor maps
spec:
    Plotting of spectral data
topo
    topographic plots
uts (uniform time series)
    plots for uts data
wavelets
    plots for wavelet data


Attributes
----------

figs:
    a list of figures created through the plot module (only if the mpl figure 
    has a parent object) 


Plans
=====

 - generally, plotting functions accept an argument 'data' that can be a single 
   segment or a list of segments.
 - a figure parent function (/class) should take care of: 
 
    - add link to plot.figs
    - spatial layout (more advanced subplot?) based on primitives (e.g., when 
      doing  several quadratic plots, I only submit one variable specifying
      side length, and Figure_Parent takes care of 

"""


import _base
figs = _base.figs

import sensors
import topo
import uts
