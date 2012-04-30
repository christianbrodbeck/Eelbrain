"""

Implementation
==============

Data-Specific Modules
---------------------

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


to produce multiple plots with the same settings, one could use a kwarg 
dicts::

    >>> tvtopo_layout{'sensors':[2,3,4], ...}
    >>> fig_tvtopo(segments, **tvtopo_layout)



Hierarchical Function Structure
-------------------------------

The top-level function (e.g., :py:func:`plot.uts.uts`) takes a list epochs,
and creates a separate plot for each epoch. The _ax_ function  (e.g., 
:py:func:`plot.uts._ax_uts`) takes a single epoch (which is in fact a list of 
layers) and uses the _plt_ function to plot each layer to the axes.

Plot function argument names show the kind of requires input:

``epochs``
    a list of epochs 
``epoch`` (== ``layers``)
    a single epoch/list of layers

if an ndvar is provided for an ``epoch`` argument, it is converted through 
the ndvar.get_summary() method


Plotting with Datasets
----------------------

Option 1: creating epochs (lists layers)


Option 2: test objects contain handles that return the structures needed for 
plotting:: 

    >>> test = ttest(dataset, 'Y', 'condition', 'c1', 'c2')
    >>> plot(test.all)    # [seg1, seg2, seg1-seg2, T&p]
    >>> plot(test.diff)   # [seg1-seg2, T&p]
    >>> plot(test.sig)    # [T&p]


Attributes
----------

figs:
    a list of figures created through the plot module (only if the mpl figure 
    has a parent object) 


Plans
=====

butterfly-plot: allow ndim=1 segments should be able to treat epoch as sensor

 - a figure parent function (/class) should take care of: 
 
    - add link to plot.figs
    - spatial layout (more advanced subplot?) based on primitives (e.g., when 
      doing  several quadratic plots, I only submit one variable specifying
      side length, and Figure_Parent takes care of 


Complementing mpl.pyplot
------------------------

it would be nice if the different plotting subfunctions could be used in 
isolation in the same way as pyplot functions


more TODO
---------

organize returned handles better


???
---

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


"""


import _base
figs = _base.figs

import sensors
import topo
import uts
import utsnd
