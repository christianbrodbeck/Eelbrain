************
Data-Objects
************

Vars, Factors, Datasets
=======================

There are two primary data-objects: 

* :class:`~eelbrain.vessels.data.var` for scalar variables
* :class:`~eelbrain.vessels.data.factor` for categorial variables

Multiple variables belonging to the same dataset can be grouped in a 
:class:`~eelbrain.vessels.data.dataset` object.


Example
-------

This example illustrates the use of those objects::

    >>> import numpy as np
    >>> from eelbrain.useful import *
    >>> y = np.empty(21)
    >>> y[:14] = np.random.normal(0, 1, 14)
    >>> y[14:] = np.random.normal(1.5, 1, 7)
    >>> Y = var(y, 'Y')
    >>> A = factor('abc', 'A', rep=7)
    >>> print dataset(Y, A).as_table()
    A   Y         
    a   -0.8935   
    a   0.220171  
    a   2.22897   
    a   -0.63312  
    a   -0.929644 
    a   -0.0646979
    a   0.189562  
    b   1.52957   
    b   -0.757737 
    b   -0.294725 
    b   -0.977417 
    b   -0.564112 
    b   0.402234  
    b   -0.772369 
    c   1.94017   
    c   2.19559   
    c   0.364439  
    c   0.802289  
    c   1.92717   
    c   2.44292   
    c   1.23042   
    >>> print anova(Y, A)
                SS      df   MS       F        p  
    ----------------------------------------------
    A           12.90    2   6.45   7.52**    .004
    Residuals   15.43   18   0.86                 
    ----------------------------------------------
    Total       28.33   20


Exporting Data
--------------

:class:`~eelbrain.vessels.data.dataset`, :class:`~eelbrain.vessels.data.var` and 
:class:`~eelbrain.vessels.data.factor` objects have an ``export()`` method for
saving in various formats. In addition, the dataset's
:py:meth:`~eelbrain.vessels.data.dataset.as_table` method can create tables with 
more flexibility.


Class Documentation
===================

.. autoclass:: eelbrain.vessels.data.var
	:members:

.. autoclass:: eelbrain.vessels.data.factor

.. autoclass:: eelbrain.vessels.data.dataset
