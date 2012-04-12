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


.. _statistics-example:

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
    >>> print dataset(Y, A)
    A   Y        
    -------------
    a   -1.3564  
    a   -0.74355 
    a   1.0533   
    a   -1.1824  
    a   -0.49033 
    a   0.24881  
    a   0.088035 
    b   2.0232   
    b   1.5045   
    b   0.13688  
    b   -0.53848 
    b   -0.34026 
    b   -0.36607 
    b   1.9893   
    c   2.2071   
    c   1.1281   
    c   1.9894   
    c   -0.062099
    c   1.7671   
    c   0.75025  
    c   0.6192   
    >>> print anova(Y, A)
                SS      df   MS       F        p  
    ----------------------------------------------
    A            8.49    2   4.24   4.61*     .024
    Residuals   16.59   18   0.92                 
    ----------------------------------------------
    Total       25.08   20
    >>> test.pairwise(Y, A, corr='Hochberg')
    
    Pairwise t-Tests (independent samples)
    
        b                 c              
    -------------------------------------
    a   t(12)=-1.78       t(12)=-3.42*   
        p=0.101           p=0.005        
        p(c)=.201         p(c)=.015      
    b                     t(12)=-1.06    
                          p=0.311        
                          p(c)=.311      
    (* Corrected after Hochberg, 1988)
    >>> t = test.pairwise(Y, A, corr='Hochberg')
    >>> print t.get_tex()
    \begin{center}
    \begin{tabular}{lll}
    \toprule
     & b & c \\
    \midrule
    \textbf{a} & $t_{12}=-1.78^{    \ \ \ \ }$ & $t_{12}=-3.42^{*   \ \ \ }$ \\
     & $p=0.101$ & $p=0.005$ \\
     & $p_{c}=.201$ & $p_{c}=.015$ \\
    \textbf{b} &  & $t_{12}=-1.06^{    \ \ \ \ }$ \\
     &  & $p=0.311$ \\
     &  & $p_{c}=.311$ \\
    \bottomrule
    \end{tabular}
    \end{center}



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
