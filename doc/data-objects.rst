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
^^^^^^^

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
    a   -0.41676 
    a   -0.056267
    a   -2.1362  
    a   1.6403   
    a   -1.7934  
    a   -0.84175 
    a   0.50288  
    b   -1.2453  
    b   -1.058   
    b   -0.90901 
    b   0.55145  
    b   2.2922   
    b   0.041539 
    b   -1.1179  
    c   2.0391   
    c   0.90384  
    c   1.4809   
    c   2.675    
    c   0.75213  
    c   1.509    
    c   0.62189  
    >>> print anova(Y, A)
                SS      df   MS       F        p  
    ----------------------------------------------
    A           14.50    2   7.25   5.54*     .013
    Residuals   23.56   18   1.31                 
    ----------------------------------------------
    Total       38.06   20
    >>> test.pairwise(Y, A, corr='Hochberg')
    
    Pairwise t-Tests (independent samples)
    
        b                 c              
    -------------------------------------
    a   t(12)=-0.34       t(12)=-3.29*   
        p=0.739           p=0.006        
        p(c)=.739         p(c)=.019      
    b                     t(12)=-2.90*   
                          p=0.013        
                          p(c)=.027      
    (* Corrected after Hochberg, 1988)
    >>> t = test.pairwise(Y, A, corr='Hochberg')
    >>> print t.get_tex()
    \begin{center}
    \begin{tabular}{lll}
    \toprule
     & b & c \\
    \midrule
    \textbf{a} & $t_{12}=-0.34^{    \ \ \ \ }$ & $t_{12}=-3.29^{*   \ \ \ }$ \\
     & $p=0.739$ & $p=0.006$ \\
     & $p_{c}=.739$ & $p_{c}=.019$ \\
    \textbf{b} &  & $t_{12}=-2.90^{*   \ \ \ }$ \\
     &  & $p=0.013$ \\
     &  & $p_{c}=.027$ \\
    \bottomrule
    \end{tabular}
    \end{center}
    >>> plot.boxplot(Y, A, title="My Boxplot", ylabel="value", corr='Hochberg')

.. image:: _static/statistics-example.png


Exporting Data
^^^^^^^^^^^^^^

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
