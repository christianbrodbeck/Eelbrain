************
Data-Objects
************


There are two primary data-objects: 

* :class:`~eelbrain.vessels.data.var` for scalar variables
* :class:`~eelbrain.vessels.data.factor` for categorial variables

Multiple variables belonging to the same dataset can be grouped in a 
:class:`~eelbrain.vessels.data.dataset` object.


Factor
======

A factor is a container for one-dimensional, categorial data: Each case is 
described by a string label. The most obvious way to initialize a factor 
is a list of strings::

    >>> A = factor(['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'], name='A')

There are many shortcuts to initialize factors  (for more shortcuts see 
the factor's class documentation)::

    >>> A = factor(['a', 'b', 'c'], rep=4, name='A')
    >>> A
    factor(['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'c', 'c', 'c', 'c'], name='A')

Indexing works like for arrays::

    >>> A[0]
    'a'
    >>> A[0:6]
    factor(['a', 'a', 'a', 'a', 'b', 'b'], name='A')

All values present in a factor are accessible in its ``.cells`` attribute::

    >>> A.cells
    ['a', 'b', 'c']

Based on the factor's values, boolean indexes can be generated::

    >>> A == 'a'
    array([ True,  True,  True,  True, False, False, False, False, False,
           False, False, False], dtype=bool)
    >>> A.isany('a', 'b')
    array([ True,  True,  True,  True,  True,  True,  True,  True, False,
           False, False, False], dtype=bool)
    >>> A.isnot('a', 'b')
    array([False, False, False, False, False, False, False, False,  True,
            True,  True,  True], dtype=bool)

Interaction effects can be constructed from multiple factors::

    >>> B = factor(['d', 'e'], rep=2, tile=3, name='B')
    >>> B
    factor(['d', 'd', 'e', 'e', 'd', 'd', 'e', 'e', 'd', 'd', 'e', 'e'], name='B')
    >>> i = A % B
    >>> i
    interaction(A, B)

Interaction effects are in many ways interchangeable with factors in places 
where a categorial model is required::
 
    >>> i.cells
    [('a', 'd'), ('a', 'e'), ('b', 'd'), ('b', 'e'), ('c', 'd'), ('c', 'e')]
    >>> i == ('a', 'd')
    array([ True,  True, False, False, False, False, False, False, False,
           False, False, False], dtype=bool)


Vars
====


Models
======


Datasets
========


.. _statistics-example:

Example
=======

Below is a simple example using data objects. For more examples, see the 
``Eelbrain/examples/statistics`` folder::

    >>> import numpy as np
    >>> from eelbrain.eellab import *
    >>> y = np.empty(21)
    >>> y[:14] = np.random.normal(0, 1, 14)
    >>> y[14:] = np.random.normal(1.5, 1, 7)
    >>> Y = var(y, 'Y')
    >>> Y
    var([-0.417, -0.0563, -2.14, 1.64, -1.79, -0.842, 0.503, -1.25, -1.06,
    -0.909, 0.551, 2.29, 0.0415, -1.12, 2.04, 0.904, 1.48, 2.68, 0.752, 1.51, 
    0.622], name='Y')
    >>> A = factor('abc', 'A', rep=7)
    >>> A
    factor(['a', 'a', 'a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b', 'b',
    'b', 'c', 'c', 'c', 'c', 'c', 'c', 'c'], name='A')
    >>> print dataset(Y, A)
    Y           A
    -------------
    -0.41676    a
    -0.056267   a
    -2.1362     a
    1.6403      a
    -1.7934     a
    -0.84175    a
    0.50288     a
    -1.2453     b
    -1.058      b
    -0.90901    b
    0.55145     b
    2.2922      b
    0.041539    b
    -1.1179     b
    2.0391      c
    0.90384     c
    1.4809      c
    2.675       c
    0.75213     c
    1.509       c
    0.62189     c
    >>> test.anova(Y, A)
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
        p=.739            p=.006         
        p(c)=.739         p(c)=.019      
    b                     t(12)=-2.90*   
                          p=.013         
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
     & $p=.739$ & $p=.006$ \\
     & $p_{c}=.739$ & $p_{c}=.019$ \\
    \textbf{b} &  & $t_{12}=-2.90^{*   \ \ \ }$ \\
     &  & $p=.013$ \\
     &  & $p_{c}=.027$ \\
    \bottomrule
    \end{tabular}
    \end{center}
    >>> plot.uv.boxplot(Y, A, title="My Boxplot", ylabel="value", corr='Hochberg')

.. image:: _static/statistics-example.png


Exporting Data
==============

:class:`~eelbrain.vessels.data.dataset` objects have an ``export()`` method for
saving in various formats. In addition, the dataset's
:py:meth:`~eelbrain.vessels.data.dataset.as_table` method can create tables with 
more flexibility.

Iterators (such as :class:`~eelbrain.vessels.data.var` and 
:class:`~eelbrain.vessels.data.factor`) can be exported using the
:func:`eelbrain.save.txt` function.

.. 
    not nice enough ...
    
    Class Documentation
    ===================
    
    .. autoclass:: eelbrain.vessels.data.var
    	:members:
    
    .. autoclass:: eelbrain.vessels.data.factor
    
    .. autoclass:: eelbrain.vessels.data.dataset
