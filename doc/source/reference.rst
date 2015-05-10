*********
Reference
*********

^^^^^^^^^^^^
Data Classes
^^^^^^^^^^^^

.. currentmodule:: eelbrain

Primary data classes:

.. autosummary::
   :toctree: generated

   Dataset
   Factor
   Var
   NDVar
   Datalist


Secondary classes (not usually initialized by themselves but through operations
on primary data-objects):

.. autosummary::
   :toctree: generated

   Interaction
   Model


^^^^^^^^
File I/O
^^^^^^^^

Load
====

.. py:module:: eelbrain.load
.. py:currentmodule:: eelbrain

Eelbrain has its own function for unpickling. In contrast to `normal unpickling
<https://docs.python.org/2/library/pickle.html>`_, this function can also load 
files pickled with earlier Eelbrain versions:

.. autosummary::
   :toctree: generated

   load.unpickle

Modules:

.. autosummary::
   :toctree: generated

   load.eyelink
   load.fiff
   load.txt
   load.besa


Save
====

.. py:module:: eelbrain.save
.. py:currentmodule:: eelbrain

* `Pickling <http://docs.python.org/library/pickle.html>`_: All data-objects
  can be pickled. :func:`save.pickle` provides a shortcut for pickling objects.
* Text file export: Save a Dataset using its :py:meth:`~Dataset.save_txt`
  method. Save any iterator with :py:func:`save.txt`.

.. autosummary::
   :toctree: generated

   save.pickle
   save.txt


^^^^^^^^^^^^^^^^^^^^^^
Sorting and Reordering
^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated

   align
   align1
   combine
   Celltable
   

^^^^^^^^^^^^^^^^^^^^^
NDVar Transformations
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated

   cwt_morlet
   labels_from_clusters
   morph_source_space


^^^^^^
Tables
^^^^^^

.. py:module:: eelbrain.table
.. py:currentmodule:: eelbrain

Manipulate data tables and compile information about data objects such as cell
frequencies:

.. autosummary::
   :toctree: generated

    table.difference
    table.frequencies
    table.melt
    table.melt_ndvar
    table.repmeas
    table.stats


^^^^^^^^^^
Statistics
^^^^^^^^^^

.. py:module:: eelbrain.test
.. py:currentmodule:: eelbrain

Univariate statistical tests:

.. autosummary::
   :toctree: generated
   
   test.anova
   test.pairwise
   test.ttest
   test.correlations
   test.lilliefors


^^^^^^^^^^^^^^^^^^^^^^^^^^
Mass-Univariate Statistics
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:module:: eelbrain.testnd
.. py:currentmodule:: eelbrain

.. autosummary::
   :toctree: generated

   testnd.ttest_1samp
   testnd.ttest_rel
   testnd.ttest_ind
   testnd.t_contrast_rel
   testnd.anova
   testnd.corr


By default the tests in this module produce maps of statistical parameters
along with maps of p-values uncorrected for multiple comparison. Using different
parameters, different methods for multiple comparison correction can be applied
(for more details and options see the documentation for individual tests):

**1: permutation for maximum statistic** (``samples=n``)
    Look for the maximum
    value of the test statistic in ``n`` permutations and calculate a p-value
    for each data point based on this distribution of maximum statistics.
**2: Threshold-based clusters** (``samples=n, pmin=p``)
    Find clusters of data
    points where the original statistic exceeds a value corresponding to an
    uncorrected p-value of ``p``. For each cluster, calculate the sum of the
    statistic values that are part of the cluster. Do the same in ``n``
    permutations of the original data and retain for each permutation the value
    of the largest cluster. Evaluate all cluster values in the original data
    against the distributiom of maximum cluster values (see [1]_).
**3: Threshold-free cluster enhancement** (``samples=n, tfce=True``)
    Similar to
    (1), but each statistical parameter map is first processed with the
    cluster-enhancement algorithm (see [2]_). This is the most computationally
    intensive option.

To get information about the progress of permutation tests use
:func:`set_log_level` to change the logging level to 'info'::

    >>> set_log_level('info')


.. [1] Maris, E., & Oostenveld, R. (2007). Nonparametric
    statistical testing of EEG- and MEG-data. Journal of Neuroscience Methods,
    164(1), 177-190. doi:10.1016/j.jneumeth.2007.03.024
.. [2] Smith, S. M., and Nichols, T. E. (2009). Threshold-Free Cluster
    Enhancement: Addressing Problems of Smoothing, Threshold Dependence and
    Localisation in Cluster Inference. NeuroImage, 44(1), 83-98.
    doi:10.1016/j.neuroimage.2008.03.061


.. _ref-plotting:

^^^^^^^^
Plotting
^^^^^^^^

The :mod:`.plot` module contains all general Matplotlib-based plots, listed
below. See the module documentation for general information on plotting.
Mayavi/PySurfer based plots for MNE source estimates are separately located in
the :mod:`.plot.brain` module:


.. autosummary::
   :toctree: generated

    plot
    plot.brain


Plot univariate data (:class:`Var` objects):

.. autosummary::
   :toctree: generated

   plot.Barplot
   plot.Boxplot
   plot.PairwiseLegend
   plot.Correlation
   plot.Histogram
   plot.Regression
   plot.Timeplot


Color tools for plotting:

.. autosummary::
   :toctree: generated

   plot.colors_for_categorial
   plot.colors_for_oneway
   plot.colors_for_twoway
   plot.ColorBar
   plot.ColorGrid
   plot.ColorList


Plot uniform time-series:

.. autosummary::
   :toctree: generated

   plot.UTSClusters
   plot.UTSStat
   plot.UTS


Plot multidimensional uniform time series:

.. autosummary::
   :toctree: generated

   plot.Array
   plot.Butterfly


Plot topographic maps of sensor space data:

.. autosummary::
   :toctree: generated

   plot.TopoArray
   plot.TopoButterfly
   plot.Topomap


Plot sensor layout maps:

.. autosummary::
   :toctree: generated

   plot.SensorMap
   plot.SensorMaps


.. _ref-guis:

^^^^
GUIs
^^^^

.. py:module:: eelbrain.gui
.. py:currentmodule:: eelbrain

Tools with a graphical user interface (GUI):

.. autosummary::
   :toctree: generated

    gui.select_epochs


.. _gui:

Controlling the GUI Application
===============================

Eelbrain uses a wxPython based application to create GUIs. This application can
not take input form the user at the same time as the shell from which the GUI
is invoked. By default, the GUI application is activated whenever a gui is
created in interactive mode. While the application is processing user input,
the shell can not be used. In order to return to the shell, simply quit the
application (the *python/Quit Eelbrain* menu command or Command-Q). In order to
return to the terminal without closing all windows, use the alternative
*Go/Yield to Terminal* command (Command-Alt-Q). To return to the application
from the shell, run :func:`gui.run`. Beware that if you terminate the Python
session from the terminal, the application is not given a chance to assure that
information in open windows is saved.

.. autosummary::
   :toctree: generated

    gui.run


^^^^^^^^^^^^^^^^^^
Experiment Classes
^^^^^^^^^^^^^^^^^^

The :class:`MneExperiment` class serves as a base class for analyzing MEG
data (gradiometer only) with MNE:

.. autosummary::
   :toctree: generated

   MneExperiment

.. seealso::
    For the guide on working with the MneExperiment class see
    :ref:`experiment-class-guide`.

The :mod:`eelbrain.experiment` module contains tools for managing hierarchical
collections of templates on which the :class:`MneExperiment` is based.

.. autosummary::
   :toctree: generated

   experiment.TreeModel
   experiment.FileTree
