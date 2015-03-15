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

* `Pickling <http://docs.python.org/library/pickle.html>`_: All data-objects
  can be pickled. :func:`save.pickle` provides a shortcut for pickling objects.
* Text file export: Save a Dataset using its :py:meth:`~Dataset.save_txt`
  method. Save any iterator with :py:func:`save.txt`.

.. autosummary::
   :toctree: generated

   save.pickle
   save.txt


^^^^^^^^^^^^^^^
Data Processing
^^^^^^^^^^^^^^^

Sorting and reordering
======================

.. autosummary::
   :toctree: generated

   align
   align1
   combine
   Celltable
   

NDVar Transformations
=====================

.. autosummary::
   :toctree: generated

   cwt_morlet
   labels_from_clusters
   morph_source_space


^^^^^^
Tables
^^^^^^

Manipulate data tables and compile information about data objects such as cell
frequencies:

.. autosummary::
   :toctree: generated

    table.difference
    table.frequencies
    table.melt
    table.repmeas
    table.stats


^^^^^^^^^^
Statistics
^^^^^^^^^^

Modules with statistical tests:

.. autosummary::
   :toctree: generated
   
   test
   testnd

To get information about the progress of permutation tests use
:func:`set_log_level` to change the logging level to 'info'::

    >>> set_log_level('info')


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


.. automodule:: eelbrain.plot._uv

.. autosummary::
   :toctree: generated

   Barplot
   Boxplot
   PairwiseLegend
   Correlation
   Histogram
   Regression
   Timeplot


.. automodule:: eelbrain.plot._colors

.. autosummary::
   :toctree: generated

   colors_for_categorial
   colors_for_oneway
   colors_for_twoway
   ColorBar
   ColorGrid
   ColorList


.. automodule:: eelbrain.plot._uts

.. autosummary::
   :toctree: generated

   UTSClusters
   UTSStat
   UTS


.. automodule:: eelbrain.plot._utsnd

.. autosummary::
   :toctree: generated

   Array
   Butterfly


.. automodule:: eelbrain.plot._topo

.. autosummary::
   :toctree: generated

   TopoArray
   TopoButterfly
   Topomap


.. automodule:: eelbrain.plot._sensors

.. autosummary::
   :toctree: generated

   SensorMap
   SensorMaps


.. currentmodule:: eelbrain

.. _ref-guis:

^^^^
GUIs
^^^^

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
