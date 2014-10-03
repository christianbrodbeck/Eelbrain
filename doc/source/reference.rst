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


.. _ref-plotting:

^^^^^^^^
Plotting
^^^^^^^^

Modules for plotting. General Matplotlib-based plots are described in the
:mod:`.plot` module; Mayavi/PySurfer based plots for MNE current source
estimates are described in :mod:`.plot.brain`.

.. autosummary::
   :toctree: generated
   
    plot
    plot.brain


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
application. All windows stay open. To return to the application from the
shell, run :func:`gui.run`.

.. autosummary::
   :toctree: generated

    gui.run


^^^^^^^^^^^^^^^^^^
Experiment Classes
^^^^^^^^^^^^^^^^^^

The :mod:`eelbrain.experiment` module contains tools for managing hierarchical
collections of templates.

.. autosummary::
   :toctree: generated

   experiment.TreeModel
   experiment.FileTree
   experiment.MneExperiment

