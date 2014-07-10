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

.. currentmodule:: eelbrain.data

Functions:

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

* `Pickling <http://docs.python.org/library/pickle.html>`_: 
  All data-objects can be pickled. :func:`save.pickle` provides a 
  shortcut for pickling objects.
* Text file export: Save a Dataset using it's 
  :py:meth:`~Dataset.save_txt` method. 
  Save any iterator with 
  :py:func:`save.txt`.

.. autosummary::
   :toctree: generated

   save.pickle
   save.txt


^^^^^^^^^^^^^^^
Data Processing
^^^^^^^^^^^^^^^

Sorting and reordering
======================

.. currentmodule:: eelbrain

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
   source_induced_power


^^^^^^
Tables
^^^^^^

Manipulate data tables and compile information about data objects such as cell 
frequencies:

.. currentmodule:: eelbrain.data

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


^^^^^^^^
Plotting
^^^^^^^^

Modules for plotting:

.. autosummary::
   :toctree: generated
   
    plot
    plot.brain
    plot.uv

.. seealso:: :ref:`plotting-notes`
