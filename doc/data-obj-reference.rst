************
Data Classes
************

.. currentmodule:: eelbrain.data

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

   data_obj.Interaction


********
File I/O
********

Load
====

.. currentmodule:: eelbrain.data

.. autosummary::
   :toctree: generated

   load.eyelink.Edf
   load.fiff
   load.tsv
   load.txt.var
   load.unpickle


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


***************
Data Processing
***************

Sorting and reordering
======================

.. currentmodule:: eelbrain.data

.. autosummary::
   :toctree: generated

   align
   align1
   combine
   Celltable
   

Data Transformations
====================

.. autosummary::
   :toctree: generated

   cwt_morlet


**********
Statistics
**********

.. currentmodule:: eelbrain.data

.. autosummary::
   :toctree: generated
   
   test
   testnd
