***************
Data Containers
***************


Data-Container Classes
======================

Basic data container classes:

.. currentmodule:: eelbrain.data

.. autosummary::
   :toctree: generated

   Dataset
   Datalist
   Factor
   data_obj.Interaction
   NDVar
   Var


Functions and classes operating with data containers:

.. currentmodule:: eelbrain.eellab

.. autosummary::
   :toctree: generated

   align
   align1
   combine
   Celltable


File I/O
========

Read
----

.. currentmodule:: eelbrain.data

.. autosummary::
   :toctree: generated

   load.eyelink.Edf
   load.tsv
   load.txt.var


Write
-----

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
