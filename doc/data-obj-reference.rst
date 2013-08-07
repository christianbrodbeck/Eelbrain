***************
Data Containers
***************


Data-Container Classes
======================

Basic data container classes:

.. currentmodule:: eelbrain.data

.. autosummary::
   :toctree: generated

   dataset
   datalist
   factor
   data_obj.interaction
   ndvar
   var


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

.. autosummary::
   :toctree: generated

   load.eyelink.Edf
   load.txt.tsv
   load.txt.var


Write
-----

* `Pickling <http://docs.python.org/library/pickle.html>`_: 
  All data-objects can be pickled. :func:`save.pickle` provides a 
  shortcut for pickling objects.
* ``txt`` export: Save a dataset using it's 
  :py:meth:`~dataset.export` method. 
  Save any iterator with 
  :py:func:`save.txt`.

.. autosummary::
   :toctree: generated

   save.pickle
   save.txt
