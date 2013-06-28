***************
Data Containers
***************


Data-Container Classes
======================

Basic data container classes:

.. currentmodule:: eelbrain.vessels.data

.. autosummary::
   :toctree: generated

   dataset
   datalist
   factor
   interaction
   ndvar
   var


Functions and classes operating with data containers:

.. currentmodule:: eelbrain.eellab

.. autosummary::
   :toctree: generated

   align
   align1
   combine
   celltable


File I/O
========

Read
----

.. currentmodule:: eelbrain

.. autosummary::
   :toctree: generated

   load.eyelink.Edf
   load.txt.tsv
   load.txt.var


Write
-----

* `Pickling <http://docs.python.org/library/pickle.html>`_: 
  All data-objects can be pickled. :func:`eelbrain.save.pickle` provides a 
  shortcut for pickling objects.
* ``txt`` export: Save a dataset using it's 
  :py:meth:`~eelbrain.vessels.data.dataset.export` method. 
  Save any iterator with 
  :py:func:`eelbrain.save.txt`.

.. autosummary::
   :toctree: generated

   save.pickle
   save.txt
