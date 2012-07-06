=========
Reference
=========

Data-Containers
===============

Factor
------

.. autoclass:: eelbrain.vessels.data.factor


Var
---

.. autoclass:: eelbrain.vessels.data.var


Dataset
-------

.. autoclass:: eelbrain.vessels.data.dataset


File I/O
========

Read
----

.. autofunction:: eelbrain.load.txt.tsv
.. autofunction:: eelbrain.load.txt.var


Write
-----

* `Pickling <http://docs.python.org/library/pickle.html>`_: 
  Any data-object can be pickled. 
* ``txt`` export: Save a dataset using it's 
  :py:meth:`~eelbrain.vessels.data.dataset.export` method. 
  Save any iterator with 
  :py:func:`eelbrain.save.txt`.

.. automethod:: eelbrain.vessels.data.dataset.export
.. autofunction:: eelbrain.save.txt