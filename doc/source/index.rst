.. Eelbrain documentation master file, created by
   sphinx-quickstart on Tue Mar 29 23:16:11 2011.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
   
.. Building: $ make html
   If you get ValueError: unknown locale: UTF-8, run 
   export LC_ALL=en_US.UTF-8 
   before make (http://readthedocs.org/docs/flightdeck/en/latest/installation.html).

Eelbrain
========

Eelbrain is an open-source `Python <https://www.python.org>`_ package
for statistical analysis of MEG and EEG data using permutation tests.
Eelbrain is developed by
`Christian Brodbeck <http://www.psych.nyu.edu/pylkkanen/lab/christianbrodbeck.html>`_
at the
`Neuroscience of Language Lab <http://www.psych.nyu.edu/nellab/meglab.html>`_
at New York University. This work is currently funded by grant G1001 from the
`NYU Abu Dhabi Institute
<http://nyuad.nyu.edu/en/research/nyuad-institute.html>`_.

If you use Eelbrain to a substantial degree in work that is published, please
acknowledge it with a link to
`pythonhosted.org/eelbrain <https://pythonhosted.org/eelbrain/>`_.

Manual:

.. toctree::
   :maxdepth: 1

   installing
   changes
   intro

.. toctree::
   :maxdepth: 2

   reference
   recipes
   experiment


.. seealso:: Eelbrain on `GitHub <http://github.com/christianbrodbeck/Eelbrain>`_,
    the `Python Package Index <https://pypi.python.org/pypi/eelbrain>`_,
    and `example scripts
    <https://github.com/christianbrodbeck/Eelbrain/tree/master/examples>`_

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Eelbrain relies on
`NumPy <http://www.numpy.org>`_,
`SciPy <http://scipy.org>`_,
`Matplotlib <http://matplotlib.org>`_,
`MNE-Python <http://martinos.org/mne/stable/index.html>`_,
`PySurfer <http://pysurfer.github.io>`_,
`WxPython <http://wxpython.org>`_ and
`Cython <http://cython.org>`_.
