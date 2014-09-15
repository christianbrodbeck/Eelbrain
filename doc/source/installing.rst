.. highlight:: rst

Installing
==========

.. note::
   If you are setting up a new Python environment, the easiest option might be 
   through Enthought `Canopy <https://www.enthought.com/products/canopy>`_; 
   see the :ref:`dependencies` section below).

The easiest way to install Eelbrain is from the
`Python Package Index <https://pypi.python.org/pypi/eelbrain>`_ with
``easy_install``::

    $ easy_install eelbrain[plot.brain]

And it can be updated similarly::

    $ easy_install -U eelbrain[plot.brain]

The optional ``[plot.brain]`` flag installs optional dependencies for plotting
brain data with pysurfer. Since PySurfer itself does not yet support automatic
dependency management, it has to be installed separately::

    $ easy_install -U pysurfer


.. note:: 
    Since version 0.3 Eelbrain comes with a C extension. For Intel Macs, 
    ``easy_install`` has access to a precompiled "egg" and should install 
    automatically as before. ``Pip`` on the other hand always compiles from 
    source and requires a recent version of XCode.


.. _dependencies:

Dependencies
------------

``$ easy_install`` tries to install all required dependencies from the Python
Package Index. However it might be more convenient to install them through
Enthought `Canopy <https://www.enthought.com/products/canopy>`_.


With Canopy
^^^^^^^^^^^

Make sure that you are using the Canopy Python distribution (see
`here <https://support.enthought.com/entries/23646538-Make-Canopy-User-Python-be-your-default-Python-i-e-on-the-PATH->`_).

Prepare the Canopy distribution before installing Eelbrain to make sure
``$ easy_install`` does not try to install the dependencies. The only
additional package needed is Mayavi::

   $ enpkg mayavi

To make sure you are working with the latest versions of all dependencies,
update the Canopy distribution before installing or updating Eelbrain::

   $ enpkg --update-all


Optional Dependencies
^^^^^^^^^^^^^^^^^^^^^

The following items provide additional functionality if they are installed:

* `rpy2 <http://rpy.sourceforge.net>`_ - in order to install it, first install
  `R <http://www.r-project.org>`_, then use pip to install ``rpy2``:
  ``$ pip install rpy2``.
* A working `LaTeX <http://www.latex-project.org/>`_ installation (enables
  exporting tables as pdf).
* `wxPython <http://www.wxpython.org>`_ (included in Canopy) for using the GUI
  (the epoch rejection GUI and enhanced plot toolbars). Installers are provided
  `here <http://www.wxpython.org/download.php>`_ (currently it
  `can not be installed through distutils <http://stackoverflow.com/q/477573/166700>`_).


.. _obtain-source:

Installing from GitHub
----------------------

The Eelbrain source code is hosted on
`GitHub <https://github.com/christianbrodbeck/Eelbrain>`_. The source for the
latest development version can be downloaded as a
`zip archive <https://github.com/christianbrodbeck/Eelbrain/zipball/master>`_.
However, since the code is evolving, the better option is to clone the project
with git. A way to do this is::

    $ cd /target/directory
    $ git clone https://github.com/christianbrodbeck/Eelbrain.git

This will create the folder ``/target/directory/Eelbrain`` containing all the
source files.

The source can then always be updated to the latest version from within the
``Eelbrain`` directory::

    $ cd /target/directory/Eelbrain
    $ git pull

If Eelbrain is installed in ``develop`` mode, changes in the source folder
(e.g., after running ``$ git pull``) take effect without re-installing::

	$ cd /target/directory/Eelbrain
	$ python setup.py develop
