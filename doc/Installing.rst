.. highlight:: rst

Installing Eelbrain
===================

Eelbrain itself is a pure Python project, but it has a lot of :ref:`dependencies 
<installing-dependencies>`, some of which are optional.

#.  Take care of the :ref:`dependencies <installing-dependencies>`
#.  Get :Ref:`Eelbrain <installing-eelbrain>`
#.  On OS X, an :ref:`Eelbrain.app Application <installing-OS-X>` can be created
    with :py:mod:`py2app` (optional)


.. _installing-dependencies:

Dependencies
^^^^^^^^^^^^

Besides an installation of Python 2.6 or 2.7, Eelbrain requires a number of 
Python modules to run. The `Enthough Python Distribution <http://enthought.com/
products/epd.php>`_ (EPD) contains most required 
dependencies, so the easiest way to get started is to install EPD.

.. note::
    The EPD also
    installs `setuptools <http://pypi.python.org/pypi/setuptools>`_, which means 
    that you can install any additional modules using::
    
        $ easy_install modulename

The following modules are included in the EPD and are required:

* `WxPython <http://www.wxpython.org/>`_
* `NumPy <http://numpy.scipy.org>`_
* `Matplotlib <http://matplotlib.sourceforge.net/>`_
* `SciPy <http://www.scipy.org/>`_
* `MDP <http://mdp-toolkit.sourceforge.net/>`_


In EPD and optional:

* `docutils <http://docutils.sourceforge.net/>`_: nicer formatting in the wxterm help viewer 


The following modules are not included in the EPD and provide optional 
functionality:
    
* `mne <https://github.com/mne-tools/mne-python>`_
* `tex <http://pypi.python.org/pypi/tex>`_ Enables exporting tables as pdf 
  (also requires a working tex distribution)
  (also requires a working `LaTeX <http://www.latex-project.org/>`_ installation)
* `bioread <http://pypi.python.org/pypi/bioread>`_ Enables an importer for 
  ``.acq`` files.


.. _installing-eelbrain:

Get and run Eelbrain
^^^^^^^^^^^^^^^^^^^^

The Eelbrain source code is hosted on `GitHub 
<https://github.com/christianmbrodbeck/Eelbrain>`_. 
Since the code is currently evolving, the best option is to clone (or fork) 
the project and run Eelbrain from the source.

A way to do this is::

    $ cd /target/directory
    $ git clone git@github.com:christianmbrodbeck/Eelbrain.git

After the source is downloaded, Eelbrain can be started with::

	$ cd Eelbrain
	$ python eelbrain.py

From within the same directory (``/target/directory/Eelbrain``), 
the source can be updated to the latest version with::

    $ git pull


.. _installing-OS-X:

Create Eelbrian.app on OS X
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In Terminal, change to the eelbrain directory and create an application file::

    $ cd /target/directory/Eelbrain
    $ python setup.py py2app -A

This will create a small application in 
:file:`/target/directory/Eelbrain/dist/Eelbrain.app`. You can copy this application 
to your Applications folder (or anywhere else). However, the application file 
keeps references to the original source, so you must leave the 
source folder intact. For the same reason, you can make 
changes to the source (such as ``git pull``) which will be 
reflected as soon as you restart the application.

.. note::
    Make sure to run setup.py with the python version that you want to run
    Eelbrain with.


 