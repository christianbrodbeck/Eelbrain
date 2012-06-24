.. highlight:: rst

Installing Eelbrain
===================

Eelbrain itself is a pure Python project, but it has a lot of :ref:`dependencies 
<installing-dependencies>`, some of which are optional.

#.  Take care of the :ref:`dependencies <installing-dependencies>`
#.  :ref:`Get Eelbrain <installing-eelbrain>`
#.  Run Eelbrain

    a. :ref:`From source <run-from-source>`
    b. On OS X, an :ref:`Eelbrain.app Application <installing-OS-X>` can be created
       with :py:mod:`py2app`
    c. :ref:`Install with setup.py <install>` 


.. _installing-dependencies:

Dependencies
^^^^^^^^^^^^

Besides an installation of Python 2.6 or 2.7, Eelbrain requires a number of 
Python modules to run. The `Enthough Python Distribution <http://enthought.com/
products/epd.php>`_ (EPD) contains most required 
dependencies, so the easiest way to get started is to install EPD.

.. warning::
    In EPD 7.3, ``mdp`` seems to be broken; For now, I recommend using EPD 7.2.
    In addition, on OS X the 32 bit version has to be used because the 64 bit 
    version does not contain ``wxPython``. 

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

* `docutils <http://docutils.sourceforge.net/>`_: nicer formatting in the 
  wxterm help viewer 


The following modules are not included in the EPD and provide optional 
functionality:
    
* `mne <https://github.com/mne-tools/mne-python>`_
* `tex <http://pypi.python.org/pypi/tex>`_ Enables exporting tables as pdf 
  (also requires a working tex distribution)
  (also requires a working `LaTeX <http://www.latex-project.org/>`_ installation)
* `bioread <http://pypi.python.org/pypi/bioread>`_ Enables an importer for 
  ``.acq`` files.


.. _installing-eelbrain:

Get Eelbrain
^^^^^^^^^^^^

The Eelbrain source code is hosted on `GitHub 
<https://github.com/christianmbrodbeck/Eelbrain>`_. 
Since the code is currently evolving, the best option is to clone (or fork) 
the project. A way to do this is::

    $ cd /target/directory
    $ git clone git@github.com:christianmbrodbeck/Eelbrain.git

After the source is downloaded, the source can be updated to the latest version
from within the ``Eelbrain`` directory::

    $ cd /target/directory/Eelbrain
    $ git pull

Now, there are several options to run Eelbrain:


.. _run-from-source:

A. Run from Source
------------------

Eelbrain can be launched by running ``Eelbrain/eelbrain.py``.::

	$ cd /target/directory/Eelbrain
	$ python eelbrain.py
   

.. _installing-OS-X:

B. Create Eelbrian.app on OS X
------------------------------

On OS X a convenient application file can be generated::

    $ cd /target/directory/Eelbrain
    $ python setup.py py2app -A

This will create a small application in 
:file:`/target/directory/Eelbrain/dist/Eelbrain.app`. You can copy this application 
to your Applications folder (or anywhere else). However, the application file 
keeps references to the original source (due to the ``-A`` flag), 
so you must leave the source folder intact. 
The advantage of this method is that any 
changes in the source (such as ``git pull``) will be 
reflected as soon as you restart the application.

.. note::
    Make sure to run setup.py with the python version that you want to run
    Eelbrain with.


.. _install:

C. Installing
-------------

Eelbrain can also be installed using the ``setup.py`` script, but this has to
be repeated every time the source is updated::

    $ cd /target/directory/Eelbrain
    $ python setup.py install

After this, the command ``eelbrain`` is available in the Terminal to start 
Eelbrain.


 