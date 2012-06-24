.. highlight:: rst

Installing Eelbrain
===================

Eelbrain is a pure Python project and uses `Distribute 
<http://packages.python.org/distribute/setuptools.html>`_. 
Since Eelbrain is still under development, it is not on the Python Packaging 
index yet. In order to run Eelbrain, 

#.  Depending on your purpose, install :ref:`optional dependencies <dependencies>`

    .. warning::
        At least under OS X, it seems that wxPython can not be automatically 
        installed by distutils. Unless it is already installed on your system 
        (e.g. through `EPD <http://enthought.com/products/epd.php>`_), wxPython
        has to be installed manually with an installer
        from `here <http://www.wxpython.org/download.php>`_.

#.  :ref:`Obtain the Eelbrian source code <obtain-source>`
#.  Install and run Eelbrain in one of those ways:

    a. :ref:`Install as a package <install-package>`
    b. On OS X, :ref:`create an Eelbrain.app Application <OS-X-app>`
       with :py:mod:`py2app`


.. _dependencies:

Optional Dependencies
^^^^^^^^^^^^^^^^^^^^^

The following modules provide additional functionality if they are installed:
    
* `mne <https://github.com/mne-tools/mne-python>`_
* `tex <http://pypi.python.org/pypi/tex>`_ Enables exporting tables as pdf 
  (also requires a working `LaTeX <http://www.latex-project.org/>`_ installation)
* `bioread <http://pypi.python.org/pypi/bioread>`_ Enables an importer for 
  ``.acq`` files.


.. _obtain-source:

Obtain the Eelbrain Source
^^^^^^^^^^^^^^^^^^^^^^^^^^

The Eelbrain source code is hosted on `GitHub 
<https://github.com/christianmbrodbeck/Eelbrain>`_. The latest source can be 
downloaded as a 
`zip archive <https://github.com/christianmbrodbeck/Eelbrain/zipball/master>`_.
However, since the code is currently evolving, the better option is to clone 
the project with git. A way to do this is::

    $ cd /target/directory
    $ git clone git@github.com:christianmbrodbeck/Eelbrain.git

The source can then always be updated to the latest version
from within the ``Eelbrain`` directory::

    $ cd /target/directory/Eelbrain
    $ git pull

After obtaining the source, there are several options to use Eelbrain:

.. note::
    Make sure to run setup.py with the python version that you want to run
    Eelbrain with.



.. _install-package:

A. Install as a Package
-----------------------

If you install Eelbrain as a package, you can use it in two ways:

- import as a module
- launch as an application

I recommend to install Eelbrain in ``develop`` mode. This has the
benefit that changes in the source folder (e.g., after running 
``$ git pull``) take effect without re-installing::

	$ cd /target/directory/Eelbrain
	$ python setup.py develop

Besides installing the ``eelbrain`` module, this installs a shell script so 
that Eelbrain can be launched with::

    $ eelbrain 


.. _OS-X-app:

B. Create Eelbrian.app on OS X
------------------------------

On OS X, you can generate an application::

    $ cd /target/directory/Eelbrain
    $ python setup.py py2app -A

This will create a small application in 
:file:`/target/directory/Eelbrain/dist/Eelbrain.app`. You can copy this application 
to your Applications folder (or anywhere else). However, the application file 
keeps references to the original source (due to the ``-A`` flag), 
so you must leave the source folder intact. 
The advantage of this method is that any 
changes in the source (such as ``$ git pull``) will be 
reflected as soon as you restart the application.

