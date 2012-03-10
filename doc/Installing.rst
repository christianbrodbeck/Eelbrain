.. highlight:: rst

Installing Eelbrain
===================

This page describes different ways of getting a working copy of eelbrain. 

.. Depending on your operating system, you have the following options:
	* Os-X
	* Winows
	* Ubuntu: install from source 


Binary Distributions
--------------------

These are big files containing all the required modules plus their own Python 
interpreter. 

Mac OS X
^^^^^^^^

Download `this zip file <http://dl.dropbox.com/u/659990/eelbrain_dist/
Eelbrain.app.zip>`_ (~70 mb) which contains the application. Unpack and start.


Windows
^^^^^^^

Download `this rar file <http://dl.dropbox.com/u/659990/eelbrain_dist/
eelbrain.rar>`_ (~30 mb). It contains a folder which contains most files necessary to 
run eelbrain. Find and run ``eelbrain.exe`` inside this folder. If this does 
not work you might need to install the `Microsoft Visual C++ 2008 
Redistributable Package <http://www.microsoft.com/downloads/en/details.aspx?
FamilyID=9b2da534-3e03-4391-8a4d-074b9f2bc1bf&displaylang=en>`_ (see section 
5.2 of `this document <http://www.py2exe.org/index.cgi/Tutorial>`_ on why).


Source Distribution
-------------------
	
If you already have a working Python 2.7 installation, you can install Eelbrain
from its source distribution (this should work on any system supporting the 
:ref:`installing-dependencies`, but has been tested only on OS-X, Ubuntu and Windows). 
Download the  
`source file <http://dl.dropbox.com/u/659990/eelbrain_dist/eelbrain-0.0.3.tar.gz>`_.
Unpack it, change into the eelbrain folder, and run the setup script::

	$ cd eelbrain
	$ python setup.py install

From now on you can run eelbrain on Linux::

	$ eelbrain

On Windows this does not seem to work, instead you have to use the full path 
(assuming ``C:\python27`` is the location of your Python installation)::

	$ python C:\python27\scripts\eelbrain


.. _installing-dependencies:

Dependencies
^^^^^^^^^^^^

Besides an installation of Python 2.6 or 2.7, Eelbrain requires a number of 
Python modules to run. The `Enthough Python Distribution <http://
www.enthought.com/products/edudownload.php>`_ (EPD) contains all required 
dependencies, so the easiest way to get started is to install EPD.

..note::
	The EPD also
	installs `setuptools <http://pypi.python.org/pypi/setuptools>`_, which means 
	that you can install any additional modules using::

		$ easy_install modulename

* Required (included in EPD)

	* `wxPython <http://www.wxpython.org/>`_
	* `NumPy <http://numpy.scipy.org>`_ 
	* `matplotlib <http://matplotlib.sourceforge.net/>`_
	* `SciPy <http://www.scipy.org/>`_

* Optional

	* `tex <http://pypi.python.org/pypi/tex>`_ Enables exporting tables as pdf
	  (also requires a working Latex installation) 
	* `bioread <http://pypi.python.org/pypi/bioread>`_ Enables an importer for 
	  ``.acq`` files.


Installing for Development (Launchpad)
--------------------------------------

Make sure you have all :ref:`installing-dependencies`.

After installing `Bazaar <http://wiki.bazaar.canonical.com/Download>`_, open a 
system shell, change to the directory where you would like to keep
the files (Windows: ``cd x:\target\folder``, Linux: ``cd /target/folder``), 
and type::

	$ bzr branch lp:eelbrain

This will create a new folder 
(``x:\target\folder\eelbrain`` / :file:`/target/folder/eelbrain`)
which contains all the source files. 


Running on Os-X
^^^^^^^^^^^^^^^

In Terminal, change to the eelbrain directory and create an application file 
for development::

	$ cd /target/dir/eelbrain
	$ python setup.py py2app -A

This will create a small application in 
:file:`/target/dir/eelbrain/dist/Eelbrain.app`. You can copy this aaplication 
to your Applications folder (or anywhere else). However, you must leave the 
source folder intact, since the app uses those scripts. Thus, if you make 
changes in the source :file:`/target/dir/eelbrain/eelbrain`, this will be 
reflected as soon as you restart the application.

..note::
	Make sure to run setup.py with the python version that you want to run
	Eelbrain with. 


Running on Windows
^^^^^^^^^^^^^^^^^^

In order to run the scripts, you will need to add the eelbrain folder 
(``x:\target\folder\eelbrain``) to your system's path. Follow `this link 
<http://geekswithblogs.net/renso/archive/2009/10/21/how-to-set-the-windows-path-in-windows-7.aspx>`_.

Then you can use :file:`eelbrain.bat` in ``x:\target\folder\eelbrain`` to
conveniently start eelbrain.


Updating your Local Copy
^^^^^^^^^^^^^^^^^^^^^^^^

You can update your source files to the latest version with a simple command. 
Change to the eelbrain directory (:file:`/target/dir/eelbrain`) and type::

	$ bzr pull

(To learn more type ``bzr help pull``)


Modifying the Source Code
^^^^^^^^^^^^^^^^^^^^^^^^^

Using Bazaar will also make it easier to make changes to the source code and 
integrate them with the main project.

.. seealso:: `Bazaar documentation <http://wiki.bazaar.canonical.com/Documentation>`_

 