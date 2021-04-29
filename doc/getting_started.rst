***************
Getting Started
***************

.. contents:: Contents
   :local:

.. currentmodule:: eelbrain


Documentation
-------------

For an introduction to Eelbrain, see :ref:`exa-intro` and the other :ref:`examples`.
For details see the API :ref:`reference`.


Interacting with other Python libraries
---------------------------------------

`Pandas <https://pandas.pydata.org>`_
    Convert an Eelbrain :class:`Dataset` to a :class:`pandas.DataFrame` using :meth:`Dataset.as_dataframe`. Useful libraries: `Pingouin <https://pingouin-stats.org>`_ (statistics); `Seaborn <http://seaborn.pydata.org>`_ (plotting).
`R <http://r-project.org>`_
    When using R from Python through the :mod:`rpy2` bridge, transfer data between R ``data.frame`` and Eelbrain :class:`Dataset` using :meth:`Dataset.from_r` and :meth:`Dataset.to_r`.


MacOS: Framework Build
----------------------

On macOS, the GUI tool Eelbrain uses requires a special build of Python called
a "Framework build". You might see this error when trying to create a plot::

    SystemExit: This program needs access to the screen.
    Please run with a Framework build of python, and only when you are
    logged in on the main display of your Mac.

In order to avoid this, Eelbrain installs a shortcut to start `IPython
<ipython.readthedocs.io>`_ with a Framework build::

    $ eelbrain

This automatically launches IPython with the "eelbrain" profile. A default
startup script that executes ``from eelbrain import *`` is created, and can be
changed in the corresponding `IPython profile <http://ipython.readthedocs.io/
en/stable/interactive/tutorial.html?highlight=startup#startup-files>`_.


Quitting iPython
----------------

Sometimes iPython seems to get stuck after this line::

    Do you really want to exit ([y]/n)? y

In those instances, pressing ctrl-c usually terminates iPython immediately.


Windows: Scrolling
------------------

Scrolling inside a plot axes normally uses arrow keys, but this is currently
not possible on Windows (due to an issue in Matplotlib). Instead, the following
keys can be used:

+--------+--------+--------+
|        | ↑ ``i``|        |
+--------+--------+--------+
| ← ``j``|        | → ``l``|
+--------+--------+--------+
|        | ↓ ``k``|        |
+--------+--------+--------+
