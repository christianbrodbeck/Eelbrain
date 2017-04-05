***************
Getting Started
***************


Framework Build on macOS
------------------------

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
