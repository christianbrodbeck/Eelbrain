Eelbrain wxPython Interface
===========================

WxPython Gui
------------

The Eelbrain GUI is based on `wxPython <http://www.wxpython.org/>`_. 


Getting Help
^^^^^^^^^^^^

* Pressing the ``f1`` key will open the help viewer. If the terminal or an 
  editor is currently active, the object help viewer will show the 
  docomentation for the object currently under the cursor. Alternatvely, you 
  can call ``>>> help(object)``.
* Many GUI elements (e.g., toolbar buttons) have a short-help which can be
  displayed by hovering the mouse pointer over it.


Debugging
^^^^^^^^^

If something crashes you can start the debugger with::

	>>> import pdb
	>>> pdb.pm()
	
To end the debugger use ``q`` for quit::

	(Pdb) q

For more information on pdb see the `pdb documentation <http://docs.python.org/library/pdb.html>`_.



Keyboard Shortcuts
------------------

Some keyboard shortcuts are available in all windows, others are available 
depending on the type of the active window. On Os-X, for many system commands
such as saving and opening files, replace ``ctrl`` with ``command``


Display
^^^^^^^

* ``cmd`` - ``shift`` - ``L``: Show/hide line numbers. 
* Font:

	* ``ctrl`` - ``]``: Increase font size.
	* ``ctrl`` - ``[``: Decrease font size.
	* ``ctrl`` - ``=``: Default font size.

* Plotting

    * ``ctrl`` - ``p``: Draw plots (``pyplot.draw()``)


Entering Text
^^^^^^^^^^^^^

These shortcuts are relevant when entering text (in the shell or an editor)

* ``F1``: Open help for the command under the caret; in the help viewer the whole 
	object name must be selected.

* Moving the caret (cursor):

	* ``home``:  Go to the beginning of the command or line.
	* ``shift`` - ``Home``:  Select to the beginning of the command or line.
	* ``shift`` - ``End``:  Select to the end of the line.
	* ``end``:  Go to the end of the line.
	* ``ctrl`` - ``F``:  Search 
	* ``F3``:  Search next
	* ``ctrl`` - ``H``:  "Hide" lines containing selection / "unhide"

* Auto Completion:

	* ``ctrl``-``space``: Auto completion of attribute.
	* ``shift``-``ctrl``-``space``: Auto completion inside call.

* Multi-Line Editing:

	* ``ctrl`` - ``Enter``: Insert new line into multiline command.
	* ``F12``: Turn "free-edit" mode on/off.


Shell
^^^^^

* History:

    * ``ctrl`` - ``Up Arrow`` or ``alt`` - ``p``:  Retrieve previous history item.
    * ``ctrl`` - ``Down Arrow`` or ``alt`` - ``n``:  Retrieve next history item.
    * ``shift`` - ``Up Arrow``:  Insert previous history item.
    * ``shift`` - ``Down Arrow``:  Insert next history item.
    * ``shift`` - ``enter``: Choose command from history to add to the prompt.
    * ``F8``:  Command-completion of history item. (Type a few characters of a previous 
      command and press F8.)

* Copying Text:

	* ``ctrl`` - ``d``:  Copy the selected commands to the topmost editor window
	* ``ctrl`` - ``c``:  Copy selected text without prompts.
	* ``ctrl`` - ``Shift`` - ``C``:  Copy selected text, retaining prompts.
	* ``ctrl`` - ``X``:  Cut selected text.
	* ``ctrl`` - ``V``:  Paste from clipboard.
	* ``ctrl`` - ``Shift`` - ``V``:  Paste and run multiple commands from clipboard.


Editor
^^^^^^

* ``ctrl`` - ``/``:  Comment or uncomment selected lines
* ``ctrl`` - ``s``:  Save current document
* ``alt`` - ``arrow (up/down)``:  Move current line up or down (!!! uses copy-paste)
* Executing Code:

	* ``ctrl`` - ``r``: Save the script and execute the whole script from disk. 
	* ``ctrl`` - ``e``: Execute the selection. If nothing is selected, execute 
	  the current line.


Shell
-----

Functions
^^^^^^^^^

The following commands are available in the shell in addition to normal Python
commands. For more information, use help(command):

.. py:function:: attach(dictionary)

    Updates the global namespace with ``dictionary``, as can be shown with
    a locally defined dictionary::
    
        >>> a
        Traceback (most recent call last):
             File "<input>", line 1, in <module>
           NameError: name 'a' is not defined
           
        >>> attach({'a': 'something'})
        attached: ['a']
        >>> a
        'something'

    Many dictionary-like Eelbrain objects can be attached like that for 
    convenient access, for example: experiment.variables, datasets. The wxterm
    shell will keep track of any attached variables and
    :py:func:`detach` will remove any variables that were attached using 
    this function from the global namespace.  
	 

.. py:function:: detach()

    remove from the global namespace any variables that were added to it 
    using the :py:func:`attach` function.

.. py:function:: curdir([path])

    Called without argument: returns the current working directory. Called with
    a path (as ``str``): changes the current working directory. 

.. py:function:: help([object])

	open the help viewer for any object 

.. py:function:: printdict(dictionary)

	prints a more readable representation for complex dictionaries.


Startup Script
^^^^^^^^^^^^^^

Through the menu Eelbrain->Preferences..., a ``dataDir`` can be set. If this 
dataDir contains a Python script named ``'startup'`` (note: no extension), this
script is executed every time the shell starts up (this is a feature of the
:py:class:`wx.py.shell.ShellFrame <http://www.wxpython.org/docs/api/wx.py.shell.ShellFrame-class.html>`). 


Editor
------

Executing Scripts
^^^^^^^^^^^^^^^^^

By default, scripts are executed in the global namespace of the shell. That 
means, an variables the script defines will be replaced in the shell. E.g.,
you type ``a=1`` in the shell, then run a script that includes a line ``a=2``,
and then inspect the value of ``a`` again in the shell, it will be ``2``.

This execution mode can be changed using the |exec-mode-public| toggle button.
When the button is in |exec-mode-private| mode, the script will be executed in 
a separate namespace and will not affect any variables defined in the shell
(it will also not have access to any of the variables in the shell, so e.g. all
required modules need to be imported in the script)

.. |exec-mode-private| image:: ../../icons/actions/terminal-off.png
.. |exec-mode-public| image:: ../../icons/actions/terminal-on.png

When any part of a script is executed, and the script is associated with a path
(i.e., has been loaded or saved), the current directory is automatically set to 
the folder containing the script before the script is executed. 

.. TODO: Toolbar buttons:
