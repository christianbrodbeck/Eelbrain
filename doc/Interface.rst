Eelbrain wxPython Interface
===========================

WxPython Gui
------------

The Eelbrain GUI is based on `wxPython <http://www.wxpython.org/>`_'s 
`PyShell <http://wiki.wxpython.org/PyShell>`_. 


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
such as saving and opening files, replace ``Ctrl`` with ``command``

Application Shortcuts
^^^^^^^^^^^^^^^^^^^^^

* ``Cmd`` - ``L``: Bring the shell window into focus and move the caret to the
  end of the prompt.
* ``Cmd`` - ``P``: Draw all plots (issue ``pyplot.draw()`` command)
* ``Cmd`` - ``S``:  Save the contents of the current window (works for Editor, 
  Shell and Matplotlib Figures).


Text Editor Shortcuts
^^^^^^^^^^^^^^^^^^^^^

(Valid in both shell and editor.)

* Display options:

  * ``Cmd`` - ``Shift`` - ``L``: Show/hide line numbers. 
  * Font properties:
    * ``Ctrl`` - ``]``: Increase font size.
    * ``Ctrl`` - ``[``: Decrease font size.
    * ``Ctrl`` - ``=``: Default font size.


* Moving the caret locally inside text:

  * ``home``:  Move the caret to the beginning of the current line.
  * ``Ctrl`` - ``Home``:  Move the caret to the beginning of the text.
  * ``End``:  Move the caret to the end of the current line.
  * ``Ctrl`` - ``End``:  Move the caret to the end of the text.
  * ``Shift`` (- ``Ctrl``) - ``Home``/``End``:  Select the text between the 
    current position and the target position.
  * ``Ctrl`` - ``F``:  Search 
  * ``F3``:  Search next
  * ``Ctrl`` - ``H``:  "Hide" lines containing selection / "unhide"


* Contextual Help:

  * ``F1``: Open help for the command under the caret.
  * ``Ctrl``-``space``: Bring up possible continuations for current command.
  * ``Shift``-``Ctrl``-``space``: Bring up quick help inside a call.


Shell
^^^^^

* History:

  * ``Ctrl`` - ``Up Arrow`` or ``Alt`` - ``P``:  Retrieve previous history item.
  * ``Ctrl`` - ``Down Arrow`` or ``Alt`` - ``N``:  Retrieve next history item.
  * ``Shift`` - ``Up Arrow``:  Insert previous history item.
  * ``Shift`` - ``Down Arrow``:  Insert next history item.
  * ``Shift`` - ``enter``: Choose command from history to add to the prompt.
  * ``F8``:  Command-completion of history item. (Type a few characters of a previous 
    command and press F8.)

* Copying Text:

  * ``Cmd`` - ``Return``:  Copy the command under the caret to the prompt.
  * ``Cmd`` - ``D``:  Duplicate the selected commands to the topmost editor window.
  * ``Cmd`` - ``Shift`` - ``D``:  Duplicate the selected text to the topmost 
    editor window (including command prompts and output).
  * ``Cmd`` - ``C``:  Copy selected text without prompts.
  * ``Cmd`` - ``Shift`` - ``C``:  Copy selected text, including prompts.
  * ``Cmd`` - ``X``:  Cut selected text.
  * ``Cmd`` - ``V``:  Paste from clipboard.
  * ``Cmd`` - ``Shift`` - ``V``:  Paste and run multiple commands from clipboard.

* Multi-Line Editing:

  * ``Ctrl`` - ``Enter``: Insert new line into multiline command.
  * ``F12``: Turn "free-edit" mode on/off.


Editor
^^^^^^

* ``Cmd`` - ``/``:  Comment or uncomment selected lines
* ``Cmd`` - ``D``:  Duplicate the selected command(s) to the shell prompt.
* ``Alt`` - ``arrow (up/down)``:  Move current line up or down (!!! uses copy-paste)
* Executing Code:

  * ``Ctrl`` - ``R``: Save the script and execute the whole script from disk. 
  * ``Ctrl`` - ``E``: Execute the selection. If nothing is selected, execute 
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

.. py:function:: cd([path])

    (Only with wxPython < 2.9) Called without argument: returns the current 
    working directory. Called with a path (as ``str``): changes the current 
    working directory. 

.. py:function:: help([object])

	open the help viewer for any object 

.. py:function:: printdict(dictionary)

	prints a more readable representation for complex dictionaries.


Startup Script
^^^^^^^^^^^^^^

Can be modified through the menu Eelbrain -> Preferences. 


Editor
------


Executing Scripts
^^^^^^^^^^^^^^^^^

There are 3 toolbar buttons to execute scripts: 

*  |exec-basic| executes the text of the script without saving it.
*  |exec-sel| (ctrl-e) executes only the selected text.
*  |exec-drive| (ctrl-r) saves the script and executes it form disk.

.. |exec-basic| image:: ../icons/actions/python-run.png
.. |exec-sel| image:: ../icons/actions/python-run-selection.png
.. |exec-drive| image:: ../icons/actions/python-run-drive.png

By default, scripts are executed in the global namespace of the shell. That 
means, an variables the script defines will be replaced in the shell. E.g.,
you type ``a=1`` in the shell, then run a script that includes a line ``a=2``,
and then inspect the value of ``a`` again in the shell, it will be ``2``.

This execution mode can be changed using the |exec-mode-public| toggle button.
When the button is in |exec-mode-private| mode, the script will be executed in 
a separate namespace and will not affect any variables defined in the shell
(it will also not have access to any of the variables in the shell, so e.g. all
required modules need to be imported in the script)

.. |exec-mode-private| image:: ../icons/actions/terminal-off.png
.. |exec-mode-public| image:: ../icons/actions/terminal-on.png

When any part of a script is executed, and the script is associated with a path
(i.e., has been loaded or saved), the current directory is automatically set to 
the folder containing the script before the script is executed. 

.. TODO: Toolbar buttons:
