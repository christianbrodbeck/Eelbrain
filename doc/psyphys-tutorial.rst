Psychophysiology Tutorial: Skin Conductance Responses
=====================================================

The tutorial is based on some simulated data which can be downloaded `here 
<http://dl.dropbox.com/u/659990/eelbrain_dist/simulated_scr.zip>`_. 
The complete scripts can be downloaded from here:

* `Import <http://dl.dropbox.com/u/659990/eelbrain_dist/tutorial_import.py>`_
* `Analysis <http://dl.dropbox.com/u/659990/eelbrain_dist/tutorial_analyze.py>`_


Overview
--------

A continuous stream of data is stored in a :class:`segment`. For example, in 
an experiment where skin conductance is recorded while stimuli are presented
on a computer screen, you will probably end up with two segments for each 
subject: one segment containing skin conductance, and one segment containing 
triggers recorded whenever a stimulus was presented. 

Multiple segments (usually one per subject) are collected in a :class:`dataset`:
In the example above you would have one dataset containing all skin conductance
segments, and another dataset containing all stimulus event segments. 

In eelbrain, data transformation are applied as child datasets. In the example,
you would add a child dataset that extracts skin conductance responses to the
skin conductance dataset. You would also add child datasets to the event 
dataset in order to add information about the stimuli (for example, you could
add a child dataset that counts the occurrence of each type of stimulus). 

This structure is illustrated in the :ref:`figure-guitree` figure: 
	
.. _figure-guitree:

.. figure:: _static/Tutorial_gui.png
	:alt: experiment outline
	:align: center
	:figwidth: 80%
	
	Example experiment tree
	
	The importer imports two datasets, events (Evt1) and Skin conductance (SC).
	The icons reflect the data type: the black list indicates discrete events,
	while the colorful lines indicate uniform time series data. 

Finally, you will want to use the events in the event dataset to define time 
windows to extract statistics from the skin conductance response dataset.
 

Importing Data
--------------

The first step is to import the psychophysiology module. Then, you can 
define an Experiment and add a data importer (for txt files in this case)::

    >>> import eelbrain.psyphys as pp
	>>> e = Experiment()
	>>> i = importer.txt(e)

When defining the txt importer, submitting ``e`` as the first argument assigns the 
importer as a direct child of the experiment. All datasets that manipulate 
data store the settings in an attribute called ``p``. We can look at the these 
settings of the importer by typing::

	>>> i.p

This shows all the settings that can be modified for this experiment_item 
under various headings::

    -<File Source>------------------------------------------------------------------
     source         : Extension: txt  (use set_ext() to change)
                      0 directories and 0 separate files
     vars_from_names: (No Variables Extracted)
                      (No example filenames available)
	
	-<Data Properties>--------------------------------------------------------------
	 t0             : 0
	
	-<UTS>--------------------------------------------------------------------------
	 channels       : No Channels in Data (select files first)
	 samplingrate   : 200
	
	-<Epoching>---------------------------------------------------------------------
	 epoch_length   : None
	 epoch_var      : None
	
	-<General>----------------------------------------------------------------------
	 color          : (0.078125, 0.578125, 0.99609375)

Some settings are accompanied only by their current value (e.g. ``samplingrate:
200``), whereas other settings also provide some instructions on their use (e.g.
``source``).
You can learn more about the function of individual parameters with::

    >>> i.p.HELP()

Each setting can be accessed individually through ``p.setting``. Again, 
simply typing the name 
prints out the settings::

    >>> i.p.source
    Extension: txt  (use set_ext() to change)
    0 directories and 0 separate files

We can now use the source parameter to tell the importer where to look for 
data files. We do this through the parameter ``source`` (the tutorial data 
files can be downloaded from `here 
<http://dl.dropbox.com/u/659990/eelbrain_dist/simulated_scr.zip>`_)::

	>>> i.p.source.set('/Users/christian/Data/simulated_scr')

Printing the parameter again shows that it has changed::

    >>> i.p.source
    Extension: txt  (use set_ext() to change)
    1 directories and 0 separate files:
    d: /Users/christian/Data/simulated_scr

.. Note:: You could also simply have called ``i.p.source.set()``. This would have opened a 
    system dialog and let you select the relevant folder. However, writing out the
    code has the advantage that you can save the script in the end and re-run it
    without manual intervention. 

.. Note:: You can add paths to the shell's 
    prompt by using either the menu command ``Insert-->Path-->Directory``, the 
    ``file`` dropdown menu in the toolbar, or 
    simply dragging the file from the system to the shell window.


After you specify the source folder, you can plot a preview of the data::

	>>> i.plot()

This should provide you with a figure like the following:


..	figure:: _static/Tutorial_1.png
	:alt: sample figure from importer.plot()
	:align: center
	
	Figure returned by importer.plot().
	
	This figure should help identifying the different data channels. The 
	channel numbers are indicated on the left side of the plot. 


Since the text files don't contain information on the samplingrate, we have to
manually specify it::

	>>> i.p.samplingrate = 200
	
.. Hint :: ``i.p.samplingrate = 200`` is equivalent to 
    ``i.p.samplingrate.set(200)``, although the ``set`` method's autocompletion
    feature might be useful. 

Next, we will specify which channels the importer should import::

    >>> i.p.channels[0] = 'events', 'evt'
    >>> i.p.channels[1] = 'skin_conductance', 'uts'

This parameter works like a Python dictionary. The keys (``0`` and ``1``)
specify the channel number, and the values (``'events', 'evt'`` and
``'skin_conductance', 'uts'``) the extraction parameters. ``uts`` stands for
uniform time-series, i.e., a signal that is sampled at regular intervals in 
time. ``evt`` stands for events, i.e., samples occurred at arbitrary points
in time and time has to be listed for each sample.   

.. Note:: The names that you assign to the extracted channels (the first 
	argument, i.e. ``'events'`` and ``'skin_conductance'`` are going to be used as
	channel names, so they can only contain alphanumeric characters and underlines. 

.. Hint :: In order to get more help for a specific parameter 
    type, look at the documentation for that parameter: either use 
    ``help(i.p.channels)``, or type ``i.p.channels`` and hit ``f1``.

You can call ``i.plot()`` again to check the settings. Channels are colored to
illustrate extraction settings (uts data: black, events: blue).

Next, since our filenames contain subject identifiers, we can use
this information. We can look at the names by calling the relevant parameter::

	>>> i.p.vars_from_names
	(No Variables Extracted)
	  0123456
	  001.txt  
	  002.txt  
	  003.txt  
	  004.txt  
	  005.txt  
	  ...

We see that the first 3 characters indicate the subject identifier. Thus, we 
extract the first three characters and name the variable 'subject'. Printing 
the parameter again shows the effect::

	>>> i.p.vars_from_names[:3] = 'subject'
	>>> i.p.vars_from_names
	index   name
	------------------
	(0, 3)  subject
	
	  0123456   subject
	  001.txt         001
	  002.txt         002
	  003.txt         003
	  004.txt         004
	  005.txt         005
	  ...

Now we are ready to import the data. However, since we want the script to be 
reproducible without human interference, we save the experiment before 
importing the data (this is necessary because eelbrain needs to know where to
store the imported data)::

	>>> e.saveas('/Users/christian/Data/tutorial_scr')

Now we can import the data::

	>>> i.get()

(this might take a while).


Saving the Procedure as Python Script
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

While saving the experiment will save the result of what you just did, you 
might also want to keep the script to produce this result. The script is much 
smaller file and can reproduce the results from the raw data. In addition, when
the raw data changes (e.g. more subjects are added), often simply rerunning the 
script can incorporate the new data.

..	Note:: A helpful keyboard shortcut in this respect is to select the 
	desired lines in the shell and press ``ctrl-d``. This copies the lines to the 
	frontmost Python editor (or creates a new editor if none is open). You can 
	select a large section in the shell, since only the actual commands are 
	copied.

..	Note:: In script files you can also use relative paths (e.g., 
    ``"../data"``). This only works after the script has been saved, 
    since then the system path is set to the directory containing the script 
    when the script is executed.


Inspecting Data
---------------

In the Shell
^^^^^^^^^^^^

The experiment instance contains as attributes references to each dataset. 
These can be seen using the print command::

    >>> print e
    |importer
    | |event
    | 
    |skin_conductance

Each dataset contains its segments in the segment attribute, which acts like a 
list of segments::
    
    >>> len(e.skin_conductance.segments)
    20
    >>> e.skin_conductance.segments[0]
    UTS_Segment("001.txt", uts)

There are two types of segments: 
For uts-data segments, the data itself can be retrieved as the data attribute:

    >>> segment = e.skin_conductance.segments[0]
    >>> segment.data
    memmap([[ 1.      ],
           [ 0.99    ],
           [ 0.9851  ],
           ..., 
           [ 0.070447],
           [ 0.073286],
           [ 0.076122]])
    >>> segment.data.shape
    (28000, 1)
    >>> type(segment.data)
    <class 'numpy.core.memmap.memmap'>

For event-segments, the data actual can also be accessed through the data 
attribute, but the string representation (retrieved by the print function)
is more readable::

    >>> e.event[0]  # (a short-cut for e.event.segments[0])
    Event_Segment("001.txt", event)
    >>> e.event[0].data
    memmap([[  10.,    6.,    4.],
           [  25.,    6.,    5.],
           [  40.,    6.,    4.],
           [  55.,    6.,    5.],
           [  70.,    6.,    4.],
           [  85.,    6.,    5.],
           [ 110.,    6.,    4.],
           [ 125.,    6.,    5.]])
    >>> print e.event[0]
        time   duration   magnitude
    -------------------------------
    0   10     6          4        
    1   25     6          5        
    2   40     6          4        
    3   55     6          5        
    4   70     6          4        
    5   85     6          5        
    6   110    6          4        
    7   125    6          5        


GUIs
^^^^

There are also GUI elements based on wxpython. The dataset hierarchy of an 
experiment can be seen in an experiment frame (which at the moment does not
do much apart from that)::

    >>> import eelbrain.wxgui.psyphys as ppgui
    >>> ppgui.frame_experiment(e)

As you can see, the ``txt`` importer has two children with the names you 
specified earlier (``events`` and ``skin_conductance``). Their icons reflect 
the data type. The GUI does provide a convenient button to save the experiment 
in the Toolbar.

..  
    Note:: 
    hover the mouse pointer over any toolbar buttons to get information
    about its function)

Data can be visualized with a :ref:`figure-list-viewer`::

    >>> v = ppgui.list(e.skin_conductance, e.event)
    
.. _figure-list-viewer:

.. figure:: _static/Tutorial_list-viewer1.png
    :alt: experiment outline
    :align: center
    :figwidth: 100%
    
    List Viewer
    
    A list viewer displaying the tutorial data. The viewer only displays 2 
    plots per page, which is achieved through the keyword-argument ``y=2``
    (using ``>>> v = ppgui.list(e.skin_conductance, e.event, y=2)``).

While the viewer that opens has a toolbar with a few controls, more controls 
are available through the shell. That is why we assigned the viewer to a short 
variable (``v``). For example, use the following command to restrict the view
to a certain time range::

    >>> v.set_window(20, 60)

You can also change the source data parameters while the viewer is open::

    >>> e.event.p.color((1, 0, 0))

In order to see the changes, however, you need to refresh |view-refresh| the 
viewer.

.. |VIEW-REFRESH| image:: ../../icons/tango/actions/view-refresh.png



Signal Processing
-----------------

Our next step is to extract the skin conductance responses (SCRs) from the raw
data. Any data transformations are applied as child datasets in eelbrain. All
possible operations are available through the ``psyphys.op`` module (short for 
"operation"). To extract the SCRs, use::

	>>> d = pp.op.physio.SCR(e.skin_conductance, name='SCRs') 

We assign the new dataset to the variable ``d`` to have easier access to 
the new dataset. All datasets can also be access as attribute of their parent 
experiment, which you can confirm with::

    >>> d is e.SCRs
    True

Just as the importer, the new dataset has parameters that can 
be adjusted in its ``p`` attribute (``e.SCRs.p``). 
We can leave them at the default settings for the present purpose.

Now you can inspect the result in the list viewer::

    >>> v = ppgui.list(e.skin_conductance, e.SCRs, e.event)


Event Processing
----------------

Similar to data segments, event segments can be elaborated. In order to 
examine sequence effects, we want to add a trial counter to the event-
segments::

    >>> d = pp.op.evt.Enum(e.event, 'event2_enum') 
    >>> d.p.var = 'trial'

The result can be seen by looking at one of the segments::

    >>> print e.event2_enum[0]
        time   duration   magnitude   trial
    ---------------------------------------
    0   10     6          4           0    
    1   25     6          5           1    
    2   40     6          4           2    
    3   55     6          5           3    
    4   70     6          4           4    
    5   85     6          5           5    
    6   110    6          4           6    
    7   125    6          5           7    

This counts each single event. However, it might be more useful to count 
events of each condition (coded in ``magnitude``) separately. This can be 
achieved through the ``count`` parameter, which specifies which 
events should be counted:: 

    >>> d.p.count = 'magnitude'
    >>> print e.event2_enum[0]
        time   duration   magnitude   trial
    ---------------------------------------
    0   10     6          4           0    
    1   25     6          5           0    
    2   40     6          4           1    
    3   55     6          5           1    
    4   70     6          4           2    
    5   85     6          5           2    
    6   110    6          4           3    
    7   125    6          5           3    
    
..  Note:: to learn more about the parameters you could use ``d.p.HELP()`` or
    ``help(d)``.

..
    Note:: The dataset hierarchy in eelbrain is structured in such a way that when
    you modify parameters, the changes automatically propagate to the datasets
    which are lower in the hierarchy.


Statistics
----------

Collecting Statistics
^^^^^^^^^^^^^^^^^^^^^

The :py:func:`!psyphys.collect.timewindow` can be used to collect statistics 
from the experiment that we built up in the earlier part of the tutorial. 
Using the variables contained in the experiment, we can construct a 
model for which we want to collect statistics
(using the :py:func:`attach` function for convenience)::

    >>> attach(e.variables)
    attached: ['subject', 'time', 'duration', 'magnitude', 'trial']
    >>> subject + magnitude
    Address(subject + magnitude)

Crossing subjects and magnitude will collect a statistic for each cell in this 
model. Collect the statistics in a dataset::

    >>> ds = pp.collect.timewindow(subject * magnitude, e.SCRs, e.event, tstart=.1, tend=.6)

A :py:class:`~vessels.data.dataset` stores a data table containing multiple 
variables, and works like a dictionary::

    >>> ds
    <dataset '???' N=40: 'Y'(V), 'magnitude'(V), 'subject'(F)
    >>> ds['Y']
    var([0.27, 0.00, 0.00, 0.07, 0.06, ... n=40], name='Y')
    >>> ds['subject']
    factor([0, 0, 1, 1, 2, ...n=40], name="subject", random=True, labels={0: u'001', 1: u'002', 2: u'003', 3: u'004', 4: u'005', ...})

The dataset contains :py:class:`~vessels.data.var` and 
:py:class:`~vessels.data.factor` objects, which correspond to scalar and 
categorical variables. The table can be shown with ``print``::

    >>> print ds
    Y          magnitude   subject
    ------------------------------
    0.27063    5           001    
    0          4           001    
    0          4           002    
    0.069958   5           002    
    0.05791    5           003    
    0          4           003    
    0.16843    4           004    
    0          5           004    
    0          5           005    
    0          4           005    
         (use .as_table() method to see the whole dataset)

A dataset can be retrieved as table object, and any table object can be 
exported as tab-separated values (tsv) file::

    >>> t = ds.as_table()
    >>> t.save_tsv() # saving without path argument opens save-dialog

That way, the data can be analyzed in any statistics package. Eelbrain also 
contains some functions for statistical analysis and plotting, which is
illustrated in the next section. 


Analyzing Statistics
^^^^^^^^^^^^^^^^^^^^

    
The :py:mod:`eelbrain.analyze` module contains functions for analyzing the 
resulting dataset::

    >>> import eelbrain.analyze as A
    >>> attach(ds)
    >>> fig = A.plot.boxplot(Y, magnitude, match=subject)
    >>> print A.test.pairwise(Y, magnitude, match=subject)
    
    Pairwise t-Tests (paired samples)
    
        5              
    -------------------
    4   t(19)=-2.95**  
        p=0.008        
        p(c)=.008      
    (* Uncorrected)

..  Note:: These functions are called with 2 arguments: the dependent variable,
    and the model (which in this case is only ``magnitude``). The ``match``
    keyword argument specifies the variable on which the data is related (for 
    the related samples t-test).
    
