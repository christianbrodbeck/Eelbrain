Psychophysiology Tutorial: Skin Conductance Responses
=====================================================

The tutorial is based on some simulated data which can be downloaded `here 
<http://dl.dropbox.com/u/659990/eelbrain_doc/files/simulated_scr.zip>`_. 
The complete scripts can be downloaded from here:

* `Import <http://dl.dropbox.com/u/659990/eelbrain_doc/files/tutorial_import.py>`_
* `Analysis <http://dl.dropbox.com/u/659990/eelbrain_doc/files/tutorial_analyze.py>`_
* `Plot timeseries comparison <http://dl.dropbox.com/u/659990/eelbrain_doc/files/tutorial_analyze_ts.py>`_


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
	>>> e = pp.Experiment()
	>>> i = pp.importer.txt(e)

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
<http://dl.dropbox.com/u/659990/eelbrain_doc/files/simulated_scr.zip>`_)::

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

Similar to data segments, event segments can be elaborated. First, when 
inspcting the data the first time, we saw that the event magnitude in the 
source files is represented as a scalar variable::

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

It is convenient to have a categorial variable reflecting the event condition.
Any variable transformation where values in the source variable(s) map to 
values in the target variable can be implemented using a parasite variable::

    >>> attach(e.variables)
    attached: ['subject', 'time', 'duration', 'magnitude']
    >>> e.variables.new_parasite(magnitude, 'condition', 'dict', {4:'control', 5:'test'})
    <Parasite:  'condition' <- magnitude, 'dict', labels={0: 'control', 1: 'test'}>
    >>> detach()
    >>> print e.event[0]
        time   duration   magnitude   condition
    -------------------------------------------
    0   10     6          4           control  
    1   25     6          5           test     
    2   40     6          4           control  
    3   55     6          5           test     
    4   70     6          4           control  
    5   85     6          5           test     
    6   110    6          4           control  
    7   125    6          5           test     

.. Note:: Parasite variables are automatically included in data tables when
    all the variables  they are based on are present. 

In order to 
examine sequence effects, we might want to add a trial counter to the event-
segments::

    >>> d = pp.op.evt.Enum(e.event, 'event2_enum') 
    >>> d.p.var = 'trial'

The result can be seen by looking at one of the segments again::

    >>> print d[0]
        time   duration   magnitude   trial   condition
    ---------------------------------------------------
    0   10     6          4           0       control  
    1   25     6          5           1       test     
    2   40     6          4           2       control  
    3   55     6          5           3       test     
    4   70     6          4           4       control  
    5   85     6          5           5       test     
    6   110    6          4           6       control  
    7   125    6          5           7       test     

This counts each single event. However, it might be more useful to count 
events of each condition (coded in ``magnitude``) separately. This can be 
achieved through the ``count`` parameter, which specifies which 
events should be counted:: 

    >>> d.p.count = 'magnitude'
    >>> print d[0]
        time   duration   magnitude   trial   condition
    ---------------------------------------------------
    0   10     6          4           0       control  
    1   25     6          5           0       test     
    2   40     6          4           1       control  
    3   55     6          5           1       test     
    4   70     6          4           2       control  
    5   85     6          5           2       test     
    6   110    6          4           3       control  
    7   125    6          5           3       test     
    
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

The :py:mod:`!psyphys.collect` module contains tools to  collect statistics 
from the datasets. There are two main functions:

 - :py:func:`!psyphys.collect.timeseries` to collect timeseries data (e.g., 
   the temporal evolution of heart-rate or skin-conductance responses around
   different events.
 - :py:func:`!psyphys.collect.timewindow` to collect scalar dependent
   variables, e.g. the maximum SCR in the time window .05 to .45 seconds
   after different events.

In a first step, :py:func:`!psyphys.collect.timeseries` can be used to explore
the data.

Using the variables contained in the experiment, we can construct a 
model for which we want to collect statistics
(using the :py:func:`attach` function for convenience)::

    >>> attach(e.variables)
    attached: ['subject', 'time', 'duration', 'magnitude', 'trial']
    >>> subject + magnitude
    Address(subject + magnitude)

Crossing subjects and magnitude will collect a statistic for each cell in this 
model::

    >>> ds = pp.collect.timeseries(subject * condition, e.SCRs, e.event, 
    ...                            tstart=-.2, tend=1.5, sr=20, 
    ...                            mode='mw', windur=.5)

We will extract a time-series from -.2 seconds before the cue (``tstart``) 
until 1.5 seconds after it (``tend``), at a samplingrate (``sr``) of 20 Hz.
The ``mode`` kwarg ``'mw'`` indicates moving window which is appropriate for
discrete events like SCRs (for a continuous rate, e.g., heart rate, we would 
choose ``'lin'`` for 'linear'). Finally, ``windur`` specifies the length of 
the window used, which will determine the smoothness of the final time series.
The result can be plotted with::

    >>> detach()
    >>> attach(ds)
    >>> from eelbrain.eellab import *
    >>> plot.uts.stat(Y, condition)

This plot can suggest good time-windows for further analysis with
:py:func:`!psyphys.collect.timewindow`::

    >>> ds = pp.collect.timewindow(subject * condition, e.SCRs, e.event, tstart=.1, tend=.6)

Collectors return their result in the form of a :py:class:`~vessels.data.dataset`. 
A :py:class:`~vessels.data.dataset` stores a data table containing multiple 
variables, and works like a dictionary::

    >>> ds
    <dataset '???' N=40: 'Y'(V), 'condition'(F), 'subject'(F)>
    >>> ds['Y']
    var([0.27, 0.00, 0.00, 0.07, 0.06, ... n=40], name='Y')
    >>> ds['subject']
    factor([0, 0, 1, 1, 2, ...n=40], name="subject", random=True, labels={0: u'001', 1: u'002', 2: u'003', 3: u'004', 4: u'005', ...})

The dataset contains :py:class:`~vessels.data.var` and 
:py:class:`~vessels.data.factor` objects, which correspond to scalar and 
categorical variables. The table can be shown with ``print``::

    >>> print ds
    Y           subject   condition
    -------------------------------
    0.010823    019       control  
    0.84226     013       test     
    0.54688     010       test     
    0.16843     004       control  
    0           009       control  
    0.05791     003       test     
    0.20071     016       test     
    0.069872    015       control  
    0           020       control  
    0.086678    006       test     
    0.084659    019       test     
    0           005       control  
    0           010       control  
    0           009       test     
    0           016       control  
    0.027371    015       test     
    0           012       test     
    0.012217    006       control  
    0           011       control  
    0           005       test     
    0.069958    002       test     
    0           001       control  
    0.0040346   008       test     
    0.007754    007       control  
    1.0231      018       test     
    0           012       control  
    0.14718     017       control  
    0.12535     011       test     
    0           002       control  
    0.27063     001       test     
    0.32829     014       test     
    0.20724     008       control  
    0.15325     013       control  
    0.4314      007       test     
    0           004       test     
    0.35768     017       test     
    0           003       control  
    0.048485    020       test     
    0.24732     018       control  
    0           014       control  


A dataset can be retrieved as table object, and any table object can be 
exported as tab-separated values (tsv) file::

    >>> t = ds.as_table()
    >>> t.save_tsv() # saving without path argument opens a system file dialog

That way, the data can be analyzed in any statistics package. Eelbrain also 
contains some functions for statistical analysis and plotting, which is
illustrated in the next section. 


Analyzing Statistics
^^^^^^^^^^^^^^^^^^^^

    
The :py:mod:`eelbrain.eellab` module contains functions for analyzing the 
resulting dataset::

    >>> from eelbrain.eellab import *
    >>> attach(ds)
    attached: ['Y', 'condition', 'subject']
    >>> fig = plot.uv.boxplot(Y, condition, match=subject)
    >>> print test.pairwise(Y, condition, match=subject)
    
    Pairwise t-Tests (paired samples)
    
              test           
    -------------------------
    control   t(19)=-2.95**  
              p=0.008        
              p(c)=.008      
    (* Uncorrected)

..  Note:: These functions are called with 2 arguments: the dependent variable,
    and the model (which in this case is only ``magnitude``). The ``match``
    keyword argument specifies the variable on which the data is related (for 
    the related samples t-test).
    
