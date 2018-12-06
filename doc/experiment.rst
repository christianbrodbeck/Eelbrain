.. currentmodule:: eelbrain

.. _experiment-class-guide:

********************************
The :class:`MneExperiment` Class
********************************

MneExperiment is a base class for managing data analysis for an MEG
experiment with MNE.

.. seealso::
    :class:`MneExperiment` class reference for details on all available methods

.. contents:: Contents
   :local:


Step by step
============

.. contents:: Contents
   :local:


.. _MneExperiment-filestructure:

Setting up the file structure
-----------------------------

.. py:attribute:: MneExperiment.sessions

The first step is to define an :class:`MneExperiment` subclass with the name
of the experiment::

    from eelbrain import *

    class WordExperiment(MneExperiment):

        sessions = 'words'


Where ``sessions`` is the name which you included in your raw data files after
the subject identifier.
Once this basic experiment class is defined, it can be initialized without root
(i.e., without data files). This is useful to see the required file structure::

    >>> e = WordExperiment()
    >>> e.show_input_tree()
    root
    mri-sdir                                /mri
    mri-dir                                    /{mrisubject}
    meg-sdir                                /meg
    meg-dir                                    /{subject}
    raw-dir
    trans-file                                       /{mrisubject}-trans.fif
    raw-file                                         /{subject}_{session}-raw.fif


This output shows a template for the path structure according to which the input
files have to be organized. Assuming that ``root="/files"``, for a subject
called "R0001" this includes:

- MRI-directory at ``/files/mri/R0001``
- the raw data file at ``/files/meg/R0001/R0001_words-raw.fif`` (the
  session is called "words" which is specified in ``WordExperiment.sessions``)
- the trans-file from the coregistration at ``/files/meg/R0001/R0001-trans.fif``

Once the required files are placed in this structure, the experiment class can
be initialized with the proper root parameter, pointing to where the files are
located::

    >>> e = WordExperiment("/files")


The setup can be tested using :meth:`MneExperiment.show_subjects`, which shows
a list of the subjects that were discovered and the MRIs used::

    >>> e.show_subjects()
    #    subject   mri
    -----------------------------------------
    0    R0026     R0026
    1    R0040     fsaverage * 0.92
    2    R0176     fsaverage * 0.954746600461
    ...


.. _MneExperiment-preprocessing:

Pre-processing
--------------

Make sure an appropriate pre-processing pipeline is defined as
:attr:`MneExperiment.raw`.

To inspect raw data for a given pre-processing stage use::

    >>> e.set(raw='1-40')
    >>> y = e.load_raw(ndvar=True)
    >>> p = plot.TopoButterfly(y, xlim=5)

Which will plot 5 s excerpts and allow scrolling through the data.


Labeling events
---------------

Initially, events are only labeled with the trigger ID. Use the
:attr:`MneExperiment.variables` settings to add labels.
For more complex designs and variables, you can override
:meth:`MneExperiment.label_events`.
Events are represented as :class:`Dataset` objects and can be inspected with
corresponding methods and functions, for example::

    >>> e = WordExperiment("/files")
    >>> ds = e.load_events()
    >>> ds.head()
    >>> print(table.frequencies('trigger', ds=ds))


Defining data epochs
--------------------

Once events are properly labeled, define :attr:`MneExperiment.epochs`.

There is one special epoch to define, which is called ``'cov'``. This is the
data epoch that will be used to estimate the sensor noise covariance matrix for
source estimation.

In order to find the right ``sel`` epoch parameter, it can be useful to actually
load the events with :meth:`MneExperiment.load_events` and test different
selection strings. The epoch selection is determined by
``selection = event_ds.eval(epoch['sel'])``. Thus, a specific setting could be
tested with::

    >>> ds = e.load_events()
    >>> print(ds.sub("event == 'value'"))


Bad channels
------------

Flat channels are automatically excluded from the analysis.

An initial check for noisy channels can be done by looking at the raw data (see
:ref:`MneExperiment-preprocessing` above).
If this inspection reveals bad channels, they can be excluded using
:meth:`MneExperiment.make_bad_channels`.

Another good check for bad channels is plotting the average evoked response,,
and looking for channels which are uncorrelated with neighboring
channels. To plot the average before trial rejection, use::

    >>> ds = e.load_epochs(epoch='epoch', reject=False)
    >>> plot.TopoButterfly('meg', ds=ds)

The neighbor correlation can also be quantified, using::

    >>> nc = neighbor_correlation(concatenate(ds['meg']))
    >>> nc.sensor.names[nc < 0.3]
    Datalist([u'MEG 099'])

A simple way to cycle through subjects when performing a given pre-processing
step is :meth:`MneExperiment.next`.


ICA
---

If preprocessing includes ICA, select which ICA components should be removed.
The experiment ``raw`` state needs to be set to the ICA stage of the pipeline::

    >>> e.set(raw='ica')
    >>> e.make_ica_selection(epoch='epoch', decim=10)

Set ``epoch`` to the epoch whose data you want to display in the GUI (see
:meth:`MneExperiment.make_ica_selection` for more information, in particular on
how to precompute ICA decomposition for all subjects).

In order to select ICA components for multiple subject, a simple way to cycle
through subjects is :meth:`MneExperiment.next`, like::

    >>> e.make_ica_selection(epoch='epoch', decim=10)
    >>> e.next()
    subject: 'R1801' -> 'R2079'
    >>> e.make_ica_selection(epoch='epoch', decim=10)
    >>> e.next()
    subject: 'R2079' -> 'R2085'
    ...


Trial selection
---------------

For each primary epoch that is defined, bad trials can be rejected using
:meth:`MneExperiment.make_epoch_selection`. Rejections are specific to a given ``raw``
state::

    >>> e.set(raw='ica1-40')
    >>> e.make_epoch_selection()
    >>> e.next()
    subject: 'R1801' -> 'R2079'
    >>> e.make_epoch_selection()
    ...

To reject trials based on a pre-determined threshold, a loop can be used::

    >>> for subject in e:
    ...     e.make_epoch_selection(auto=1e-12)
    ...


.. _MneExperiment-intro-analysis:

Analysis
--------

Finally, define :attr:`MneExperiment.tests` and create a ``make-reports.py``
script so that all reports can be updated by running a single script
(see :ref:`MneExperiment-example`).

.. Warning::
    If source files are changed (raw files, epoch rejection or bad channel
    files, ...) reports are not updated unless the corresponding
    :meth:`MneExperiment.make_report` function is called again. For this reason
    it is useful to have a script that calls :meth:`MneExperiment.make_report`
    for all desired reports. Running the script ensures that all reports are
    up-to-date, and will only take seconds if nothing has to be recomputed.


.. _MneExperiment-example:

Example
=======

The following is a complete example for an experiment class definition file
(the source file can be found in the Eelbrain examples folder at
``examples/experiment/sample_experiment.py``):

.. literalinclude:: ../examples/experiment/sample_experiment.py


Given the ``SampleExperiment`` class definition above, the following is a
script that would compute/update analysis reports:

.. literalinclude:: ../examples/experiment/make_reports.py


Experiment Definition
=====================

.. contents:: Contents
   :local:


Basic setup
-----------

.. py:attribute:: MneExperiment.owner

Set :attr:`MneExperiment.owner` to your email address if you want to be able to
receive notifications. Whenever you run a sequence of commands ``with
mne_experiment.notification:`` you will get an email once the respective code
has finished executing or run into an error, for example::

    >>> e = MyExperiment()
    >>> with e.notification:
    ...     e.make_report('mytest', tstart=0.1, tstop=0.3)
    ...

will send you an email as soon as the report is finished (or the program
encountered an error)

.. py:attribute:: MneExperiment.auto_delete_results

Whenever a :class:`MneExperiment` instance is initialized with a valid
``root`` path, it checks whether changes in the class definition invalidate
previously computed results. By default, the user is prompted to confirm
the deletion of invalidated results. Set ``.auto_delete_results=True`` to
delete them automatically without interrupting initialization.

.. py:attribute:: MneExperiment.screen_log_level

Determines the amount of information displayed on the screen while using
an :class:`MneExperiment` (see :mod:`logging`).

.. py:attribute:: MneExperiment.meg_system

Starting with :mod:`mne` 0.13, fiff files converted from KIT files store
information about the system they were collected with. For files converted
earlier, the :attr:`MneExperiment.meg_system` attribute needs to specify the
system the data were collected with. For data from NYU New York, the
correct value is ``meg_system="KIT-157"``.

.. py:attribute:: MneExperiment.trigger_shift

Set this attribute to shift all trigger times by a constant (in seconds). For
example, with ``trigger_shift = 0.03`` a trigger that originally occurred
35.10 seconds into the recording will be shifted to 35.13. If the trigger delay
differs between subjects, this attribute can also be a dictionary mapping
subject names to shift values, e.g.
``trigger_shift = {'R0001': 0.02, 'R0002': 0.05, ...}``.


Subjects
--------

.. py:attribute:: MneExperiment.subject_re

Subjects are identified by looking for folders in the subjects-directory whose
name matches the ``subject_re`` regular expression (see :mod:`re`). By
default, this is ``'(R|A|Y|AD|QP)(\d{3,})$'``, which matches R-numbers like
``R1234``, but also numbers prefixed by ``A``, ``Y``, ``AD`` or ``QP``.


Defaults
--------

.. py:attribute:: MneExperiment.defaults

The defaults dictionary can contain default settings for
experiment analysis parameters, e.g.::

    defaults = {'epoch': 'my_epoch',
                'cov': 'noreg',
                'raw': '1-40'}


Pre-processing (raw)
--------------------

.. py:attribute:: MneExperiment.raw

Define a pre-processing pipeline as a series of processing steps:

.. currentmodule:: eelbrain.pipeline

.. autosummary::
   :toctree: generated
   :template: class_nomethods.rst

   RawFilter
   RawICA
   RawMaxwell
   RawSource
   RawReReference


- Each preprocessing step is defined with its input as first argument.
- If using FIFF files, no ``RawSource`` pipe is neded, and the raw data can be
  accessed as ``"raw"`` input.
- :mod:`mne` has changed default values for filtering in the past. In order to
  keep consistent settings across different versions it is advantageous to fully
  define filter parameters when starting a new experiment.

For example, to use TSSS and a band-pass, and optionally ICA::

    # as of mne 0.17
    FILTER_KWARGS = {
        'filter_length': 'auto',
        'l_trans_bandwidth': 'auto',
        'h_trans_bandwidth': 'auto',
        'phase': 'zero',
        'fir_window': 'hamming',
        'fir_design': 'firwin',
    }


    class Experiment(MneExperiment):

        sessions = 'session'

        raw = {
            'tsss': RawMaxwell('raw', st_duration=10., ignore_ref=True, st_correlation=0.9, st_only=True),
            '1-40': RawFilter('tsss', 1, 40, **FILTER_KWARGS),
            'ica': RawICA('tsss', 'session', 'extended-infomax', n_components=0.99),
            'ica1-40': RawFilter('ica', 1, 40, **FILTER_KWARGS),
        }


Event variables
---------------

.. py:attribute:: MneExperiment.variables

Event variables add labels and variables to the events:

.. autosummary::
   :toctree: generated
   :template: class_nomethods.rst

   LabelVar
   EvalVar
   GroupVar


Most of the time, the main purpose of this attribute is to turn trigger
values into meaningful labels::


    class Mouse(MneExperiment):

        variables = {
            'stimulus': LabelVar('trigger', {(162, 163): 'target', (166, 167): 'prime'}),
            'prediction': LabelVar('trigger', {162: 'expected', 163: 'unexpected'}),
        }

This defines a variable called "stimulus", and on this variable all events
that have triggers 162 and 163 have the value ``"target"``, and events with
trigger 166 and 167 have the value ``"prime"``.
Unmentioned trigger values are assigned the empty string (``''``).


Epochs
------

.. py:attribute:: MneExperiment.epochs

Epochs are specified as a ``{name: epoch_definition}`` dictionary. Names are
:class:`str`, and ``epoch_definition`` are instances of the classes
described below:

.. autosummary::
   :toctree: generated
   :template: class_nomethods.rst

   PrimaryEpoch
   SecondaryEpoch
   SuperEpoch


Examples::

    epochs = {
        # some primary epochs:
        'picture': PrimaryEpoch('words', "stimulus == 'picture'"),
        'word': PrimaryEpoch('words', "stimulus == 'word'"),
        # use the picture baseline for the sensor covariance estimate
        'cov': SecondaryEpoch('picture', tmax=0),
        # another secondary epoch:
        'animal_words': SecondaryEpoch('noun', sel="word_type == 'animal'"),
        # a superset-epoch:
        'all_stimuli': SuperEpoch(('picture', 'word')),
    }


Tests
-----

.. py:attribute:: MneExperiment.tests

Statistical tests are defined as ``{name: test_definition}`` dictionary. Test-
definitions are defined from the following:

.. autosummary::
   :toctree: generated
   :template: class_nomethods.rst

   TTestOneSample
   TTestRel
   TTestInd
   ANOVA
   TContrastRel
   TwoStageTest


Example::

    tests = {
        'my_anova': ANOVA('noise * word_type * subject'),
        'my_ttest': TTestRel('noise', 'a_lot_of_noise', 'no_noise'),
    }


Subject groups
--------------

.. py:attribute:: MneExperiment.groups

A subject group called ``'all'`` containing all subjects is always implicitly
defined. Additional subject groups can be defined in
:attr:`MneExperiment.groups` in a dictionary with ``{name: group_definition}``
entries. The simplest group definition is a tuple
of subject names, e.g. ``("R0026", "R0042", "R0066")``. In addition, a
group_definition can be a dictionary with the following entries:

base : :class:`str`
    The name of the group to base the new group on.
exclude : :class:`tuple` of :class:`str`
    A list of subjects to exclude (e.g., ``("R0026", "R0042", "R0066")``)

Examples::

    groups = {
        'some': ("R0026", "R0042", "R0066"),
        'others': {'base': 'all', 'exclude': ("R0666",)},
        # some, buth without R0042:
        'some_less': {'base': 'some', 'exclude': ("R0042",)}
    }


Parcellations (:attr:`parcs`)
-----------------------------

.. py:attribute:: MneExperiment.parcs

The parcellation determines how the brain surface is divided into regions.
A number of standard parcellations are automatically defined (see
:ref:`analysis-params-parc` below). Additional parcellations can be defined in
the :attr:`MneExperiment.parcs` dictionary with ``{name: parc_definition}``
entries. There are a couple of different ways in which parcellations can be
defined, described below.

Each ``parc_definition`` can have a ``"views"`` entry to set the views shown in
anatomical plots, e.g. ``{"views": ("medial", "lateral")}``.


Recombinations
^^^^^^^^^^^^^^

Recombinations of existing parcellations can be defined as dictionaries
include the following entries:

kind : ``'combination'``
    Has to be 'combination'.
base : :class:`str`
    The name of the parcellation that provides the input labels.
labels : :class:`dict` {:class:`str`: :class:`str`}
    New labels to create in ``{name: expression}`` format. All label names
    should be composed of alphanumeric characters (plus underline) and should
    not contain the -hemi tags. In order to create a given label only on one
    hemisphere, add the -hemi tag in the name (not in the expression, e.g.,
    ``{'occipitotemporal-lh': "occipital + temporal"}``).

Examples (these are pre-defined parcellations)::

    parcs = {'lobes-op': {'kind': 'combination',
                          'base': 'lobes',
                          'labels': {'occipitoparietal': "occipital + parietal"}},
             'lobes-ot': {'kind': 'combination',
                          'base': 'lobes',
                          'labels': {'occipitotemporal': "occipital + temporal"}}}


An example using a split label::

    parcs = {
        'medial': {
            'kind': 'combination',
            'base': 'aparc',
            'labels': {
                'medialparietal': 'precuneus + posteriorcingulate',
                'medialfrontal': 'medialorbitofrontal + '
                                 'rostralanteriorcingulate + '
                                 'split(superiorfrontal, 3)[2]',
            },
            'views': 'medial',
        },
    }


.. _MneExperiment.parc-seeded:

MNI coordinates
^^^^^^^^^^^^^^^

Labels can be constructed around known MNI coordinates using the foillowing
entries:

kind : 'seeded'
    Has to be 'seeded'.
seeds : :class:`dict`
    {name: seed(s)} dictionary, where names are strings, including -hemi tags
    (e.g., ``"mylabel-lh"``) and seed(s) are array-like, specifying one or more
    seed coordinate (shape ``(3,)`` or ``(n_seeds, 3)``).
mask : :class:`str`
    Name of a parcellation to use as mask (i.e., anything that is "unknown" in
    that parcellation is excluded from the new parcellation. Use
    ``{'mask': 'lobes'}`` to exclude the subcortical areas around the
    diencephalon.

For each seed entry, the source space vertex closest to the given MNI coordinate
will be used as actual seed, and a label will be created including all points
with a surface distance smaller than a given extent from the seed
vertex/vertices. The extent is determined when setting the parc as analysis
parameter as in ``e.set(parc="myparc-25")``, which specifies a radius of 25 mm.

Example::

     parcs = {'stg': {'kind': 'seeded',
                      'mask': 'lobes',
                      'seeds': {'anteriorstg-lh': ((-54, 10, -8), (-47, 14, -28)),
                                'middlestg-lh': (-66, -24, 8),
                                'posteriorstg-lh': (-54, -57, 16)}}}


Individual coordinates
^^^^^^^^^^^^^^^^^^^^^^

Labels can also be constructured from subjects-specific seeds. They work
like :ref:`MneExperiment.parc-seeded` parcellations, except that seeds are
provided for each subject.

Example::

     parcs = {
         'stg': {
             'kind': 'individual seeded',
             'mask': 'lobes',
             'seeds': {
                 'anteriorstg-lh': {
                     'R0001': (-54, 10, -8),
                     'R0002': (-47, 14, -28),
                 },
                 'middlestg-lh': {
                     'R0001': (-66, -24, 8),
                     'R0002': (-60, -26, 9),
                 }
             }
         }
     }


Externally created parcellations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For parcellations that are user-created, the following two definitions can be
used to determine how they are handled:

"subject_parc"
    Parcellations that are created outside Eelbrain for each subject. These
    parcellations are automatically generated only for scaled brains, for
    subjects' MRIs the user is responsible for creating the respective
    annot-files.
"fsaverage_parc"
    Parcellations that are defined for the fsaverage brain and should be morphed
    to every other subject's brain. These parcellations are automatically
    morphed to individual subjects' MRIs.

Examples (pre-defined parcellations)::

    parcs = {'aparc': 'subject_parc',
             'PALS_B12_Brodmann': 'fsaverage_parc'}


Visualization defaults
----------------------

.. py:attribute:: MneExperiment.brain_plot_defaults

The :attr:`MneExperiment.brain_plot_defaults` dictionary can contain options
that changes defaults for brain plots (for reports and movies). The following
options are available:

surf : 'inflated' | 'pial' | 'smoothwm' | 'sphere' | 'white'
    Freesurfer surface to use as brain geometry.
views : :class:`str` | iterator of :class:`str`
    View or views to show in the figure. Can also be set for each parcellation,
    see :attr:`MneExperiment.parc`.
foreground : mayavi color
    Figure foreground color (i.e., the text color).
background : mayavi color
    Figure background color.
smoothing_steps : ``None`` | :class:`int`
    Number of smoothing steps to display data.


Analysis parameters
===================

These are parameters that can be set after an :class:`MneExperiment` has been
initialized to affect the analysis, for example::

    >>> my_experiment = MneExperiment()
    >>> my_experiment.set(raw='1-40', cov='noreg')

sets up ``my_experiment`` to use raw files filtered with a 1-40 Hz band-pass
filter, and to use sensor covariance matrices without regularization.

.. contents:: Contents
   :local:


.. _MneExperiment-raw-parameter:

``raw``
-------

Which raw FIFF files to use. Can be customized (see :attr:`MneExperiment.raw`).
The default values are:

``'raw'``
    The unfiltered files (as they were added to the data).
``'0-40'`` (default)
    Low-pass filtered under 40 Hz.
``'0.1-40'``
    Band-pass filtered between 0.1 and 40 Hz.
``'1-40'``
    Band-pass filtered between 1 and 40 Hz.


``group``
---------

Any group defined in :attr:`MneExperiment.groups`. Will restrict the analysis
to that group of subjects.


``epoch``
---------

Any epoch defined in :attr:`MneExperiment.epochs`. Specify the epoch on which
the analysis should be conducted.


``rej`` (trial rejection)
-------------------------

Trial rejection can be turned off ``e.set(rej='')``, meaning that no trials are
rejected, and back on, meaning that the corresponding rejection files are used
``e.set(rej='man')``.


``equalize_evoked_count``
-------------------------

By default, the analysis uses all epoch marked as good during rejection. Set
equalize_evoked_count='eq' to discard trials to make sure the same number of
epochs goes into each cell of the model.

'' (default)
    Use all epochs.
'eq'
    Make sure the same number of epochs is used in each cell by discarding
    epochs.


``cov``
-------

The method for correcting the sensor covariance.

'noreg'
    Use raw covariance as estimated from the data (do not regularize).
'bestreg' (default)
    Find the regularization parameter that leads to optimal whitening of the
    baseline.
'reg'
    Use the default regularization parameter (0.1).
'auto'
    Use automatic selection of the optimal regularization method.


``inv``
-------

To set the inverse solution use :meth:`MneExperiment.set_inv`.


.. _analysis-params-parc:

``parc``/``mask`` (parcellations)
---------------------------------

The parcellation determines how the brain surface is divided into regions.
Parcellation are mainly used in tests and report generation:

 - ``parc`` or ``mask`` arguments for :meth:`MneExperiment.make_report`
 - ``parc`` argument to :meth:`MneExperiment.make_report_roi`

When source estimates are loaded, the parcellation can also be used to index
regions in the source estiomates. Predefined parcellations:

Freesurfer Parcellations
    ``aparc.a2005s``, ``aparc.a2009s``, ``aparc``, ``aparc.DKTatlas``,
    ``PALS_B12_Brodmann``, ``PALS_B12_Lobes``, ``PALS_B12_OrbitoFrontal``,
    ``PALS_B12_Visuotopic``.
``lobes``
    Modified version of ``PALS_B12_Lobes`` in which the limbic lobe is merged
    into the other 4 lobes.
``lobes-op``
    One large region encompassing occipital and parietal lobe in each
    hemisphere.
``lobes-ot``
    One large region encompassing occipital and temporal lobe in each
    hemisphere.


.. _analysis-params-connectivity:

``connectivity``
----------------

Possible values: ``''``, ``'link-midline'``

Connectivity refers to the edges connecting data channels (sensors for sensor
space data and sources for source space data). These edges are used to find
clusters in cluster-based permutation tests. For source spaces, the default is
to use FreeSurfer surfaces in which the two hemispheres are unconnected. By
setting ``connectivity='link-midline'``, this default connectivity can be
modified so that the midline gyri of the two hemispheres get linked at sources
that are at most 15 mm apart. This parameter currently does not affect sensor
space connectivity.


.. _analysis-params-select_clusters:

``select_clusters`` (cluster selection criteria)
------------------------------------------------

In thresholded cluster test, clusters are initially filtered with a minimum
size criterion. This can be changed with the ``select_clusters`` analysis
parameter with the following options:

================ ======== =========== ===========
name             min time min sources min sensors
================ ======== =========== ===========
``"all"``
``"10ms"``       10 ms    10          4
``""`` (default) 25 ms    10          4
``"large"``      25 ms    20          8
================ ======== =========== ===========

To change the cluster selection criterion use for example::

    >>> e.set(select_clusters='all')
