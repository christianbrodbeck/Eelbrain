Changes
=======

.. currentmodule:: eelbrain


New in 0.17
-----------

* :attr:`MneExperiment.brain_plot_defaults` can be used to customize PySurfer
  plots in movies and reports.
* The rejection GUI now allows selecting individual channels for interpolation
  using the 'i' key.
* Parcellations based on combinations of existing labels, as well as
  parcellations based on regions around points specified in MNI coordinates can
  now be defined in :attr:`MneExperiment.parcs`.
* Source space :class:`NDVars` can be index with lists of region names, e.g.,
  ``ndvar.sub(source=['cuneus-lh', 'lingual-lh'])``.
* API: :func:`plot.brain.bin_table` function signature changed slightly (more
  parameters, new ``hemi`` parameter inserted to match other functions' argument
  order).


New in 0.16
-----------

* New function for plotting a legend for annot-files:
  :func:`plot.brain.annot_legend` (automatically used in reports).
* Epoch definitions in :attr:`MneExperiment.epochs` can now include a ``'base'``
  parameter, which will copy the given "base" epoch and modify it with the
  current definition.
* :meth:`MneExperiment.make_mov_ttest` and
  :meth:`MneExperiment.make_mov_ga_dspm` are fixed but require PySurfer 0.6.
* New function: :func:`table.melt_ndvar`.
* API: :mod:`plot.brain` function signatures changed slightly to accommodate
  more layout-related arguments.
* API: use :meth:`Brain.image` instead of :func:`plot.brain.image`.


New in 0.15
-----------

* The Eelbrain package on the PYPI is now compiled with Anaconda. This means
  that the package can now be installed into an Anaconda distribution with
  ``pip``, whereas ``easy_install`` has to be used for the Canopy distribution.
* GUI :func:`gui.select_epochs`: Set marked channels through menu (View > Mark
  Channels)
* Datasets can be saved as tables in RTF format (:meth:`Dataset.save_rtf`).
* API :class:`plot.Timeplot`: the default spread indicator changed to SEM, and
  there is a new argument for `timelabels`.
* API: :func:`test.anova` is now a function with a slightly changed signature.
  The old class has been renamed to :class:`test.ANOVA`.
* API: :func:`test.oneway` was removed. Use :func:`test.anova`.
* API: the default value of the :class:`plot.Timeplot` parameter `bottom`
  changed from `0` to `None` (determined by the data).
* API: :meth:`Factor.relabel` renamed to :meth:`Factor.update_labels`.
* Plotting: New option for the figure legend ``'draggable'`` (drag the legend
  with the mouse pointer).


New in 0.14
-----------

* API: the :class:`plot.Topomap` argument `sensors` changed to `sensorlabels`.
* GUI: The *python/Quit Eelbrain* menu command now closes all windows to ensure
  that unsaved documents are handled properly. In order to yield to the terminal
  without closing windows, use the *Go/Yield to Terminal* command
  (Command-Alt-Q).
* :class:`testnd.t_contrast_rel`:  support for unary operation `abs`.


New in 0.13
-----------

* The :func:`gui.select_epochs` GUI can now also be used to set bad channels.
  :class:`MneExperiment` subclasses will combine bad channel information from
  rejection files with bad channel information from bad channel files. Note
  that while bad channel files set bad channels for a given raw file
  globally, rejection files set bad channels only for the given epoch.
* :class:`Factor` objects can now remember a custom cell order which determines
  the order in tables and plots.
* The :meth:`Var.as_factor` method can now assign all unmentioned codes to a
  default value.
* :class:`MneExperiment`:

  - API: Subclasses should remove the ``subject`` and ``experiment`` parameters
    from :meth:`MneExperiment.label_events`.
  - API: :class:`MneExperiment` can now be imported directly from
    :mod:`eelbrain`.
  - API: The :attr:`MneExperiment._defaults` attribute should be renamed to
    :attr:`MneExperiment.defaults`.
  - A draft for a guide at :ref:`experiment-class-guide`.
  - Cached files are now saved in a separate folder at ``root/eelbrain-cache``.
    The cache can be cleared using :meth:`MneExperiment.clear_cache`. To
    preserve cached test results, move the ``root/test`` folder into the
    ``root/eelbrain-cache`` folder.


New in 0.12
-----------

* API: :class:`Dataset` construction changed, allows setting the number of
  cases in the Dataset.
* API:  :class:`plot.SensorMap2d` was renamed to :class:`plot.SensorMap`.
* :class:`~experiment.MneExperiment`:

  - API: The default number of samples for reports is now 10'000.
  - New epoch parameter ``'n_cases'``:  raise an error if an epoch definition
    does not yield expected number of trials.
  - A custom baseline period for epochs can now be specified as a parameter in
    the epoch definition (e.g., ``'baseline': (-0.2, -0.1)``). When loading
    data, specifying ``baseline=True`` uses the epoch's custom baseline.

* Use ``eelbrain.set_log_level('info')`` to get information about the progress
  of permutation tests.


New in 0.11
-----------

* :class:`~experiment.MneExperiment`:

  - Change in the way the covariance matrix
    is defined:  The epoch for the covariance matrix should be specified in
    ``MneExperiment.epochs['cov']``. The regularization is no longer part of
    :meth:`~experiment.MneExperiment.set_inv`, but is instead set with
    ``MneExperiment.set(cov='reg')`` or ``MneExperiment.set(cov='noreg')``.
  - New option ``cov='bestreg'`` automatically selects the regularization
    parameter for each subejct.

* :meth:`Var.as_factor` allows more efficient labeling when multiple values
  share the same label.
* API: Previously :func:`plot.configure_backend` is now :func:`plot.configure`


New in 0.10
-----------

* Tools for generating colors for categories (see :ref:`ref-plotting`).
* Plots now all largely respect matplotlib rc-parameters (see
  `Customizing Matplotlib <http://matplotlib.org/users/customizing.html>`_).
* Fixed an issue in the :mod:`testnd` module that could affect permutation based
  p-values when multiprocessing was used.


New in 0.9
----------

* :class:`Factor` API change:  The ``rep`` argument was renamed to ``repeat``.
* T-values for regression coefficients through :meth:`NDVar.ols_t`.
* :class:`~experiment.MneExperiment`: subject name patterns and eog_sns are
  now handled automatically.
* :class:`~plot.uts.UTSStat` and :class:`~plot.uv.Barplot` plots can use pooled
  error for variability estimates (on by default for related measures designs,
  can be turned off using the ``pool_error`` argument).

  - API: for consistency, the argument to specify the kind of error to plot
    changed to ``error`` in both plots.


New in 0.8
----------

* A new GUI application controls plots as well as the epoch selection GUI (see
  notes in the reference sections on :ref:`ref-plotting` and :ref:`ref-guis`).
* Randomization/Monte Carlo tests now seed the random state to make results replicable.


New in 0.6
----------

* New recipes for :ref:`recipe-regression`.


New in 0.5
----------

* The :mod:`eelbrain.lab` and :mod:`eelbrain.eellab` modules are deprecated.
  Everything can now me imported from :mod:`eelbrain` directly.


New in 0.4
----------

* Optimized ANOVA evaluation, support for unbalanced fixed effects models.
* `rpy2 <http://rpy.sourceforge.net>`_ interaction: :meth:`Dataset.from_r` to
  create a :class:`Dataset` from an R Data Frame, and :meth:`Dataset.to_r` to
  cerate an R Data Frame from a :class:`Dataset`.


New in 0.3
----------

* Optimized clustering for cluster permutation tests.


New in 0.2
----------

* :class:`gui.SelectEpochs` Epoch rejection GIU has a new "GA" button to plot
  the grand average of all accepted trials
* Cluster permutation tests in :mod:`testnd` use multiple cores; To disable
  multiprocessing set ``eelbrain._stats.testnd.multiprocessing = False``.


New in 0.1.7
------------

* :class:`gui.SelectEpochs` can now be initialized with a single
  :class:`mne.Epochs` instance (data needs to be preloaded).
* Parameters that take :class:`NDVar` objects now also accept
  :class:`mne.Epochs` and :class:`mne.fiff.Evoked` objects.


New in 0.1.5
------------

* :py:class:`plot.topo.TopoButterfly` plot: new keyboard commands (``t``,
  ``left arrow``, ``right arrow``).
