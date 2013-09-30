'''
Statistical tests for NDVar objects.

Tests are defined as classes that provide aspects of their results as
attributes and methods::

    >>> res = testnd.ttest(Y, X, 'test', 'control')
    >>> res.p  # an NDVar object with an uncorrected p-value for each sample

Test result objects can be directly submitted to plotting functions. To plot
only part of the results, specific attributes can be submitted (for a
description of the attributes see the relevant class documentation)::

    >>> plot.UTS(res)  # plots values in both conditions as well as
    ... difference values with p-value thresholds
    >>> plot.UTS(res.p)  # plots only p-values

The way this is implemented is that plotting functions test for the presence
of a ``._default_plot_obj`` and a ``.all`` attribute (in that order) which
is expected to provide a default object for plotting. This is implemented in
:py:mod:`plot._base.unpack_epochs_arg`.

'''

__test__ = False

from .stats.testnd import *
