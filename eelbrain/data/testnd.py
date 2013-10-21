'''Statistical tests for multidimensional data in :class:`NDVar` objects.

.. autosummary::
   :toctree: generated

   ttest_1samp
   ttest_rel
   ttest_ind
   anova
   corr
   clean_time_axis

In general, tests are defined as classes that provide results as attributes
and methods::

    >>> res = testnd.ttest_rel(Y, X, 'test', 'control')
    >>> res.p  # an NDVar object with an uncorrected p-value for each sample

Test result objects can be directly submitted to appropriate plotting
functions. To plot only part of the results, specific attributes can be
submitted (for a description of the attributes see the relevant class
documentation)::

    >>> plot.UTS(res)  # plots values in both conditions as well as
    ... difference values with p-value thresholds
    >>> plot.UTS(res.p)  # plots only p-values


Permutation Cluster Tests
=========================

By default the tests in this module produce maps of statistical parameters
uncorrected for multiple comparison. Most tests can also form clusters and
evaluate their significance using permutation of the data [1]_.
Cluster testing is enabled by providing a ``samples`` parameter defining the
number of permutations to perform.

.. warning:: Spatiotemporal permutation cluster test can take a long time to
    evaluate. It might be a good idea to estimate the time they will take using
    a very small value for ``samples`` first.


.. [1] Maris, E., & Oostenveld, R. (2007). Nonparametric
    statistical testing of EEG- and MEG-data. Journal of Neuroscience Methods,
    164(1), 177-190. doi:10.1016/j.jneumeth.2007.03.024

'''

__test__ = False

from .stats.testnd import *
