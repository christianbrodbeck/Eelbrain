'''Statistical tests for multidimensional data in :class:`NDVar` objects.

.. autosummary::
   :toctree: generated

   ttest_1samp
   ttest_rel
   ttest_ind
   t_contrast_rel
   anova
   corr

In general, tests are defined as classes that provide results as attributes
and methods::

    >>> res = testnd.ttest_rel(Y, X, 'test', 'control')
    >>> res.t  # an NDVar object with a t-value for each sample

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
number of permutations to perform. The threshold for forming clusters is
specified as the `pmin` parameter. The default (`pmin=None`) is threshold-free
cluster enhancement [2]_.

.. warning:: Spatiotemporal permutation cluster tests can take a long time to
    evaluate. It might be a good idea to estimate the time they will take using
    a very small value for ``samples`` first.


.. [1] Maris, E., & Oostenveld, R. (2007). Nonparametric
    statistical testing of EEG- and MEG-data. Journal of Neuroscience Methods,
    164(1), 177-190. doi:10.1016/j.jneumeth.2007.03.024
.. [2] Smith, S. M., and Nichols, T. E. (2009). Threshold-Free Cluster
    Enhancement: Addressing Problems of Smoothing, Threshold Dependence and
    Localisation in Cluster Inference. NeuroImage, 44(1), 83-98.
    doi:10.1016/j.neuroimage.2008.03.061

'''

__test__ = False

from ._stats.testnd import (t_contrast_rel, corr, ttest_1samp, ttest_ind,
                            ttest_rel, anova)
