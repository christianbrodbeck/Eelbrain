"""Statistical tests for multidimensional data in :class:`NDVar` objects"""
__test__ = False

from ._stats.testnd import (
    t_contrast_rel, corr, ttest_1samp, ttest_ind, ttest_rel, anova)
from ._stats.spm import LM, LMGroup


def configure(*args, **kwargs):
    raise RuntimeError("This function has been removed. Please use "
                       "eelbrain.configure() instead")
