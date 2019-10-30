"""Statistical tests for multidimensional data in :class:`NDVar` objects"""
__test__ = False

from ._stats.testnd import NDTest, MultiEffectNDTest, Correlation, TTestOneSample, TTestIndependent, TTestRelated, TContrastRelated, ANOVA, Vector, VectorDifferenceRelated
from ._stats.spm import LM, LMGroup

# for backwards compatibility
from ._stats.testnd import corr, ttest_1samp, ttest_ind, ttest_rel, t_contrast_rel, anova
