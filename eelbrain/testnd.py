"""Statistical tests for multidimensional data in :class:`NDVar` objects"""
__test__ = False
from ._utils import _deprecated_alias

from ._stats.testnd import NDTest, MultiEffectNDTest, Correlation, TTestOneSample, TTestIndependent, TTestRelated, TContrastRelated, ANOVA, Vector, VectorDifferenceRelated
from ._stats.spm import LM, LMGroup

# for backwards compatibility
corr = _deprecated_alias('corr', Correlation, '0.34')
ttest_1samp = _deprecated_alias('ttest_1samp', TTestOneSample, '0.34')
ttest_ind = _deprecated_alias('ttest_ind', TTestIndependent, '0.34')
ttest_rel = _deprecated_alias('ttest_rel', TTestRelated, '0.34')
t_contrast_rel = _deprecated_alias('t_contrast_rel', TTestIndependent, '0.34')
anova = _deprecated_alias('anova', ANOVA, '0.34')
