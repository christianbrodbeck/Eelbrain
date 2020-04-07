"""Some basic statistical tests for univariate data."""
__test__ = False
from ._utils import _deprecated_alias

from ._stats.glm import ANOVA
from ._stats.test import (
    Correlation,
    TTestOneSample, TTestIndependent, TTestRelated, MannWhitneyU, WilcoxonSignedRank,
    pairwise, ttest, correlations,
    bootstrap_pairwise, lilliefors,
    pairwise_correlations,
)

# backwards compatibility
TTest1Samp = _deprecated_alias('TTest1Samp', TTestOneSample, '0.34')
TTestInd = _deprecated_alias('TTestInd', TTestIndependent, '0.34')
TTestRel = _deprecated_alias('TTestRel', TTestRelated, '0.34')
anova = _deprecated_alias('anova', ANOVA, '0.34')
