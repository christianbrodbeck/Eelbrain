"""Some basic statistical tests for univariate data."""
__test__ = False

from ._stats.glm import anova, ANOVA
from ._stats.test import (TTest1Sample, TTestRel, pairwise, ttest, correlations,
                          bootstrap_pairwise, lilliefors)
