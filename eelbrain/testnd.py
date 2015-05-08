"""Statistical tests for multidimensional data in :class:`NDVar` objects"""

__test__ = False

from ._stats.testnd import (t_contrast_rel, corr, ttest_1samp, ttest_ind,
                            ttest_rel, anova)
