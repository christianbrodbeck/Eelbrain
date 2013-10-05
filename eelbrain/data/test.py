__test__ = False

from .stats.glm import anova, ancova
from .stats.test import (pairwise, ttest, oneway, correlations,
                         bootstrap_pairwise, lilliefors)
