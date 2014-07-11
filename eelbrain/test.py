"""Some basic statistical tests for univariate data.

.. autosummary::
   :toctree: generated

   anova
   oneway
   pairwise
   ttest
   correlations
   lilliefors

"""
__test__ = False

from ._stats.glm import anova
from ._stats.test import (pairwise, ttest, oneway, correlations,
                          bootstrap_pairwise, lilliefors)
