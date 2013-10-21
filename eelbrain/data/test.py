"""Some basic statistical tests.

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

from .stats.glm import anova
from .stats.test import (pairwise, ttest, oneway, correlations,
                         bootstrap_pairwise, lilliefors)
