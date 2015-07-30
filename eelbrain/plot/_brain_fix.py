# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Fix up surfer.Brain"""
import surfer
from ._brain_mixin import BrainMixin


class Brain(BrainMixin, surfer.Brain):

    pass
