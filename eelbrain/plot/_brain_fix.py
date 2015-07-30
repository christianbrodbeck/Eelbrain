# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Fix up surfer.Brain"""
from surfer import Brain as SurferBrain
from ._brain_mixin import BrainMixin


class Brain(BrainMixin, SurferBrain):

    def __init__(self, unit, *args, **kwargs):
        BrainMixin.__init__(self, unit)
        SurferBrain.__init__(self, *args, **kwargs)
