# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""
A copy of sample_experiment.py with multiple sessions.
To produce the data directory for this experiment use (make sure
that the directory you specify exists)::

    >>> from eelbrain import datasets
    >>> datasets.setup_samples_experiment('~/Data', n_segments=2, n_sessions=2)

Then you can use::

    >>> from sample_experiment_sessions import SampleExperiment, ROOT
    >>> e = SampleExperiment(ROOT)

"""
from eelbrain import MneExperiment


ROOT = "~/Data/SampleExperiment"


class SampleExperiment(MneExperiment):

    owner = "me@nyu.edu"

    path_version = 1

    meg_system = 'neuromag306mag'

    sessions = ('sample1', 'sample2')

    variables = {
        'event': {(1, 2, 3, 4): 'target', 5: 'smiley', 32: 'button'},
        'side': {(1, 3): 'left', (2, 4): 'right'},
        'modality': {(1, 2): 'auditory', (3, 4): 'visual'}
    }

    epochs = {
        'target1': {'session': 'sample1', 'sel': "event == 'target'"},
        'target2': {'session': 'sample2', 'sel': "event == 'target'"},
        'super': {'sub_epochs': ('target1', 'target2')},
    }
