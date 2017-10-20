# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""
Sample MneExperiment. This experiment can be used with a sample dataset that
treats different parts of the recording from the MNE sample dataset as different
subjects. To produce the data directory for this experiment use (make sure
that the directory you specify exists)::

    >>> from eelbrain import datasets
    >>> datasets.setup_samples_experiment('~/Data')

Then you can use::

    >>> from sample_experiment import SampleExperiment
    >>> e = SampleExperiment("~/Data/SampleExperiment")

"""
from eelbrain import MneExperiment


class SampleExperiment(MneExperiment):

    owner = "me@nyu.edu"

    path_version = 1

    meg_system = 'neuromag306mag'

    sessions = 'sample'

    defaults = {
        'epoch': 'target',
        'select_clusters': 'all',
    }

    variables = {
        'event': {(1, 2, 3, 4): 'target', 5: 'smiley', 32: 'button'},
        'side': {(1, 3): 'left', (2, 4): 'right'},
        'modality': {(1, 2): 'auditory', (3, 4): 'visual'}
    }

    epochs = {
        # all target stimuli:
        'target': {'sel': "event == 'target'", 'tmax': 0.3},
        # only auditory stimulation
        'auditory': {'base': 'target', 'sel': "modality == 'auditory"}
    }

    tests = {
        # T-test to compare left-sided vs right-sided stimulation
        'left=right': {'kind': 'ttest_rel', 'model': 'side',
                       'c1': 'left', 'c0': 'right'},
        # One-tailed test for auditory > visual stimulation
        'a>v': {'kind': 'ttest_rel', 'model': 'modality',
                'c1': 'auditory', 'c0': 'visual', 'tail': 1},
    }
