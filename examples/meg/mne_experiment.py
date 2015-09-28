from eelbrain import *


class WordExperiment(MneExperiment):

    path_version = 1  # path scheme for starting a new experiment

    defaults = {'experiment': 'words'}

    variables = {'stimulus': {10: 'fixation', (1, 2, 3, 4): 'word'},
                 'word_type': {(1, 2): 'noun', (3, 4): 'verb', 10: 'none'},
                 'frequency': {(1, 3): 'low', (2, 4): 'high', 10: 'none'}}

    epochs = {'target': {'sel': "stimulus == 'word'"},
              'cov': {'base': 'target', 'tmax': 0}}

    tests = {'noun>verb': {'kind': 'ttest_rel', 'model': 'word_type',
                           'c1': 'noun', 'c0': 'verb', 'tail': 1},
             'word_type*frequency': {'kind': 'anova',
                                     'model': 'word_type % frequency',
                                     'x': 'word_type * frequency * subject'}}
