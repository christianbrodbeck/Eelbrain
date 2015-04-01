from eelbrain import *


class WordExperiment(MneExperiment):

    path_version = 1  # path scheme for starting a new experiment

    variables = {'stimulus': {10: 'fixation', (1, 2, 3, 4): 'word'},
                 'word_type': {(1, 2): 'noun', (3, 4): 'verb', 10: 'none'},
                 'frequency': {(1, 3): 'low', (2, 4): 'high', 10: 'none'}}

    defaults = {'experiment': 'words'}
