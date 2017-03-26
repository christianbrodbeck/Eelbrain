# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import warnings

import mne
from numpy import newaxis

from ._version import MNE_14_OR_GREATER


if MNE_14_OR_GREATER:
    def cwt_morlet(data, sfreq, frequencies, n_cycles=7.0, zero_mean=False,
                   use_fft=True, decim=1, n_jobs=1):
        return mne.time_frequency.tfr_array_morlet(
            data[newaxis], sfreq, frequencies, n_cycles, zero_mean, use_fft,
            decim, n_jobs=n_jobs)[0]
else:
    def cwt_morlet(data, sfreq, frequencies, n_cycles=7.0, zero_mean=False,
                   use_fft=True, decim=1, n_jobs=1):
        with warnings.catch_warnings():  # divide by 0
            warnings.filterwarnings('ignore', category=DeprecationWarning,
                                    module='mne')
            return mne.time_frequency.tfr.cwt_morlet(
                data, sfreq, frequencies, use_fft, n_cycles, zero_mean, decim)
