# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import mne

from ._version import MNE_14_OR_GREATER


if MNE_14_OR_GREATER:
    MNE_EPOCHS = mne.BaseEpochs
    MNE_RAW = mne.io.BaseRaw
else:
    MNE_EPOCHS = mne.epochs._BaseEpochs
    MNE_RAW = mne.io._BaseRaw
MNE_EVOKED = mne.Evoked
MNE_LABEL = (mne.Label, mne.label.BiHemiLabel)
