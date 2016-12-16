from distutils.version import LooseVersion
import mne


MNE_VERSION = LooseVersion(mne.__version__)
if MNE_VERSION >= LooseVersion('0.14'):
    MNE_EPOCHS = mne.BaseEpochs
    MNE_RAW = mne.io.BaseRaw
else:
    MNE_EPOCHS = mne.epochs._BaseEpochs
    MNE_RAW = mne.io._BaseRaw
MNE_EVOKED = mne.Evoked
MNE_LABEL = (mne.Label, mne.label.BiHemiLabel)
