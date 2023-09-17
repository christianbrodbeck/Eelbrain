# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import mne

MNE_EPOCHS = mne.BaseEpochs
MNE_RAW = mne.io.BaseRaw
MNE_EVOKED = mne.Evoked
MNE_LABEL = (mne.Label, mne.label.BiHemiLabel)
MNE_VOLUME_STC = (mne.VolSourceEstimate, mne.VolVectorSourceEstimate)
