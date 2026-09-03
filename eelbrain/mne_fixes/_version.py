# make mne backwards-compatibility
import packaging.version

import mne


MNE_VERSION = packaging.version.parse(mne.__version__)
V1 = packaging.version.parse('1')
# Head movement compensation (RawMaxwell(head_pos=True)) relies on fixes to find_bad_channels_maxwell and maxwell_filter output handling that are not in MNE 1.12.1
MNE_SUPPORTS_HEAD_POS = MNE_VERSION > packaging.version.parse('1.12.1')

assert MNE_VERSION > V1
