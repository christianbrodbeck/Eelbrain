import mne
from pathlib import Path

from eelbrain import load, Factor, boosting, event_impulse_predictor


data_dir = Path(mne.datasets.sample.data_path())
meg_dir = data_dir / 'MEG' / 'sample'
raw_file = meg_dir / 'sample_audvis_filt-0-40_raw.fif'
raw = mne.io.Raw(raw_file, preload=True)

events = load.mne.events(raw)
trigger = events['trigger']

# use trigger to add various labels to the dataset
events['condition'] = Factor(trigger, labels={1: 'LA', 2: 'RA', 3: 'LV', 4: 'RV', 5: 'smiley', 32: 'button'})
events['side'] = Factor(trigger, labels={1: 'L', 2: 'R', 3: 'L', 4: 'R', 5: 'None', 32: 'None'})
events['modality'] = Factor(trigger, labels={1: 'A', 2: 'A', 3: 'V', 4: 'V', 5: 'None', 32: 'None'})
events['time'] = events['i_start'] / raw.info['sfreq']

y = load.mne.raw_ndvar(raw)
x = event_impulse_predictor(y.time, value="modality == 'A'", data=events)

trf = boosting(y, x, 0, 0.500, test=True, partitions=4, basis=0.050)
