# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Simulate EEG data"""
import numpy as np
import scipy.signal
import scipy.spatial

from .._data_obj import Dataset, Factor, Var, NDVar, Sensor, UTS
from .._ndvar import filter_data, segment


def _topo(sensor, center, falloff=1):
    i = sensor.names.index(center)
    loc = sensor.locs[i]
    dists = scipy.spatial.distance.cdist([loc], sensor.locs)[0]
    radius = sensor._sphere_fit[1].mean()
    dists /= radius
    topo = 1.0 - dists
    # topo **= falloff
    return NDVar(topo, (sensor,))


def _window(center: float, width: float, time: UTS):
    width_i = int(round(width / time.tstep))
    n_times = len(time)
    center_i = time._array_index(center)
    if center_i > n_times // 2:
        start = None
        stop = n_times
        window_width = 2 * center_i
    else:
        start = -n_times
        stop = None
        window_width = 2 * (n_times - center_i)
    window_data = scipy.signal.windows.gaussian(window_width, width_i)[start: stop]
    return NDVar(window_data, (time,))


def simulate_erp(n_trials=80, seed=0):
    """Simulate event-related EEG data

    Parameters
    ----------
    n_trials : int
        Number of trials (default 100).
    seed : int
        Random seed.

    Examples
    --------
    Compare with kiloword::

        ys = datasets.simulate_erp()['eeg']
        ys -= ys.mean(sensor=['M1', 'M2'])
        import mne
        path = mne.datasets.kiloword.data_path()
        y = load.fiff.epochs_ndvar(path + '/kword_metadata-epo.fif')
        plot.TopoButterfly([y, ys])

    """
    assert n_trials % 2 == 0

    sensor = Sensor.from_montage('standard_alphabetic')
    sensor.set_connectivity(connect_dist=1.66)
    time = UTS(-0.100, 0.005, 140)

    # Generate random values for the independent variable (cloze probability)
    rng = np.random.RandomState(seed)
    cloze_x = np.concatenate([
        rng.uniform(0, 0.3, n_trials // 2),
        rng.uniform(0.8, 1.0, n_trials // 2),
    ])
    rng.shuffle(cloze_x)
    cloze = Var(cloze_x)

    # Generate topography
    n400_topo = -2.0 * _topo(sensor, 'Cz')
    # Generate timing
    n400_timecourse = _window(0.400, 0.034, time)
    # Put all the dimensions together to simulate the EEG signal
    signal = (1 - cloze) * n400_timecourse * n400_topo

    # add early responses:
    # 130 ms
    amp = Var(rng.normal(1.5, 1, n_trials))
    tc = _window(0.130, 0.025, time)
    topo = _topo(sensor, 'O1') + _topo(sensor, 'O2') - 0.5 * _topo(sensor, 'Cz')
    signal += amp * tc * topo
    # 195 ms
    amp = Var(rng.normal(0.8, 1, n_trials))
    tc = _window(0.195, 0.015, time)
    topo = 1.2 * _topo(sensor, 'F3') + _topo(sensor, 'F4')
    signal += amp * tc * topo
    # 270
    amp = Var(rng.normal(1., 1, n_trials))
    tc = _window(0.270, 0.050, time)
    topo = _topo(sensor, 'O1') + _topo(sensor, 'O2')
    signal += amp * tc * topo
    # 280
    amp = Var(rng.normal(-1, 1, n_trials))
    tc = _window(0.280, 0.030, time)
    topo = _topo(sensor, 'Pz')
    signal += amp * tc * topo
    # 600
    amp = Var(rng.normal(0.5, 1, n_trials))
    tc = _window(0.590, 0.100, time)
    topo = -_topo(sensor, 'Fz')
    signal += amp * tc * topo

    # Generate noise as continuous time series
    flat_time = UTS(0, time.tstep, time.nsamples * n_trials)
    noise_shape = (len(sensor), len(flat_time))
    noise_x = rng.normal(0, 4, noise_shape)
    noise = NDVar(noise_x, (sensor, flat_time))
    noise = filter_data(noise, None, 10, h_trans_bandwidth=50)
    noise = noise.smooth('sensor', 30, 'gaussian')

    # segment noise into trials to add it to the signal
    signal += segment(noise, np.arange(.1, n_trials * .7, .7), -0.1, 0.6)

    # Store EEG data in a Dataset with trial information
    ds = Dataset()
    ds['eeg'] = signal
    ds['cloze'] = Var(cloze_x)
    ds['cloze_cat'] = Factor(cloze_x > 0.5, labels={True: 'high', False: 'low'})

    return ds
