# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Find ICA components that follow the heartbeat of a reference component."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks


# Window around each heartbeat [s]. ICA often splits the heartbeat into a sharp QRS component
# and slower components tied to the T-wave or the pulse, which lag the R-peak by a few hundred
# milliseconds, so the window extends well past the peak.
TSTART_DEFAULT = -0.3
TSTOP_DEFAULT = 0.6
MIN_INTERVAL_DEFAULT = 0.4  # minimum interval between heartbeats [s] (150 bpm)
PEAK_THRESHOLD_DEFAULT = 2.  # minimum peak prominence in the reference component [SD]


@dataclass
class CardiacResult:
    """ICA sources averaged around the heartbeats of a reference component.

    Parameters
    ----------
    reference
        Index of the reference component.
    n_peaks
        Number of heartbeats detected in the reference component.
    n_beats
        Number of heartbeats used for the average (those for which the whole window is
        inside a data segment).
    intervals
        Intervals between consecutive heartbeats within segments [s].
    time
        Time relative to the heartbeat [s].
    evoked
        Peak-locked average of each component, ``(n_components, n_times)``, in units of the
        component's standard deviation across heartbeats.
    sem
        Standard error of the mean of ``evoked``, in the same units.
    score
        Variance across time of the peak-locked average, relative to the variance expected
        for a component that is unrelated to the heartbeat (~1 for an unrelated component;
        larger for components with heartbeat-locked activity, without bound).
    """
    reference: int
    n_peaks: int
    n_beats: int
    intervals: np.ndarray
    time: np.ndarray
    evoked: np.ndarray
    sem: np.ndarray
    score: np.ndarray


def peak_locked_sources(
        segments: Sequence[np.ndarray],
        reference: int,
        tstep: float,
        tstart: float = TSTART_DEFAULT,
        tstop: float = TSTOP_DEFAULT,
        min_interval: float = MIN_INTERVAL_DEFAULT,
        threshold: float = PEAK_THRESHOLD_DEFAULT,
) -> CardiacResult:
    """Average ICA sources around the heartbeats detected in a reference component.

    A heartbeat is a peak in the reference component. The sign of an ICA component is
    arbitrary, so the reference is flipped if its largest deflection is negative.

    Parameters
    ----------
    segments
        ICA source time courses, one ``(n_components, n_times)`` array per contiguous data
        segment (one per epoch, or one for a continuous recording).
    reference
        Index of the component with a clearly recognizable heartbeat.
    tstep
        Time step of the sources [s].
    tstart
        Start of the window around each heartbeat [s].
    tstop
        End of the window around each heartbeat [s].
    min_interval
        Minimum interval between heartbeats [s].
    threshold
        Minimum peak prominence in the reference component, in standard deviations of the
        reference component.
    """
    if tstart >= tstop:
        raise ValueError(f"{tstart=}, {tstop=}: the window must have positive length")
    pre = int(round(-tstart / tstep))
    post = int(round(tstop / tstep))
    distance = max(1, int(round(min_interval / tstep)))
    reference_x = [segment[reference] for segment in segments]
    mean = np.mean(np.concatenate(reference_x))
    sd = np.std(np.concatenate(reference_x))
    reference_x = [x - mean for x in reference_x]
    if -min(x.min() for x in reference_x) > max(x.max() for x in reference_x):
        reference_x = [-x for x in reference_x]
    prominence = threshold * sd

    n_peaks = 0
    intervals = []
    beats = []  # (n_components, n_window) for each heartbeat
    for segment, x in zip(segments, reference_x):
        peaks, _ = find_peaks(x, distance=distance, prominence=prominence)
        n_peaks += len(peaks)
        intervals.append(np.diff(peaks) * tstep)
        for peak in peaks:
            if peak >= pre and peak + post <= len(x):
                beats.append(segment[:, peak - pre: peak + post])
    if not n_peaks:
        raise ValueError(f"No peaks exceeding {threshold:g} SD found in component {reference}")
    elif not beats:
        raise ValueError(f"Found {n_peaks} peaks in component {reference}, but none with the whole {tstart:g} - {tstop:g} s window inside a data segment; shorten the window")
    beats = np.array(beats)
    beats -= beats.mean(2, keepdims=True)
    n_beats = len(beats)
    beat_var = beats.var(0, ddof=1) if n_beats > 1 else np.ones(beats.shape[1:])
    # normalize each component by its pooled standard deviation across heartbeats
    beat_sd = np.sqrt(beat_var.mean(1, keepdims=True))
    beat_sd[beat_sd == 0] = 1
    evoked = beats.mean(0) / beat_sd
    sem = np.sqrt(beat_var / n_beats) / beat_sd
    # for unrelated activity, the variance of the mean across n beats is ~1 / n_beats
    score = evoked.var(1) * n_beats
    time = np.arange(-pre, post) * tstep
    return CardiacResult(reference, n_peaks, n_beats, np.concatenate(intervals), time, evoked, sem, score)
