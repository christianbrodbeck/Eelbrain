"""
In addition to matplotlib colormaps, the following names can be used:

"polar"
    white at 0, red for positive and blue for negative values.
"xpolar"
    like cm_polar, but extends the range by fading red and blue into black at
    extremes.
"sig"
    Significance map.
"sigwhite"
    Significance map where nonsignificant values are white instead of black.
"symsig"
    Symmetric bipolar significance map.
"""
import matplotlib as mpl


def make_seq_cmap(seq, val, name):
    red = []
    green = []
    blue = []
    for v, col in zip(val, seq):
        if len(col) == 3:
            r0, g0, b0 = col
            r1, g1, b1 = col
        elif len(col) == 2:
            (r0, g0, b0), (r1, g1, b1) = col
        else:
            raise ValueError('col in seq: %s' % str(col))
        red.append((v, r0, r1))
        green.append((v, g0, g1))
        blue.append((v, b0, b1))
    cdict = {'red': red, 'green': green, 'blue':blue}
    cm = mpl.colors.LinearSegmentedColormap(name, cdict)
    return cm


def make_cmaps():
    """Create some custom colormaps and register them with matplotlib"""
    _cdict = {'red':  [(.0, .0, .0),
                       (.5, 1., 1.),
                       (1., 1., 1.)],
              'green':[(.0, .0, .0),
                       (.5, 1., 1.),
                       (1., .0, .0)],
              'blue': [(.0, 1., 1.),
                       (.5, 1., 1.),
                       (1., .0, .0)]}
    cm_polar = mpl.colors.LinearSegmentedColormap("polar", _cdict)
    cm_polar.set_bad('w', alpha=0.)
    mpl.cm.register_cmap(cmap=cm_polar)

    x = .3
    _cdict = {'red':  [(0, 0., 0.),
                       (0 + x, 0., 0.),
                       (.5, 1., 1.),
                       (1 - x, 1., 1.),
                       (1, 0., 0.)],
              'green':[(0, 0., 0.),
                       (0 + x, 0., 0.),
                       (.5, 1., 1.),
                       (1 - x, 0., 0.),
                       (1., 0., 0.)],
              'blue': [(0, 0., 0.),
                       (0 + x, 1., 1.),
                       (.5, 1., 1.),
                       (1 - x, 0., 0.),
                       (1, .0, .0)]}
    cm_xpolar = mpl.colors.LinearSegmentedColormap("xpolar", _cdict)
    cm_xpolar.set_bad('w', alpha=0.)
    mpl.cm.register_cmap(cmap=cm_xpolar)

    cdict = {'red':   [(0.0, 0., 0.),
                       (0.5, 1., 1.),
                       (1.0, 0., 0.)],
             'green': [(0.0, 0., 0.),
                       (0.5, 0., 0.),
                       (1.0, 0., 0.)],
             'blue':  [(0.0, 1., 1.),
                       (0.5, 0., 0.),
                       (1.0, 1., 1.)]}
    cm_phase = mpl.colors.LinearSegmentedColormap("phase", cdict)
    cm_phase.set_bad('w', alpha=0.)
    mpl.cm.register_cmap(cmap=cm_phase)

    cdict = {'red':[(0.0, 1.0, 1.0),
                    (1.0, 1.0, 0.0)],
         'green':  [(0.0, 1.0, 1.0),
                    (0.1, 1.0, 1.0),
                    (1.0, 0.0, 0.0)],
         'blue':   [(0.0, 1.0, 1.0),
                    (0.1, 0.0, 0.0),
                    (1.0, 0.0, 0.0)]}
    cmap = mpl.colors.LinearSegmentedColormap("sig", cdict)
    cmap.set_over('k', alpha=0.)
    cmap.set_bad('b', alpha=0.)
    mpl.cm.register_cmap(cmap=cmap)

    cdict = {'red':   [(0.00, 0.0, 0.0),  # p=0.05
                       (0.40, 0.0, 0.0),  # p=0.01
                       (0.49, 1.0, 1.0),  # p=0.001
                       (0.51, 1.0, 1.0),
                       (0.60, 1.0, 1.0),
                       (1.00, 0.2, 0.2)],
             'green': [(0.00, 0.0, 0.0),
                       (0.40, 0.0, 0.0),
                       (0.49, 0.0, 0.0),
                       (0.50, 1.0, 1.0),
                       (0.51, 1.0, 1.0),
                       (0.60, 0.0, 0.0),
                       (1.00, 0.0, 0.0)],
             'blue':  [(0.00, 0.2, 0.2),
                       (0.40, 1.0, 1.0),
                       (0.49, 1.0, 1.0),
                       (0.50, 1.0, 1.0),
                       (0.51, 0.0, 0.0),
                       (0.60, 0.0, 0.0),
                       (1.00, 0.0, 0.0)]}
    cmap = mpl.colors.LinearSegmentedColormap("symsig", cdict, N=512)
    cmap.set_bad('b', alpha=0.)
    cmap.set_over('k', alpha=0.)
    cmap.set_under('k', alpha=0.)
    mpl.cm.register_cmap(cmap=cmap)

    # interaction cell coloring cmaps ---
    # yellow / blue
    seq = [ (1, 1, .2),
           ((.6, .6, .6), (.2, .2, 1)),
                          (.3, .7, 1),
                          (.6, .6, .6)]
    val = (0, .25, .5, .75, 1)
    cmap = make_seq_cmap(seq, val, "2group-yb")
    mpl.cm.register_cmap(cmap=cmap)

    # orange / blue
    seq = [ (1, .5, .1),
            (1, .6, .6),
           ((.6, .6 , .6), (.2, .2, 1)),
                           (.3, .7, 1),
                           (.6, .6, .6)]
    val = (0, .25, .5, .75, 1)
    cmap = make_seq_cmap(seq, val, "2group-ob")
    mpl.cm.register_cmap(cmap=cmap)



make_cmaps()

symmetric_cmaps = ('polar', 'xpolar', 'symsig',
                   'BrBG', 'BrBG_r', 'PRGn', 'PRGn_r', 'PuOr', 'PuOr_',
                   'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'seismic', 'seismic_r')
zerobased_cmaps = ('sig',)


def set_info_cs(info, cs={'cmap': 'jet'}, copy=True):
    """Update the plotting arguments in info to reflect a new colorspace

    Parameters
    ----------
    info : dict
        The previous info dictionary.
    cs : dict
        The new colorspace info dictionary.
    copy : bool
        Make a copy of the dictionary before modifying it.

    Returns
    -------
    info : dict
        The updated dictionary.
    """
    if copy:
        info = info.copy()
    for key in ('meas', 'unit', 'cmap', 'vmin', 'vmax', 'contours'):
        if key in info and key not in cs:
            info.pop(key)
    info.update(cs)
    return info


def default_info(meas, **kwargs):
    "Default colorspace info"
    kwargs['meas'] = meas
    return kwargs


def cluster_pmap_info():
    contours = {0.05: (0., 0., 0.)}
    info = {'meas': 'p', 'contours': contours, 'cmap': 'sig', 'vmax': 0.05}
    return info


def sig_info(p=.05, contours={.01: '.5', .001: '0'}):
    "Info dict for significance map"
    info = {'meas': 'p', 'cmap': 'sig', 'vmax': p, 'contours': contours}
    return info


def stat_info(meas, c0=None, c1=None, c2=None, tail=0, **kwargs):
    if 'contours' not in kwargs:
        contours = kwargs['contours'] = {}
        if c0 is not None:
            if tail >= 0:
                contours[ c0] = (1.0, 0.5, 0.1)
            if tail <= 0:
                contours[-c0] = (0.5, 0.1, 1.0)
        if c1 is not None:
            if tail >= 0:
                contours[ c1] = (1.0, 0.9, 0.2)
            if tail <= 0:
                contours[-c1] = (0.9, 0.2, 1.0)
        if c2 is not None:
            if tail >= 0:
                contours[ c2] = (1.0, 1.0, 0.8)
            if tail <= 0:
                contours[-c2] = (1.0, 0.8, 1.0)

    if meas == 'r':
        info = {'meas': meas, 'cmap': 'RdBu_r', 'vmax': 1}
    elif meas == 't':
        info = {'meas': meas, 'cmap': 'RdBu_r'}
    elif meas == 'f':
        info = {'meas': meas, 'cmap': 'BuPu_r', 'vmin': 0}
    else:
        info = default_info(meas)
    info.update(kwargs)
    return info


def sym_sig_info(meas='p'):
    "Info dict for bipolar, symmetric significance map"
    p = 0.05
    info = {'meas': meas, 'cmap': 'symsig', 'vmax': p}
    return info


_unit_fmt = {1: "%s",
             1e-3: "m%s",
             1e-6: r"$\mu$%s",
             1e-9: "n%s",
             1e-12: "p%s",
             1e-15: "f%s"}

def eeg_info(vmax=None, mult=1, unit='V', meas="V"):
    unit = _unit_fmt[1 / mult] % unit
    if vmax is None:
        vmax = 1.5e-6 * mult
    return dict(cmap='xpolar', vmax=vmax, meas=meas, unit=unit)

def meg_info(vmax=None, mult=1, unit='T', meas="B"):
    unit = _unit_fmt[1 / mult] % unit
    if vmax is None:
        vmax = 2e-12 * mult
    return dict(cmap='xpolar', vmax=vmax, meas=meas, unit=unit)
