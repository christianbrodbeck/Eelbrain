"""Data-specific colormaps"""
from itertools import cycle
import logging
from math import ceil
from numbers import Real

# colormath starts out at 0; needs to be set before init
logger = logging.getLogger('colormath.color_conversions')
if logger.level == 0:  # otherwise it was probably set by user (DEBUG=10)
    logger.setLevel(logging.WARNING)

from colormath.color_objects import LCHabColor, sRGBColor
from colormath.color_conversions import convert_color
from matplotlib.colors import ListedColormap
from matplotlib.cm import LUTSIZE, register_cmap, get_cmap
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb, to_rgba
import numpy as np


class LocatedColormap:
    vmin = None
    vmax = None
    symmetric = False


class LocatedListedColormap(ListedColormap, LocatedColormap):
    pass


class LocatedLinearSegmentedColormap(LinearSegmentedColormap, LocatedColormap):
    pass


def lch_to_rgb(lightness, chroma, hue):
    "Convert Lightness/Chroma/Hue color representation to RGB"
    psych = LCHabColor(lightness, chroma, hue * 360)
    rgb = convert_color(psych, sRGBColor)
    return rgb.clamped_rgb_r, rgb.clamped_rgb_g, rgb.clamped_rgb_b


def _to_rgb(arg, alpha=False):
    if isinstance(arg, (float, int)):
        arg = lch_to_rgb(50, 100, arg)

    if alpha:
        return to_rgba(arg)
    else:
        return to_rgb(arg)


def make_seq_cmap(seq, val, name):
    """Colormap from sequence of RGB values

    Parameters
    ----------
    seq : iterator
        Each entry is either an RGB tuple (if pre- and post color are
        identical) or a tuple with two RGB tuples (separate pre- and post-
        colors).
    val : iterator
        For each entry in ``seq``, the coordinate on the colormap.
    name : str
        Colormap name.
    """
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
    cdict = {'red': red, 'green': green, 'blue': blue}
    return LinearSegmentedColormap(name, cdict)


def twoway_cmap(n1, hue_start=0.1, hue_shift=0.5, name=None, hues=None):
    """Create colormap for two-way interaction

    Parameters
    ----------
    n1 : int
        Number of levels on the first factor.
    hue_start : 0 <= scalar < 1
        First hue value.
    hue_shift : 0 <= scalar < 1
        Use that part of the hue continuum between categories to shift hue
        within categories.
    name : str
        Name of the colormap.
    hues : list of scalar
        List of hue values corresponding to the levels of the first factor
        (overrides regular hue distribution).
    """
    # within each hue, create values for [-1, -0.5, 0.5, 1]
    # return list of [i, (r, g, b), (r, g, b)]
    if hues is None:
        hues = np.linspace(hue_start, hue_start + 1, n1, False) % 1.
    hue_shift *= (0.5 / n1)

    seqs = []
    for h in hues:
        h_pre = (h - hue_shift) % 1
        h_post = (h + hue_shift) % 1
        seqs.append((lch_to_rgb(0, 100, h_pre),
                     lch_to_rgb(50, 100, h),
                     lch_to_rgb(100, 100, h_post)))

    seq = []
    for i in range(n1):
        seq.append((seqs[i - 1][-1], seqs[i][0]))
        seq.append(seqs[i][1])
        if i == n1 - 1:
            seq.append((seqs[i][2], seqs[0][0]))

    loc = np.linspace(0, 1, n1 * 2 + 1)

    if name is None:
        name = "%i_by_n" % n1

    return make_seq_cmap(seq, loc, name)


def oneway_colors(n, hue_start=0.2, light_range=0.5, light_cycle=None,
                  always_cycle_hue=False, locations=None):
    """Create colors for categories

    Parameters
    ----------
    n : int
        Number of levels.
    hue_start : 0 <= scalar < 1 | sequence of scalar
        First hue value (default 0.2) or list of hue values.
    light_range : scalar | tuple
        Amount of lightness variation. If a positive scalar, the first color is
        lightest; if a negative scalar, the first color is darkest. Tuple with
        two scalar to define a specific range.
    light_cycle : int
        Cycle from light to dark in ``light_cycle`` cells to make nearby colors 
        more distinct (default cycles once).
    always_cycle_hue : bool
        Cycle hue even when cycling lightness. With ``False`` (default), hue
        is constant within a lightness cycle.
    locations : sequence of float
        Locations of the cells on the color-map (all in range [0, 1]; default is
        evenly spaced; example: ``numpy.linspace(0, 1, n) ** 0.5``).
    """
    if locations is not None:
        raise NotImplementedError('locations for non-cmap based colors')

    if light_cycle is None or always_cycle_hue:
        n_hues = n
    else:
        n_hues = int(ceil(n / light_cycle))

    if isinstance(hue_start, Real):
        hue = np.linspace(hue_start, hue_start + 1, n_hues, False) % 1.
        if light_cycle and not always_cycle_hue:
            hue = np.repeat(hue, light_cycle)
    elif len(hue_start) >= n_hues:
        hue = hue_start
    else:
        raise ValueError("If list of hues is provided it needs ot contain at "
                         "least as many hues as there are cells")

    if isinstance(light_range, (list, tuple)):
        start, stop = light_range
    else:
        start = 0.5 + 0.5 * light_range
        stop = 0.5 - 0.5 * light_range

    if light_cycle is None:
        lightness = np.linspace(100 * start, 100 * stop, n)
    else:
        tile = np.linspace(100 * start, 100 * stop, light_cycle)
        lightness = cycle(tile)
        if n % light_cycle:
            hue = hue[:n]

    return [lch_to_rgb(l, 100, h) for l, h in zip(lightness, hue)]


def twoway_colors(n1, n2, hue_start=0.2, hue_shift=0., hues=None, lightness=None):
    """Create colors for two-way interaction

    Parameters
    ----------
    n1, n2 : int
        Number of levels on the first and second factors.
    hue_start : 0 <= scalar < 1
        First hue value.
    hue_shift : 0 <= scalar < 1
        Use that part of the hue continuum between categories to shift hue
        within categories.
    hues : list of scalar
        List of hue values corresponding to the levels of the first factor
        (overrides regular hue distribution).
    lightness : scalar | list of scalar
        If specified as scalar, colors will occupy the range
        ``[lightness, 100-lightness]``. Can also be given as list with one
        value corresponding to each element in the second factor.
    """
    if hues is None:
        hues = np.linspace(hue_start, hue_start + 1, n1, False) % 1.
    else:
        hues = np.asarray(hues)
        if np.any(hues > 1) or np.any(hues < 0):
            raise ValueError(f"hues={hues}: values out of range, need to be in [0, 1]")
        elif len(hues) < n1:
            raise ValueError(f"hues={hues}: need as many hues as levels in the first factor (got {len(hues)}, need {n1})")
    hue_shift *= (1. / 3. / n1)

    if lightness is None:
        lstart = 60. / n2
        ls = np.linspace(lstart, 100 - lstart, n2)
    elif isinstance(lightness, Real):
        ls = np.linspace(lightness, 100 - lightness, n2)
    else:
        if len(lightness) != n2:
            raise ValueError(f"lightness={lightness!r}: need {n2} values")
        ls = lightness

    colors = []
    for hue in hues:
        hs = np.linspace(hue - hue_shift, hue + hue_shift, n2) % 1
        colors.extend(lch_to_rgb(l, 100, h) for l, h in zip(ls, hs))

    return colors


def two_step_colormap(left_max, left, center='transparent', right=None, right_max=None, name='two-step'):
    """Colormap using lightness to extend range

    Parameters
    ----------
    left_max : matplotlib color
        Left end of the colormap.
    left : matplotlib color
        Left middle of the colormap.
    center : matplotlib color | 'transparent'
        Color for the middle value; 'transparent to make the middle transparent
        (default).
    right : matplotlib color
        Right middle of the colormap (if not specified, tyhe colormap ends at
        the location specified by ``center``).
    right_max : matplotlib color
        Right end of the colormap.
    name : str
        Name for the colormap.

    Examples
    --------
    Standard red/blue::

        >>> cmap = plot.two_step_colormap('black', 'red', 'transparent', 'blue', 'black', name='red-blue')
        >>> plot.ColorBar(cmap, 1)

    Or somewhat more adventurous::

        >>> cmap = plot.two_step_colormap('black', (1, 0, 0.3), 'transparent', (0.3, 0, 1), 'black', name='red-blue-2')
    """
    if center == 'transparent':
        center_ = None
        transparent_middle = True
    else:
        center_ = _to_rgb(center, False)
        transparent_middle = False
    left_max_ = _to_rgb(left_max, transparent_middle)
    left_ = _to_rgb(left, transparent_middle)
    is_symmetric = right is not None
    if is_symmetric:
        right_ = _to_rgb(right, transparent_middle)
        right_max_ = _to_rgb(right_max, transparent_middle)
    else:
        right_ = right_max_ = None

    kind = (is_symmetric, transparent_middle)
    if kind == (False, False):
        clist = (
            (0.0, center_),
            (0.5, left_),
            (1.0, left_max_),
        )
    elif kind == (False, True):
        clist = (
            (0.0, (*left_[:3], 0)),
            (0.5, left_),
            (1.0, left_max_),
        )
    elif kind == (True, False):
        clist = (
            (0.0, left_max_),
            (0.25, left_),
            (0.5, center_),
            (0.75, right_),
            (1.0, right_max_),
        )
    elif kind == (True, True):
        clist = (
            (0.0, left_max_),
            (0.25, left_),
            (0.5, (*left_[:3], 0)),
            (0.5, (*right_[:3], 0)),
            (0.75, right_),
            (1.0, right_max_),
        )
    else:
        raise RuntimeError
    cmap = LocatedLinearSegmentedColormap.from_list(name, clist)
    cmap.set_bad('w', alpha=0.)
    cmap.symmetric = is_symmetric
    return cmap


def pigtailed_cmap(cmap, swap_order=('green', 'red', 'blue')):
    # nilearn colormaps with neutral middle
    orig = get_cmap(cmap)._segmentdata
    f = ((LUTSIZE - 1) // 2) / LUTSIZE
    cdict = {
        'green': [(f * (1 - p), *c) for p, *c in reversed(orig[swap_order[0]])],
        'blue': [(f * (1 - p), *c) for p, *c in reversed(orig[swap_order[1]])],
        'red': [(f * (1 - p), *c) for p, *c in reversed(orig[swap_order[2]])],
    }
    start = 1 - f * 0.5
    for color in ('red', 'green', 'blue'):
        cdict[color].extend((start + f * p, *c) for p, *c in orig[color])
    return cdict


def make_cmaps():
    """Create some custom colormaps and register them with matplotlib"""
    # polar:  blue-white-red
    cmap = LinearSegmentedColormap.from_list(
        "polar", (
            (0.0, (0.0, 0.0, 1.0)),
            (0.5, (1.0, 1.0, 1.0)),
            (1.0, (1.0, 0.0, 0.0)),
        ))
    cmap.set_bad('w', alpha=0.)
    register_cmap(cmap=cmap)

    # polar-alpha: middle is transparent instead of white
    cmap = LinearSegmentedColormap.from_list(
        "polar-a", (
            (0.0, (0.0, 0.0, 1.0, 1.0)),
            (0.5, (0.0, 0.0, 1.0, 0.0)),
            (0.5, (1.0, 0.0, 0.0, 0.0)),
            (1.0, (1.0, 0.0, 0.0, 1.0)),
        ))
    cmap.set_bad('w', alpha=0.)
    register_cmap(cmap=cmap)

    # xpolar ("extra-polar"): fade ends into black
    cmap = LinearSegmentedColormap.from_list(
        "xpolar", (
            (0.0, (0.0, 0.0, 0.0)),
            (0.3, (0.0, 0.0, 1.0)),
            (0.5, (1.0, 1.0, 1.0)),
            (0.7, (1.0, 0.0, 0.0)),
            (1.0, (0.0, 0.0, 0.0)),
        ))
    cmap.set_bad('w', alpha=0.)
    register_cmap(cmap=cmap)

    # extra-polar alpha: middle is transparent instead of white
    cmap = LinearSegmentedColormap.from_list(
        "xpolar-a", (
            (0.0, (0.0, 0.0, 0.0, 1.0)),
            (0.3, (0.0, 0.0, 1.0, 1.0)),
            (0.5, (0.0, 0.0, 1.0, 0.0)),
            (0.5, (1.0, 0.0, 0.0, 0.0)),
            (0.7, (1.0, 0.0, 0.0, 1.0)),
            (1.0, (0.0, 0.0, 0.0, 1.0)),
        ))
    cmap.set_bad('w', alpha=0.)
    register_cmap(cmap=cmap)

    # phase
    cmap = LinearSegmentedColormap.from_list(
        "phase", (
            (0.0, (0.0, 0.0, 1.0)),
            (0.5, (1.0, 0.0, 0.0)),
            (1.0, (0.0, 0.0, 1.0)),
        ))
    cmap.set_bad('w', alpha=0.)
    register_cmap(cmap=cmap)

    # sig:  significance map for specific vmax=0.05
    cmap = LinearSegmentedColormap.from_list(
        "sig", (
            (0.0,  (1.0, 1.0, 1.0)),
            (0.02, (1.0, 1.0, 0.0)),
            (0.2,  (1.0, 0.5, 0.0)),
            (1.0,  (1.0, 0.0, 0.0)),
        ))
    cmap.set_over('k', alpha=0.)
    cmap.set_bad('b', alpha=0.)
    register_cmap(cmap=cmap)

    # Nilearn cmaps
    cmap = LinearSegmentedColormap('cold_hot', pigtailed_cmap('hot'), LUTSIZE)
    register_cmap(cmap=cmap)
    cmap = LinearSegmentedColormap('cold_white_hot', pigtailed_cmap('hot_r'), LUTSIZE)
    register_cmap(cmap=cmap)


make_cmaps()

# https://matplotlib.org/tutorials/colors/colormaps.html
mpl_diverging = ('PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic')
symmetric_cmaps = ['polar', 'polar-a', 'xpolar', 'xpolar-a']
symmetric_cmaps.extend(mpl_diverging)
symmetric_cmaps.extend(f'{name}_r' for name in mpl_diverging)
zerobased_cmaps = ('sig',)
# corresponding cmaps with transparency (alpha channel)
ALPHA_CMAPS = {
    'xpolar': 'xpolar-a',
    'RdBu_r': 'xpolar-a',
}
