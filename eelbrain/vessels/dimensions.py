'''
Created on Oct 25, 2012

@author: christian
'''
import re

import numpy as np
import matplotlib.delaunay as delaunay
from scipy.optimize import leastsq

import mne


class DimensionMismatchError(Exception):
    pass


def find_time_point(times, time):
    """
    Returns (index, time) for the closest point to ``time`` in ``times``

    times : array, 1d
        Monotonically increasing time values.
    time : scalar
        Time point for which to find a match.

    """
    if time in times:
        i = np.where(times == time)[0][0]
    else:
        gr = (times > time)
        if np.all(gr):
            if times[1] - times[0] > times[0] - time:
                return 0, times[0]
            else:
                name = repr(times.name) if hasattr(times, 'name') else ''
                raise ValueError("time=%s lies outside array %r" % (time, name))
        elif np.any(gr):
            i_next = np.where(gr)[0][0]
        elif times[-1] - times[-2] > time - times[-1]:
            return len(times) - 1, times[-1]
        else:
            name = repr(times.name) if hasattr(times, 'name') else ''
            raise ValueError("time=%s lies outside array %r" % (time, name))
        t_next = times[i_next]

        sm = times < time
        i_prev = np.where(sm)[0][-1]
        t_prev = times[i_prev]

        if (t_next - time) < (time - t_prev):
            i = i_next
            time = t_next
        else:
            i = i_prev
            time = t_prev
    return i, time



class Dimension(object):
    """
    Base class for dimensions.
    """
    name = 'Dimension'

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def _dimrepr_(self):
        return repr(self.name)

    def dimindex(self, arg):
        raise NotImplementedError



class Sensor(Dimension):
    """
    Class for representing topographic sensor locations.


    Transforms
    ----------

    ``None``:
        Just use horizontal coordinates

    ``'z root'``:
        the radius of each sensor is set to equal the root of the vertical
        distance from the top of the net.

    ``'cone'``:
        derive x/y coordinate from height based on a cone transformation

    ``'lower cone'``:
        only use cone for sensors with z < 0

    """
    name = 'sensor'

    def __init__(self, locs, names=None, groups=None, sysname=None, transform_2d='z root'):
        """
        Arguments
        ---------
        locs : array-like
            list of (x, y, z) coordinates;
            ``x``: anterior - posterior,
            ``y``: left - right,
            ``z``: top - bottom
        names : list of str | None
            sensor names, same order as locs (optional)
        groups : None | dict
            Named sensor groups.
        sysname : None | str
            Name of the sensor system (only used for information purposes).
        transform_2d:
            default transform that is applied when the getLocs2d method is
            called. For options, see the class documentation.


        Examples
        --------

        >>> sensors = [(0,  0,   0),
                       (0, -.25, -.45)]
        >>> sensor_dim = Sensor(sensors, names=["Cz", "Pz"])

        """
        self.sysname = sysname
        self.default_transform_2d = transform_2d

        # 'z root' transformation fails with 32-bit floats
        self.locs = locs = np.array(locs, dtype=np.float64)
        self.n = n = len(locs)

        if names is None:
            names = [str(i) for i in xrange(n)]
        self.names = np.array(names)

        # transformed locations
        self._transformed = {}
        self._triangulations = {}

        # groups
        if groups:
            self.groups = groups
        else:
            self.groups = {}

    def __repr__(self):
        return "<Sensor n=%i, name=%r>" % (self.n, self.sysname)

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        index = self.dimindex(index)
        if np.isscalar(index) or len(index) == 0:
            return None
        elif len(index) > 1:
            locs = self.locs[index]
            names = self.names[index]
            # TODO: groups
            return Sensor(locs, names, transform_2d=self.default_transform_2d,
                          sysname=self.sysname)

    def dimindex(self, arg):
        if isinstance(arg, str):
            idx = self.label2idx(arg)
        elif np.iterable(arg) and isinstance(arg[0], str):
            idx = [self.label2idx(name) for name in arg]
        else:
            idx = arg
        return idx

    @classmethod
    def from_xyz(cls, path=None, **kwargs):
        locs = []
        names = []
        with open(path) as f:
            l1 = f.readline()
            n = int(l1.split()[0])
            for line in f:
                elements = line.split()
                if len(elements) == 4:
                    x, y, z, name = elements
                    x = float(x)
                    y = float(y)
                    z = float(z)
                    locs.append((x, y, z))
                    names.append(name)
        assert len(names) == n
        return cls(locs, names, **kwargs)

    @classmethod
    def from_sfp(cls, path=None, **kwargs):
        locs = []
        names = []
        for line in open(path):
            elements = line.split()
            if len(elements) == 4:
                name, x, y, z = elements
                x = float(x)
                y = float(y)
                z = float(z)
                locs.append((x, y, z))
                names.append(name)
        return cls(locs, names, **kwargs)

    @classmethod
    def from_lout(cls, path=None, transform_2d=None, **kwargs):
        kwargs['transform_2d'] = transform_2d
        locs = []
        names = []
        with open(path) as fileobj:
            fileobj.readline()
            for line in fileobj:
                w, x, y, t, f, name = line.split('\t')
                x = float(x)
                y = float(y)
                locs.append((x, y, 0))
                names.append(name)
        return cls(locs, names, **kwargs)

    def getLocs2d(self, proj='default', extent=1):
        """
        returns a sensor X location array, the first column reflecting the x,
        and the second column containing the y coordinate of each sensor.


        Arguments
        ---------

        ``proj``:
            How to transform 3d coordinates into a 2d map; see class
            documentation for options.
        ``extent``:
            coordinates will be scaled with minimum value 0 and maximum value
            defined by the value of ``extent``.

        """
        if proj == 'default':
            proj = self.default_transform_2d

        if proj is None:
            proj = 'z+'

        index = (proj, extent)
        if index in self._transformed:
            return self._transformed[index]


        if proj in ['cone', 'lower cone', 'z root']:

            # fit the 3d sensor locations to a sphere with center (cx, cy, cz)
            # and radius r

            # error function
            def err(params):
                r, cx, cy, cz = params
                return   (self.locs[:, 0] - cx) ** 2 \
                       + (self.locs[:, 1] - cy) ** 2 \
                       + (self.locs[:, 2] - cz) ** 2 \
                       - r ** 2

            # initial guess of sphere parameters (radius and center)
            params = (1, 0, 0, 0)
            # do fit
            (r, cx, cy, cz), _ = leastsq(err, params)

            # center the sensor locations based on the sphere and scale to
            # radius 1
            sphere_center = np.array((cx, cy, cz))
#            logging.debug("Sensor sphere projection: %r, %r" % (sphere_center, r))
            locs3d = self.locs - sphere_center
            locs3d /= r

            # implement projection
            locs2d = np.copy(locs3d[:, :2])

            if proj == 'cone':
                locs2d[:, [0, 1]] *= (1 - locs3d[:, [2]])
            elif proj == 'lower cone':
                lower_half = locs3d[:, 2] < 0
                if any(lower_half):
                    locs2d[lower_half] *= (1 - locs3d[lower_half][:, [2]])
            elif proj == 'z root':
                z = max(locs3d[:, 2]) - locs3d[:, 2]  # distance form top
                r = np.sqrt(z)  # desired 2d radius
                r_xy = np.sqrt(np.sum(locs3d[:, :2] ** 2, 1))  # current radius in xy
                idx = (r_xy != 0)  # avoid zero division
                F = r[idx] / r_xy[idx]  # stretching factor accounting for current r
                locs2d[idx, :] *= F[:, None]

        else:
            pattern = re.compile('([xyz])([+-])')
            match = pattern.match(proj.lower())
            if match:
                ax = match.group(1)
                sign = match.group(2)
                if ax == 'x':
                    locs2d = np.copy(self.locs[:, 1:])
                    if sign == '-':
                        locs2d[:, 0] = -locs2d[:, 0]
                elif ax == 'y':
                    locs2d = np.copy(self.locs[:, [0, 2]])
                    if sign == '+':
                        locs2d[:, 0] = -locs2d[:, 0]
                elif ax == 'z':
                    locs2d = np.copy(self.locs[:, :2])
                    if sign == '-':
                        locs2d[:, 1] = -locs2d[:, 1]
            else:
                raise ValueError("invalid proj kwarg: %r" % proj)

        # correct extent
        if extent:
            locs2d -= np.min(locs2d, axis=0)  # move to bottom left
            locs2d /= (np.max(locs2d) / extent)  # scale to extent
            locs2d -= np.min(locs2d, axis=0) / 2  # center

        # save for future access
        self._transformed[index] = locs2d
        return locs2d

    def get_tri(self, proj, resolution, frame):
        """
        Returns delaunay triangulation and meshgrid objects
        (for projecting sensor maps to ims)

        Based on matplotlib.mlab.griddata function
        """
        locs = self.getLocs2d(proj)
        tri = delaunay.Triangulation(locs[:, 0], locs[:, 1])

        emin = -frame
        emax = 1 + frame
        x = np.linspace(emin, emax, resolution)
        xi, yi = np.meshgrid(x, x)

        return tri, xi, yi

    def get_im_for_topo(self, Z, proj='default', res=100, frame=.03, interp='linear'):
        """
        Returns an im for an arrray in sensor space X

        Based on matplotlib.mlab.griddata function
        """
        if proj == 'default':
            proj = self.default_transform_2d

        index = (proj, res, frame)

        tri, xi, yi = self._triangulations.setdefault(index, self.get_tri(*index))

        if interp == 'nn':
            interp = tri.nn_interpolator(Z)
            zo = interp(xi, yi)
        elif interp == 'linear':
            interp = tri.linear_interpolator(Z)
            zo = interp[yi.min():yi.max():complex(0, yi.shape[0]),
                        xi.min():xi.max():complex(0, xi.shape[1])]
        else:
            raise ValueError("interp keyword must be one of"
            " 'linear' (for linear interpolation) or 'nn'"
            " (for natural neighbor interpolation). Default is 'nn'.")
        # mask points on grid outside convex hull of input data.
        if np.any(np.isnan(zo)):
            zo = np.ma.masked_where(np.isnan(zo), zo)
        return zo

    def get_ROIs(self, base):
        """
        returns list if list of sensors, grouped according to closest
        spatial proximity to elements of base (=list of sensor ids)"

        """
        locs3d = self.locs
        # print loc3d
        base_locs = locs3d[base]
        ROI_dic = dict((i, [Id]) for i, Id in enumerate(base))
        for i, loc in enumerate(locs3d):
            if i not in base:
                dist = np.sqrt(np.sum((base_locs - loc) ** 2, 1))
                min_i = np.argmin(dist)
                ROI_dic[min_i].append(i)
        out = ROI_dic.values()
        return out

    def get_subnet_ROIs(self, ROIs, loc='first'):
        """
        returns new Sensor instance, combining groups of sensors in the old
        instance into single sensors in the new instance. All sensors for
        each element in ROIs are the basis for one new sensor.

        ! Only implemented for numeric indexes, not for boolean indexes !

        **parameters:**

        ROIs : list of lists of sensor ids
            each ROI defines one sensor in the new net
        loc : str
            'first': use the location of the first sensor of each ROI (default);
            'mean': use the mean location

        """
        names = []
        locs = np.empty((len(ROIs, 3)))
        for i, ROI in enumerate(ROIs):
            i = ROI[0]
            names.append(self.names[i])

            if loc == 'first':
                ROI_loc = self.locs[i]
            elif loc == 'mean':
                ROI_loc = self.locs[ROI].mean(0)
            else:
                raise ValueError("invalid value for loc (%s)" % loc)
            locs[i] = ROI_loc

        return Sensor(locs, names, sysname=self.sysname)

    def label2idx(self, label):
        """
        Returns the index of the sensor with the given label. Raises a
        KeyError if no sensor with that label exists or if several sensors with
        that label exist.

        """
        idxs = np.where(self.names == label)[0]
        if len(idxs) == 0:
            raise KeyError("No sensor named %r" % label)
        elif len(idxs) == 1:
            return idxs[0]
        else:
            raise KeyError("More than one index named %r" % label)


class SourceSpace(Dimension):
    name = 'source'
    """
    Indexing
    --------

    besides numpy indexing, the following indexes are possible:

     - mne Label objects
     - 'lh' or 'rh' to select an entire hemisphere

    """
    def __init__(self, vertno, subject='fsaverage'):
        """
        Parameters
        ----------
        vertno : list of array
            The indices of the dipoles in the different source spaces.
            Each array has shape [n_dipoles] for in each source space]
        subject : str
            The mri-subject (used to load brain).
        """
        self.vertno = vertno
        self.lh_vertno = vertno[0]
        self.rh_vertno = vertno[1]
        self.lh_n = len(self.lh_vertno)
        self.rh_n = len(self.rh_vertno)
        self.subject = subject

    def __repr__(self):
        return "<dim SourceSpace: %i (lh), %i (rh)>" % (self.lh_n, self.rh_n)

    def __len__(self):
        return self.lh_n + self.rh_n

    def __getitem__(self, index):
        vert = np.hstack(self.vertno)
        hemi = np.zeros(len(vert))
        hemi[self.lh_n:] = 1

        vert = vert[index]
        hemi = hemi[index]

        new_vert = (vert[hemi == 0], vert[hemi == 1])
        dim = SourceSpace(new_vert, subject=self.subject)
        return dim

    def dimindex(self, obj):
        if isinstance(obj, (mne.Label, mne.label.BiHemiLabel)):
            return self.label_index(obj)
        elif isinstance(obj, str):
            if obj == 'lh':
                if self.lh_n:
                    return slice(None, self.lh_n)
                else:
                    raise IndexError("lh is empty")
            if obj == 'rh':
                if self.rh_n:
                    return slice(self.lh_n, None)
                else:
                    raise IndexError("rh is empty")
            else:
                raise IndexError('%r' % obj)
        else:
            return obj

    def _hemilabel_index(self, label):
        if label.hemi == 'lh':
            stc_vertices = self.vertno[0]
            base = 0
        else:
            stc_vertices = self.vertno[1]
            base = len(self.vertno[0])

        idx = np.nonzero(map(label.vertices.__contains__, stc_vertices))[0]
        return idx + base

    def label_index(self, label):
        """Returns the index for a label

        Parameters
        ----------
        label : Label | BiHemiLabel
            The label (as created for example by mne.read_label). If the label
            does not match any sources in the SourceEstimate, a ValueError is
            raised.
        """
        if label.hemi == 'both':
            lh_idx = self._hemilabel_index(label.lh)
            rh_idx = self._hemilabel_index(label.rh)
            idx = np.hstack((lh_idx, rh_idx))
        else:
            idx = self._hemilabel_index(label)

        if len(idx) == 0:
            raise ValueError('No vertices match the label in the stc file')

        return idx



class UTS(Dimension):
    """Dimension object for representing uniform time series

    Special Indexing
    ----------------

    (tstart, tstop) : tuple
        Restrict the time to the indicated window (either end-point can be
        None).

    """
    name = 'time'
    def __init__(self, tmin, tstep, nsamples):
        self.nsamples = nsamples = int(nsamples)
        self.times = np.arange(tmin, tmin + tstep * nsamples, tstep)
        self.tmin = tmin
        self.tstep = tstep
        self.tmax = self.times[-1]

    def __repr__(self):
        return "UTS(%s, %s, %s)" % (self.tmin, self.tstep, self.nsamples)

    def _dimrepr_(self):
        tmax = self.times[-1]
        sfreq = 1. / self.tstep
        r = '%r: %.3f - %.3f s, %s Hz' % (self.name, self.tmin, tmax, sfreq)
        return r

    def __len__(self):
        return len(self.times)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.times[index]
        elif isinstance(index, slice):
            if index.start is None:
                start = 0
            else:
                start = index.start

            if index.stop is None:
                stop = len(self)
            else:
                stop = index.stop

            tmin = self.times[start]
            nsteps = stop - start - 1

            if index.step is None:
                tstep = self.tstep
            else:
                tstep = self.tstep * index.step
        else:
            times = self.times[index]
            tmin = times[0]
            nsteps = len(times)
            steps = np.unique(np.diff(times))
            if len(steps) > 1:
                raise NotImplementedError("non-uniform time series")
            tstep = steps[0]

        return UTS(tmin, tstep, nsteps)

    def dimindex(self, arg):
        if np.isscalar(arg):
            i, _ = find_time_point(self.times, arg)
            return i
        if isinstance(arg, tuple) and len(arg) == 2:
            tstart, tstop = arg
            if tstart is None:
                start = None
            else:
                start, _ = find_time_point(self.times, tstart)

            if tstop is None:
                stop = None
            else:
                stop, _ = find_time_point(self.times, tstop)

            s = slice(start, stop)
            return s
        else:
            return arg

    @property
    def x(self):
        return self.times
