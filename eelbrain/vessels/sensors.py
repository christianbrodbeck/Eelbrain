"""
Sensor Nets
===========

Defines the sensor_net class and functions to import sensor nets from 
different file types. 

"""


import os

import numpy as np
import matplotlib.pyplot as P
import matplotlib.delaunay as delaunay
#from matplotlib import mlab
from scipy.optimize import leastsq



# find path for sfp files
sfp_path = __file__
sfp_path = os.path.dirname(sfp_path)
sfp_path = os.path.join(sfp_path, "sfp")
sfp_path = os.path.abspath(sfp_path)



class sensor_net(object):
    """
    Transforms
    ----------
    
    ``None``:
        Just use horizontal coordinates
    
    ``'ideal'``:
        use xxx
        
    ``'cone'``:
        derive x/y coordinate from height based on a cone transformation 
    
    ``'lower cone'``:
        only use cone for sensors with z < 0

    """
    def __init__(self, sensors, name=None, groups=None, mirror_map=None,
                 transform_2d='ideal'):
        """
        Arguments
        ---------
        
        sensors: 
            expects list of (x, y, z [, name]) tuples (name is optional)
            ``x``: anterior - posterior, 
            ``y``: left - right, 
            ``z``: top - bottom
        
        transform_2d:
            default transform that is applied when the getLocs2d method is 
            called. For options, see the class documentation.
        
        
        Example::

            >>> sensors = [(0,  0,   0, "Cz"),
                           (0, -.25, -.45, "Pz")]
            >>> net = snesor_net(sensors, ...
        
        
        """
        self.name = 'sensor' # so that dimension is recognized as such
        self.net_name = name
        self.default_transform_2d = transform_2d
        
        self.n = n = len(sensors)
        self.locs3d = locs3d = np.empty((n,3))
        self.names = names = []
        for i, sensor in enumerate(sensors):
            if len(sensor) == 3:
                x, y, z = sensor
                label = str(i)
            elif len(sensor) == 4:
                x, y, z, label = sensor
            
            locs3d[i] = x, y, z
            names.append(label)
        
        # transformed locations
        self._transformed = {}
        self._triangulations = {}
        
        # groups
        if groups:
            self.groups = groups
        else:
            self.groups = {}
        # mirror-map
        if mirror_map:
            self.mirror_map = mirror_map
        else:
            # TODO: construct
            self.mirror_map = {}
    
    def __repr__(self):
        return "sensor_net([<n=%i>], name=%r)" % (self.n, self.net_name)
        
    def __len__(self):
        return self.n
    
#    def get_improj(self, Y, proj='default', resolution=100, im_frame=0.02):
#        loc2d = self.getLocs2d(proj=proj)
#        emin = -im_frame
#        emax = 1 + im_frame
#        x = np.linspace(emin, emax, resolution)
#        x, y = np.meshgrid(x, x)
#        im = mlab.griddata(loc2d[:,0], loc2d[:,1], np.ravel(Y), 
#                           x, y, interp='linear') # linear about 6 times faster
#        return im
    
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
            coordinates will be scale with minimum value 0 and maximum value
            defined by the value of ``extent``.
        
        """
        if proj == 'default':
            proj = self.default_transform_2d
           
        index = (proj, extent) 
        if index in self._transformed:
            return self._transformed[index]
        
        
        if proj in ['cone', 'lowerCone', 'ideal']:
            
            # fit the 3d sensor locations to a sphere with center (cx, cy, cz)
            # and radius r
            
            # error function
            def err(params):
                r, cx, cy, cz = params
                return   (self.locs3d[:, 0] - cx) ** 2 \
                       + (self.locs3d[:, 1] - cy) ** 2 \
                       + (self.locs3d[:, 2] - cz) ** 2 \
                       -  r ** 2
    
            # initial guess of sphere parameters (radius and center)
            params = (1, 0, 0, 0)
            # do fit
            (r, cx, cy, cz), stuff = leastsq(err, params)
            
            # center the sensor locations based on the sphere and scale to
            # radius 1
            sphere_center = np.array((cx, cy, cz))
            locs3d = self.locs3d - sphere_center
            locs3d /= r
            
            # implement projection
            locs2d = np.copy(locs3d[:,:2])
            if proj=='cone':
                locs2d[:,0] *= (1 - locs3d[:,2]) 
                locs2d[:,1] *= (1 - locs3d[:,2])
            elif proj=='lower cone':
                for i in range(locs2d.shape[0]):
                    if locs3d[i,2] < 0:
                        locs2d[i,:2] *= (1 - locs3d[i,2]) 
            elif proj == 'ideal':
                r_sq = max(locs3d[:,2]) - locs3d[:,2]
                r = np.sqrt(r_sq)  # get radius dependent on z
                r_c = np.sqrt(locs3d[:,0]**2 + locs3d[:,1]**2) # current r
                F = r / r_c # stretching factor accounting for current r 
                locs2d[:,0] *= F
                locs2d[:,1] *= F
        
        elif proj == None:
            locs2d = np.copy(self.locs3d[:,:2])
        else:
            raise ValueError("invalid proj kwarg")
                
        # correct extent
        if extent:
            locs2d -= np.min(locs2d)
            locs2d /= (np.max(locs2d) / extent)
        
        # save for future access
        self._transformed[index] = locs2d
        return locs2d
    
    def get_tri(self, proj, resolution, frame):
        """
        Returns the farest generalizable object for projecting sensor maps
        to ims 
        
        Based on matplotlib.mlab.griddata function
        """
        locs = self.getLocs2d(proj)
        tri = delaunay.Triangulation(locs[:,0], locs[:,1])
        
        emin = - frame
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
            zo = interp(xi,yi)
        elif interp == 'linear':
            interp = tri.linear_interpolator(Z)
            zo = interp[yi.min():yi.max():complex(0,yi.shape[0]),
                        xi.min():xi.max():complex(0,xi.shape[1])]
        else:
            raise ValueError("interp keyword must be one of"
            " 'linear' (for linear interpolation) or 'nn'"
            " (for natural neighbor interpolation). Default is 'nn'.")
        # mask points on grid outside convex hull of input data.
        if np.any(np.isnan(zo)):
            zo = np.ma.masked_where(np.isnan(zo),zo)
        return zo
    
    def get_ROIs(self, base):
        """
        returns list if list of sensors, grouped according to closest
        spatial proximity to elements of base (=list of sensor ids)"
        
        """
        locs3d = self.locs3d
        #print loc3d
        base_locs = locs3d[base]
        ROI_dic = dict((i, [Id]) for i,Id in enumerate(base))
        for i, loc in enumerate(locs3d):
            if i not in base:
                dist = np.sqrt(np.sum((base_locs - loc)**2, 1))
                min_i = np.argmin(dist)
                ROI_dic[min_i].append(i)
        out = ROI_dic.values()
        return out
    
    def id2label(self, Id):
        return self.names[Id] 
    
    def ids2labels(self, *ids):
        return [self.names[Id] for Id in ids]
    
    def label2id(self, label):
        return self.names.index(label)
    
    def labels2ids(self, *labels):
        return [self.names.index(label) for label in labels]
    
    def plot_ROIs(self, ROIs, colors=['r','c','y','m','b', '.5']):
        colors = colors * (int(len(ROIs)/len(colors))+1)
        fig = P.figure(figsize=(4,4))
        locs2d = self.getLocs2d()
        for ROI in ROIs:
            locs = locs2d[ROI]
            P.scatter(locs[:,0], locs[:,1])
        return fig
    
    def subnet(self, sensors):
        """
        returns a new Sensor Net with a subset of sensors (specified as indexes)
        
        """
        if len(sensors) > 1:
            new_sensors = []
            for i in sensors:
                new_sensors.append(tuple(self.locs3d[i]) + (self.names[i],))
            return sensor_net(sensors)
        else:
            return None
    def subnet_ROIs(self, ROIs, loc='first'):
        """
        returns new sensor_net object based on senros in ROIs
        
        parameters
        ---------
        ROIs: list of lists of sensor ids
        
        loc: 'first': use the location of the first sensor of each ROI
             'mean': use the mean location
        
        """
        sensors = []
        for ROI in ROIs:
            i = ROI[0]
            name = self.names[i]
            if loc == 'first':
                l = self.locs3d[i]
            elif loc == 'mean':
                locs = self.locs3d[ROI]
                l = locs.mean(0)
            else:
                raise ValueError("invalid value for loc (%s)"%loc)
            sensors.append(tuple(l) + (name,))
        return sensor_net(sensors)



# MARK: constructors
# ==================

def from_list(sensorlist):
    """
    other list order than sensor_net.__init__
    
    expects list in format [ ["name",loc x, y, z], ...]
    
    e.g.:
    sensorlist = [("Cz", 0,  0,   0),
                  ("Pz", 0, -.25, -.45)]
    
    >>> net = SensorList(sensors)
    """
    sensors = []
    for name, x, y, z in sensorlist:
        sensors.append((x, y, z, name))
    return sensor_net(sensors)



def from_xyz(path=None, **kwargs):
#    if path == None:
#        path = ui.askfile()
    sensors = []
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
                sensors.append((x, y, z, name))
    assert len(sensors) == n
    return sensor_net(sensors, **kwargs)


def from_sfp(path=None, **kwargs):
    sensors = []
    for line in open(path):
        elements = line.split()
        if len(elements) == 4:
            name, x, y, z = elements
            x = float(x)
            y = float(y)
            z = float(z)
            sensors.append((x, y, z, name))
    return sensor_net(sensors, **kwargs)


def from_lout(path=None, transform_2d=None, **kwargs):
    kwargs['transform_2d'] = transform_2d
    sensors = []
    with open(path) as fileobj:
        fileobj.readline()
        for line in fileobj:
            w, x, y, t, f, name = line.split('\t')
            x = float(x)
            y = float(y)
            sensors.append((x, y, 0, name))
    return sensor_net(sensors, **kwargs)



def hcgsn129():
    path = os.path.join(sfp_path, "GSN-HydroCel-129Eclassic.sfp")
    groups = {'e16':[8, 10, 21, 23, 32, 44, 51, 57, 61, 74, 91, 95, 
                     107, 121, 123, 128],
              'eMid':[10, 5, 128, 61, 74],
              'pzEnv':[71, 61, 60, 66, 77, 76, 59, 84],
              'czEnv':[128, 105, 79, 54, 30, 6],
              'pczbig':[61, 60, 77, 78, 128, 105, 79, 53, 54, 30, 6],
              'pcz':[61, 60, 77, 78, 128, 79, 53, 54, 30],
              'fzEnv':[10, 15, 9, 3, 4, 5, 11, 18, 17],
              'tLeft':[33, 34, 39, 40, 44, 45, 46, 35, 41, 51, 55, 56, 49, 50, 
                       43, 48, 38, 62, 57],
              'tRight':[91, 92, 95, 96, 97, 98, 99, 100, 101, 102, 103, 106, 
                        107, 108, 109, 112, 113, 114, 115],
              'latr': [91, 92, 96, 97, 100, 101, 102, 103, 107, 108, 109, 113, 
                       114, 115],
              'linkedMast': [56, 99],
              }
    mirror_map = {13:20, 125:126, 
                  9:17, 8:21, 7:24, 
                  3:18, 2:22, 1:25, 0:31, 124:127,
                  123:23, 122:26, 121:32, 120:37, 119:42, 118:47,
                  4:11, 117:19, 116:27, 115:33, 113:43, 114:38, 112:48,
                  111:12, 110:28, 109:34, 108:39,
                  105:6, 104:29, 103:35, 102:40,
                  79:30, 86:36, 92:41, 97:46, 101:45, 107:44,
                  78:53, 85:52, 91:51, 96:50, 100:49, 99:56, 106:55,
                  77:60, 84:59, 90:58, 95:57,
                  76:66, 83:65, 89:64, 94:63, 98:62,
                  82:69, 88:68, 93:67, 81:73, 87:72
                  }
    net = from_sfp(path, name = "EGI HCGSN 129",
                   groups = groups,
                   mirror_map = mirror_map)
    return net

