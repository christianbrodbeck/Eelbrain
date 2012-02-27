
import numpy as np
import matplotlib.pyplot as plt
try:
    from mpl_toolkits.mplot3d import Axes3D as _Axes3D
except:
    _Axes3d = None



# some useful kwarg dictionaries for different plot layouts
kwargs_mono = dict(mc='k',
                   lc='.5',
                   hllc='k',
                   hlmc='k',
                   hlms=7,
                   strlc='k')


def _ax_map2d_fast(ax, sensor_net, proj='default', 
                   m='x', mew=.5, mc='b', ms=3,):
    if hasattr(sensor_net, 'sensors'):
        sensor_net = sensor_net.sensors
    
    locs = sensor_net.getLocs2d(proj=proj)
    h = plt.plot(locs[:,0], locs[:,1], m, color=mc, ms=ms, markeredgewidth=mew)
    
    return h

    
def _ax_map2d(ax, sensor_net, proj='default', hl=[], 
              labels='name', lc='k', ls=8, l_dist=.01, # labels, l colors, l size
              m='x', mew=.5, mc='b', ms=3, # marker, m edge width, m color, m size,
              strm=None, strc=None, strms=None, strlc='r', # ...same for string labels; None -> same as digits
              hlm='*', hlmc='r', hlms=5, hllc='r'): # ...same for highlight
    # in case sensor_net parent is submitted
    if hasattr(sensor_net, 'sensors'):
        sensor_net = sensor_net.sensors
    
    if strm == None:
        strm = m
    if strc == None:
        strmc = mc
    if strms == None:
        strms = ms
    if strlc == None:
        strlc = lc
    
    ax.set_aspect('equal')
    ax.set_frame_on(False)
    ax.set_axis_off()
    
    locs = sensor_net.getLocs2d(proj=proj)
    # labels
#    if labels == 'name':
#        labels = []
#        markers
#        colorList = []
#        for s in sensor_net:
#            label = s.name
#            if label.isdigit():
#                label = r'$'+label+'$'
#                colorList.append([])
#            else:
#                colorList.append('r')
#            labels.append(label)
#    elif labels == 'id':
#        labels = range(sensor_net.n)
#        colorList = ['k'] * len(labels)
#    elif labels== 'legend':
#        separator=':'
#        labels = [r"$%s$%s%s"%(i, separator, s.name) for i, s \
#                  in enumerate(sensor_net) ]
#        colorList = ['k']*len(labels)
#    else:
#        colorList = labels = [None]*sensor_net.n
#    # markers
#    markers = np.array([marker] * sensor_net.n, dtype='S2')
#    markers[highlight] = highlightMarker
    #transOffset = plt.offset_copy(plt.gca().transData, fig=fig, x = 0.05, y=0.10, units='inches')
    for i in range(sensor_net.n):
        x = locs[i,0]
        y = locs[i,1]
        # label
        if labels is None:
            label = None
        elif labels == 'id':
            label = label_for_c = str(i)
        elif labels == 'legend':
            separator=':'
            label_for_c = sensor_net.names[i]
            label = r"$%s$%s%s"%(i, separator, label_for_c)
        else:
            label = label_for_c = sensor_net.names[i]
        # properties
        if i in hl:
            marker, marker_c, marker_s, label_c, label_s = hlm, hlmc, hlms, hllc, ls
        elif (label!=None) and label_for_c.isdigit():
            marker, marker_c, marker_s, label_c, label_s = m, mc, ms, lc, ls
        else:
            marker, marker_c, marker_s, label_c, label_s = strm, strmc, strms, strlc, ls
        plt.plot([x],[y], marker, color=marker_c, ms=marker_s, markeredgewidth=mew)#,label=label)
        if label != None:
            plt.text(x, y+l_dist, label, fontsize=label_s,# style='oblique', 
                   horizontalalignment='center', verticalalignment='bottom', 
                   color=label_c)



def map2d(sensor_net, figsize=(5,5), frame=.01, **kwargs):
    """
    Arguments
    ---------
    
    ax: mpl.axes or ``None``
        target axes; a new fiigure is created if ``None``
    
    figsize:
        mpl figsize
    
    highlight: = []
        sensors which should be highlighted    
    
    labels: 
        how the sensors should be labelled: ``'name'``, ``'id'``, ``'legend'`` 
        (names and id), ``None``. Labels can be custmized with the following 
        additional arguments: ``lc='k'`` (label color), ``ls=8`` (label 
        font size), and ``ldist`` (distance from the marker).
    
    markers: 
        markers can be customized with the following arguments: ``m='x'`` 
        (marker symbol), ``mc='b'`` (color), ``ms=3`` (size) and ``mew=0.5`` 
        (marker edge width).
    
    proj:
        Transform to apply to 3 dimensional sensor coordinates for plotting 
        locations in a plane
    
    """    
    # figure
    fig = plt.figure(figsize=figsize, facecolor='w')
    ax = plt.axes([frame, frame, 1 - 2 * frame, 1 - 2 * frame])
        # the following does not make the plot
#        fig = mpl.figure.Figure(figsize=figsize, facecolor='w')
#        ax = fig.add_axes([0,0,1,1])
    _ax_map2d(ax, sensor_net, **kwargs)
    
    return fig




def map3d(sensor_net, marker='c*', labels=False, headBall=0):
    """not very helpful..."""
    if _Axes3D is None:
        raise ImportError("mpl_toolkits.mplot3d.Axes3D could not be imported")
    
    if hasattr(sensor_net, 'sensors'):
        sensor_net = sensor_net.sensors
    locs = sensor_net.locs3d
    fig = plt.gcf()
    ax = _Axes3D(fig)
    ax.scatter(locs[:,0], locs[:,1], locs[:,2])
    # plot head ball
    if headBall>0:
        u = np.linspace(0, 1 * np.pi, 10)
        v = np.linspace(0, np.pi, 10)
        
        x = 5 * headBall * np.outer( np.cos(u), np.sin(v))
        z = 10 * (headBall * np.outer( np.sin(u), np.sin(v)) -.5)         # vertical
        y = 5 * headBall * np.outer( np.ones(np.size(u)), np.cos(v))  # axis of the sphere
        ax.plot_surface(x, y, z,  rstride=1, cstride=1, color='w')
    #n = 100
    #for c, zl, zh in [('r', -50, -25), ('b', -30, -5)]:
    #xs, ys, zs = zip(*
    #               [(random.randrange(23, 32),
    #                 random.randrange(100),
    #                 random.randrange(zl, zh)
    #                 ) for i in range(n)])
    #ax.scatter(xs, ys, zs, c=c)
