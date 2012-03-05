"""
plots for universal time series


Plot types
----------

topo:
    show a tv plot for different sensors, arranged according to a topographical
    sensor layout.
corr:
    plots the correlation between a timeseries and a variable over time
uts:
    
"""

from __future__ import division

import logging
import os
from copy import deepcopy

import numpy as np
import scipy as sp
#import matplotlib as mpl
import matplotlib.pyplot as P

import eelbrain.fmtxt as fmtxt
from eelbrain.analyze.plot import _simple_fig

from eelbrain.utils import _basic_ops_
from eelbrain.vessels import colorspaces as _cs
#from eelbrain.signal_processing import segment_ops
from eelbrain.analyze.testnd import test as _test

import _base




def array(segments, sensors=None, plotAllSubjects=False, plotCbar=True, test='nonparametric', p=.05, **kwargs):
    """
    NOT MAINTAINED
    plots tv plots to a rectangular grid instead of topographic spacing
     
    kwargs:
    sensors: List of sensor IDs
    test='nonparametric', None if len(segments) > 2
    
    """
    P.clf()
    # prepare args
    segments = _basic_ops_.toTuple(segments)
    kwargs['test']=test
    if sensors==None:
        sensors = range(len(segments[0].sensors))
    else:
        sensors = _basic_ops_.toTuple(sensors)
    statsColorspace = _cs.get_sig(p)
    # determine plotting grid
    nPlots = len(sensors) * (1+plotAllSubjects) + plotCbar
    nColumnsEst = np.sqrt( nPlots ) 
    nColumns    = int(nColumnsEst)
    if nColumns != nColumnsEst:
        nColumns += 1
    if nColumns % (1+plotAllSubjects) != 0:
        nColumns += 1
    nRows = int( nPlots / nColumns ) + (1- ( nPlots % nColumns == 0 ))
    logging.debug("plotting electrodes %s, fig shape (%s, %s)"%(str(sensors), nRows, nColumns))
#    fig = P.gcf()
    P.subplots_adjust(left=.05, bottom=.075, right=.99, top=.94, wspace=.25, hspace=.3)
    #grid = AxesGrid(    fig, 111,
    #                    nrows_ncols = (nRows, nColumns),
    #                    axes_pad = .01 )
    # plot
    kwargs['labelNames']=True
    for i,sensor in enumerate(sensors):
        kwargs['lastRow'] = ( i > (nRows-1) * nColumns)
        if plotAllSubjects:
            #axes = grid[ (i+1)*2 ] # 
            axes = P.subplot(nRows, nColumns, (i+1) * 2 -1 )
            kwargs["plotType"]='mean'
            _ax_utsStats(segments, sensor, statsColorspace=statsColorspace, **kwargs)
            axes.set_title( segments[0].sensors[sensor].name )
            #axes = grid[ (i+1)*2+1 ] #
            axes = P.subplot(nRows, nColumns, (i+1) * (1+plotAllSubjects) )
            kwargs["plotType"]='all'
            _ax_utsStats(segments, sensor, statsColorspace=statsColorspace, **kwargs)
            axes.set_title( ', '.join(( segments[0].sensors[sensor].name, "All Subjects")) )
        else:
            #axes = grid[i] #
            axes = P.subplot(nRows, nColumns, (i+1) )
            kwargs["plotType"]='mean'
            _ax_utsStats(segments, sensor, statsColorspace=statsColorspace, 
                            firstColumn=( i%nColumns==0 ),
                            **kwargs)
            axes.set_title( segments[0].sensors[sensor].name )
        kwargs['labelNames']=False
    if test!=None and plotCbar:
        #axes = grid[-1] #
        axes = P.subplot(nRows, nColumns, nRows*nColumns)
        pos = axes.get_position()
        newPos = [pos.xmin, pos.ymin+pos.width*.3, pos.width, pos.width*.15]
        axes.set_position(newPos)
        statsColorspace.toAxes_(axes)


def corr(stats, var, p=[.05],
         figsize=(6.,4.),
         axrect=[.08,.15,.9,.75],
         legend_ncol=4,
         ):
    """
    plots the correlation between a timeseries and a variable over time
    
    """
#    if iscollection(stats):
#        stats = [stats]
#    assert all(iscollection(s) for s in stats)
    corr = [segment_ops.corr(s, var) for s in stats] 
    
    if np.isscalar(p):
        p = [p]
    # P (http://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient#Inference)
    n = len(corr[0]['slist'])
    if not all([len(corr[i]['slist']) == n for i in range(1, len(corr))]):
        raise ValueError("StatsSegments have different N --> p threshold cannot be drawn correctly")
    Ps = np.array(p) / 2 # 2 tailed
    df = n-2
    Ts = sp.stats.t.isf(Ps, df)
    R_thresholds = Ts / np.sqrt(df + Ts**2)
    
    P.figure(figsize=figsize)
    ax = P.axes(axrect)
    legend_names = []
    legend_handles = []
    # corr
    T = stats[0].t
    for c in corr:
        R = c.param
        handle = P.plot(T, R, label=c.name)
        legend_names.append(fmtxt.texify(c.name))
        legend_handles.append(handle)
    # thresholds
    P.axhline(0, c='k')
    for p in R_thresholds:
        P.axhline(p)
        P.axhline(-p)
    # Figure stuff
    P.ylabel('r')
    P.xlabel('time')
    P.suptitle("Correlation with {v}".format(v=fmtxt.texify(var.name)), fontsize=16)
    P.figlegend(legend_handles, legend_names, 'lower center', ncol=legend_ncol)



def uts(statsSegments, sensor=0, 
        lineprops={},
        title=None, legend='lower left',
         c=None, # list of colors, e.g.:  c= ['r','.75','c','m']]
        tw=None, # time window
        marktw = None, # [y, t1, t2, c] Mark a time window with a horizontal line
        legend_ncol=2, legend_fig=False,
        sensor_name = True,
        #sensor_name_y = 1.5,
        figsize=(3.,3.),
        savepath=False,
        saveformat='pdf',
        tax = True, # use T-axes instead of normal axes (for EEG) 
        xlabel="t [seconds]", ylabel=True,
        test=False,
        **kwargs):
    """
    Single plot for on or more time-series 
    
    Arguments
    ---------
    
    
    spread:
        all = False: True  -> plot all subjects 
                 False -> plot mean
    
    
    Also accepts 
    - subdata kwargs
    - uts plot kwargs:
    
    corr:
        dict: id->value
              str(label)->value
        Parasite
    
    """
    # kwars
    if 'ROI' in kwargs:
        pass # sensor = 0
    else:
        kwargs['sensor'] = sensor
    statsSegments = _keywords_.sub_segments(statsSegments, kwargs)
    if test:
        kwargs['test_segment'] = _test(statsSegments)
    names, s_name, seg_t_start, seg_t_end = _base._get_attributes(statsSegments, sensor)
    if ylabel is True:
        ylabel = False
    if sensor_name == True:
        sensor_name = s_name
    if c is not None:
        lineprops=[{'color':color} for color in c] # e.g.:  c= ['r','.75','c','m']]
    
    
    # axes
    if tax:
        x0 = .07 + bool(ylabel)*.05
        x1 = .97 - x0
        y0 = .15 + bool(xlabel)*.05
        y1 = .97 - y0 - bool(title)*.05
        axrect = (x0, y0, x1, y1)
#        axrect = (x0, x1, y0, y1)
        fig = P.figure(figsize=figsize)
        ax = _t_axes(axrect, xmin=seg_t_start, xmax=seg_t_end, ticks=True)
        ax.set_xlabel(xlabel)
        if ylabel not in [False, None]:
            ax.set_ylabel(ylabel)
    else:
        fig = _simple_fig(title, xlabel, ylabel, figsize=figsize)#, titlekwargs, **simple_kwargs)
        ax = fig.ax
#        ax = P.axes(axrect)
    
    # plot uts segments
    handles = _ax_utsStats(statsSegments, sensor, ax=ax,
                           lineprops=lineprops,
                           **kwargs)
    if tw:
        P.axvspan(tw[0], tw[1], facecolor=(1.,1.,0), edgecolor='y', alpha=1., zorder=-10)
#        if title is None:
#            title = "Time Window %s - %s s"%tuple(tw)
    if sensor_name:
        #P.text(seg_t_start, sensor_name_y, sensor_name, family='serif', size=12)
        P.figtext(.01,.85, sensor_name, family='serif', size=12)
    if marktw:
        if type(marktw[0]) not in [list, tuple]:
            marktw = [marktw]
        for y, t1, t2, c in marktw:
            ax.plot([t1, t2], [y, y], color=c, marker='|')
    
    if tax:
        if title:
            P.suptitle(title, size=12, family='serif')
        if legend:
            P.figlegend(handles, names, loc=legend, ncol=legend_ncol)
    else:
        fig.add_legend_handles(*handles)
        if legend:
            fig.legend(loc=legend, fig=legend_fig, zorder=-1, ncol=legend_ncol)
        fig.finish()
    
    if savepath:
        filename = 'twtv_'+','.join([s.name for s in statsSegments])+'.'+saveformat
        P.savefig(os.path.join(savepath, filename))
    return fig



def mark_tw(tstart, tend, y=0, c='r', ax=None):
    "mark a time window in an existing axes object"
    if ax == None:
        ax = P.gca()
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    
    ax.plot((tstart, tend), (y,y), color=c, marker='|')
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)



def _t_axes(rect, xmin=-.5, xmax=1., vmax=1.5, vbar=1., markevery=.5, ticks=False):
    """
    creates and returns a new t-axes using rect
    
    vbar: extent of the vertical bar at the origin (+vbar to -vbar)
    
    """
    ax = P.axes(rect, frameon=False)
    ax.set_axis_off()
    
    # vertical bar
    P.plot([0,0], [-vbar,vbar], 'k', 
           marker='_', mec='k', mew=1, mfc='k')
    
    # horizontal bar
    xdata = np.arange(-markevery, xmax + 1e-5, markevery)
    xdata[0] = xmin
    P.plot(xdata, xdata*0, 'k', marker='|', mec='k', mew=1, mfc='k')
    logging.debug("xdata: %s"%str(xdata))
    
    # labels
    if ticks:
        ax.text(xmax, vbar*.1, r"$%s s$"%xmax, verticalalignment='bottom', horizontalalignment='center')
        ax.text(-vbar*.05, vbar, r"$%s \mu V$"%vbar, verticalalignment='center', horizontalalignment='right')
        ax.text(-vbar*.05, -vbar, r"$-%s \mu V$"%vbar, verticalalignment='center', horizontalalignment='right')
    ax.set_ylim(-vmax, vmax)
    ax.set_xlim(xmin, xmax)
    return ax



def _axgrid_sensors(sensorLocs2d, figsize=_base.defaults['figsize'], 
                    spacing=.2, frame=.01, figvstretch=1, 
                    header=0, footer=0,     # in inches
                    axes_legend_loc='lower left', **axkwargs):
    """
    creates topographocally distributed t-axes
    
     returns 
        - list of t-axes
        - axes-legend (indicating t and V) (or None if axes_legend_loc == False) 
    """
    # determine figure size and create figure
    sensorLocs2d[:,1] *= figvstretch 
    x, y = sensorLocs2d.max(axis=0) - sensorLocs2d.min(axis=0)
    ratio = (x+spacing) / (y+spacing)
    x_size, y_size = figsize
    if x_size==-1:  
        x_size = y_size * ratio
    elif y_size==-1:
        y_size = x_size / ratio
    # take into account footer & header
    y_size += footer + header
    relative_footer = footer / float(y_size)
    relative_header = header / float(y_size)
    logging.debug(" _axgrid_sensors determined figsize %s x %s"%(x_size, y_size))
    fig = P.figure(figsize=(x_size, y_size))
    # determine axes locations
    locs = sensorLocs2d
    # normalize range to 0--1
    locs -= locs.min(0)
    locs /= locs.max(0)
    # locs--> lower left points of axes
    locs[:,0] *= (1 - spacing - 2*frame)
    locs[:,0] += frame
    locs[:,1] *= (1 - spacing - 2*frame - relative_header - relative_footer)
    locs[:,1] += frame + relative_footer
    #print locs
    # add axes
    axes = [ _t_axes([x,y,spacing,spacing], **axkwargs) for x,y in locs ]
    if axes_legend_loc:
        x, y = _base._loc(axes_legend_loc, size=(1.1*spacing, spacing), frame=frame)
        axes_legend = _t_axes([x,y,spacing,spacing], ticks=True, **axkwargs)
    else:
        axes_legend = None
    return axes, axes_legend


def _plt_uts(ax, epoch, 
             sensors=None, # sensors (ID) to plot
             lp=None, # line-properties
             color=None,
             test_epoch=False, p=.05, softStats=False, #testWindowFreq='max',
             sem=None,        # 'sem' (float multiplier)
             plotLabel=False,
             **plot_kwargs):
    """
    plots a uts plot for a single epoch 
    
    
    Arguments
    ---------
    
    lp: dictionary (line-properties)
        any keyword-arguments for matplotlib plot
    sem: = None or float
        plot standard error of the mean (e.g., ``sem=2`` plots the mean +/- 2
        sem)
    test_epoch:
        submit a test_epoch to add to plot (efficient because usually 
        _ax_utsStats is called more than once for several epochs     
    plottype:
        ``'mean'``: plots the mean for each stats epoch
        ``'all'``:  plots all data traces contained in the stats, colors mark the 
        epoch NOT MAINTAINED
    
    NOT IMPLEMENTED epochs: submit mean epochs
    
    """
    epoch.assert_dims(('time', 'sensor'))
    # prepare args
    
    Y = epoch.get_epoch_data()
    if sensors:
        Y = Y[:,sensors]
    T = epoch.time#.x[...,None]
    
    handles = ax.plot(T, Y, label=epoch.name, **plot_kwargs)
    
    if plotLabel:
        Ymax = np.max(Y)
        ax.text(T[0]/2, Ymax/2, plotLabel, horizontalalignment='center')
    
    return handles



def _plt_extrema(ax, epoch, **plot_kwargs):
    epoch.assert_dims(('time', 'sensor'))
    data = epoch.get_epoch_data()
    Ymin = data.min(1)
    Ymax = data.max(1)
    T = epoch.time
    
    handle = ax.fill_between(T, Ymin, Ymax, **plot_kwargs)
    ax.set_xlim(T[0], T[-1])
        
    return handle


def _ax_butterfly(ax, layers, sensors=None, ylim=None, extrema=False,
                  title=True, xlabel=True, ylabel=True, color=None, 
                  **plot_kwargs):
    """
    Arguments
    ---------
    
    ylim:
        y axis limits (scalar or (min, max) tuple)
    
    """
    handles = []
    
    xmin = []
    xmax = []
    for l in layers:
        if color is None:
            plot_kwargs['color'] = l.properties.get('color', 'k')
        elif color is not True:
            plot_kwargs['color'] = color
        
        # plot
        if extrema:
            h = _plt_extrema(ax, l, **plot_kwargs)
        else:
            h = _plt_uts(ax, l, sensors=sensors, **plot_kwargs)
        
        handles.append(h)
        xmin.append(l.time[0])
        xmax.append(l.time[-1])
    
    # axes decoration
    l = layers[0]
    if xlabel is True:
        xlabel = 'Time [s]'
    if ylabel is True:
        ylabel = l.properties.get('unit', None)
    if ylim is None:
        ylim = l.properties.get('ylim', None)
    
    ax.set_xlim(min(xmin), max(xmax))
    
    if ylim:
        if np.isscalar(ylim):
            ax.set_ylim(-ylim, ylim)
        else:
            y_min, y_max = ylim
            ax.set_ylim(y_min, y_max)
    
    if xlabel not in [False, None]:
        ax.set_xlabel(xlabel)
    if ylabel not in [False, None]:
        ax.set_ylabel(ylabel)
    
    return handles


def butterfly(segment, sensors=None, title=True, ylim=None, 
              # tax=False,  
              xlabel=True, ylabel=True, color=None, **plot_kwargs):
    """
    Creates a butterfly plot
    
    Arguments
    ---------
    
    color: (mpl color)
        default (``None``): use segment color if available, otherwise black; 
        ``True``: alternate colors (mpl default)
    title: bool or string
        Title for the axes. If ``True``, the segment's name is used.
    sensors: None or list of sensor IDs
        sensors to plot (``None`` = all)
    ylim: 
        scalar or (min, max) tuple specifying the y-axis limits (the default
        ``None`` leaves mpl's default limits unaffected)
    
    """
    if title is True:
        title = segment.name
    
    # create axes
#    if tax:
#        fig = P.figure()
#        x0 = .07 + bool(ylabel)*.05
#        x1 = .97 - x0
#        y0 = .15 + bool(xlabel)*.05
#        y1 = .97 - y0 - bool(title)*.05
#        axrect = (x0, y0, x1, y1)
#        ax = _t_axes(axrect, xmin=segment.tstart, xmax=segment.tend, 
#                     ticks=True, vmax=segment.data.max())
#    else:
    fig = _simple_fig(title, xlabel, ylabel)#, titlekwargs, **simple_kwargs)
    ax = fig.ax
    
    _ax_butterfly(ax, segment, sensors=sensors, ylim=ylim, title=title, 
                  xlabel=xlabel, ylabel=ylabel, color=color, **plot_kwargs)
    
    


        
# MARK: Ax Plots

def _ax_utsStats(segments, sensor, 
                 plotMean=True,
                 all=False,
                 ax=None,
                 lineprops={},   # c, ls, lw; doct or list of dicts 
                                 # e.g. [{'c':'b'},{'c':'r'},{'c':'g'},'m','c'],
                 test_segment=False, p=.05, softStats=False, #testWindowFreq='max',
                 sem=None,        # 'sem' (float multiplier)
                 ylim=None,
                 plotLabel=False,
                 statsColorspace=None,
                 **kwargs):
    """
    takes a list of segments and plots one sensor to ax. 
    
    
    Arguments
    ---------
    
    sem: = None or float
        plot standard error of the mean (e.g., ``sem=2`` plots the mean +/- 2
        sem)
    test_segment:
        submit a test_segment to add to plot (efficient because usually 
        _ax_utsStats is called more than once for several segments     
    plottype:
        ``'mean'``: plots the mean for each stats segment
        ``'all'``:  plots all data traces contained in the stats, colors mark the 
        segment NOT MAINTAINED
    
	NOT IMPLEMENTED segments: submit mean segments
	
    """
    # prepare arguments
    if statsColorspace is None:
        statsColorspace = _cs.get_sig_white()
    if ax is None:
        logging.debug(" _ax_utsStats got None ax")
        ax = P.gca()
    if ylim is None:
        ylim = segments[0].properties.get('ylim', None)
    
    # prepare args
    segments = _basic_ops_.toTuple(segments)
    seg = segments[0]
    assert seg.ndim == 1
    if ylim is None:
        if all:
            minV, maxV = -7.5, 7.5
        else:
            minV, maxV = -1.5, 1.5
    else:
        minV, maxV = -ylim, ylim
    # t
#    tmin = -seg.t0; 
#    xaxis_length = len(seg)
#    tmax = ( float(xaxis_length)/seg.samplingrate ) - seg.t0
#    t = np.r_[tmin : tmax : xaxis_length*1j]

    # line properties
    if type(lineprops) == dict:
        lineprops = [deepcopy(lineprops) for i in range(len(segments))]
        # (need separate instance because properties are is modified later)
    
    #plot
    handlesForLegend=[]
    for s, lp in zip(segments, lineprops):
        line_properties = {'alpha':.2,
                           'zorder':-1}
        if ('color' not in lp) and s.color:
            lp['color']=s.color
        
        line_properties.update(lp)
        mean = s.mean().subdata(out='data', sensors=[sensor])[:,0] #data[:,sensor]
        t = s.t
        
        logging.debug("_ax_utsStats plotting '%s'"%s.name)
        if plotMean:
            line = ax.plot(t, mean, label=s.name, **lp)[0] # label=label
            line_properties.update({'color':line.get_color()})
        if all:
            #print "%s, %s"%(sensor, type(sensor))
            single_line = ax.plot(t[...,None], s.data[:,sensor], **line_properties)[0]
        if sem:
            #print "%s, %s"%(sensor, type(sensor))
            line_properties.update({'alpha':.2})
            sem_array = sp.stats.sem(s.data[:,sensor], axis=-1) * sem
            ax.fill_between(t, mean-sem_array, mean+sem_array, **line_properties)
        handlesForLegend.append(line)
    
    # p-bar
    if test_segment:
        imKwargs={'zorder': -100}
        statsColorspace.toKwargs(imKwargs)
        im = test_segment.P[:, [sensor]].T
#        l = len(im) 
#        im = np.ma.array(np.vstack([np.ones(l), im, np.ones(l)]), 
#                         mask=np.vstack([np.ones(l), im>p, np.ones(l)]))
        if softStats:
            pad = np.ones(im.shape)
            im = np.vstack((pad,im,im,im,pad))
            imKwargs['interpolation'] = 'bicubic'
        else:
            imKwargs['interpolation'] = 'nearest'
        
        extent = (t[0], t[-1]) + ax.get_ylim()
        
        ax.imshow(im, extent=extent, aspect='auto', **imKwargs)
    else:
        ax.set_xlim(t[0], t[-1])
    
    if plotLabel:
        ax.text(t[0]/2, maxV/2, plotLabel, horizontalalignment='center')
    return handlesForLegend
    


def _ax_sensor(segments, sensor, labelVar=None, **kwargs):
    """
    NOT MAINTAINED
    plots one sensor (index) for a list of segments
    return plot
    """
    # make sure segment and sensors are iterable
    if type(segments) != list:
        segments = [segments]
    if labelVar == None:
        labelVar = segments[0].experiment.subjectVariable
    print "ndim:",segments[0].ndim
    if segments[0].hasStatistics:
        return plotSensorStats(segments, sensor, **kwargs)
    elif segments[0].ndim == 1:
        plots = []
        for s in segments:
            try:
                label = s[labelVar]
            except:
                label = "no name"
            print "plotting segment", s
            plots.append( P.plot(s[:,sensor], label=label, **kwargs) )
        return plots
    elif segments[0].ndim == 2:
        raise NotImplementedError( "ndim 2 not implemented" )
        
        

def _ax_eArray(segment, increment=1, ax=None, **kwargs):
    """plots all channels of segment to one axes"""
    if ax==None:
        ax=P.gca()
    data = segment.data
    ad = np.arange(0, data.shape[1], increment)
    return ax.plot(data+ad, **kwargs)




class anova_results_1d():
    def __init__(self, test_segments, w=5):
        "w: figure width"
        self.fig = P.figure(figsize=(w, w/4.))
        
        # rect:  left bottom width hight
        ax = self.fig.add_axes((.18,.15,.8,.8))
        _ax_im_multi1d(test_segments, 
                       fig=self.fig, ax=ax)


#class _ax_im_multi1d(_ax_):
#    """
#    Plots multiple 1-d segments to one axis. For example for ANOVA results, 
#    where different effects are returned in different segments.
#     
#    """
#    def __init__(self, segments, cs=None, fig=None, ax=None):
#        _ax_.__init__(self, fig=fig, ax=ax)
#        
#        y_ticks = []
#        y_ticklabels = []
#        
#        im_kwargs={}
#        for i, segment in enumerate(segments):
#            # colorspace
#            if cs is None:
#                cs = segment.defaultColorspace()
#            cs.toKwargs(im_kwargs)
#            
#            # imshow
#            self.ax.imshow(segment.data.T, aspect='auto', interpolation='nearest',
#                           extent = (segment.tstart, segment.tend, i-.5, i+.5),
#                           **im_kwargs)
#            
#            # collect labels
#            y_ticks.append(i)
#            y_ticklabels.append(segment.name)
#        
#        self.ax.set_ylim(-.5, len(segments)-.5)
#        
#        # set y labels
#        self.ax.set_yticks(y_ticks)
#        self.labels = self.ax.set_yticklabels(y_ticklabels)
#        
#        # connect 
#        self.fig.canvas.mpl_connect('draw_event', self._on_draw) 
#        logging.debug('connected')
#    def _on_draw(self, event):
#        "after Matplotlib.pdf, p. 171" 
#        logging.debug("ax labels adjusting...")
#        bboxes = []
#        for label in self.labels: 
#            bbox = label.get_window_extent() 
#            # the figure transform goes from relative coords->pixels and we 
#            # want the inverse of that 
#            bboxi = bbox.inverse_transformed(self.fig.transFigure) 
#            bboxes.append(bboxi)
#        
#        # this is the bbox that bounds all the bboxes, again in relative 
#        # figure coords bbox = mtransforms.Bbox.union(bboxes) 
#        if self.fig.subplotpars.left < bbox.width:
#            # we need to move it over 
#            self.fig.subplots_adjust(left=1.1*bbox.width) # pad a little 
#            logging.debug('adjust!')
#            self.fig.canvas.draw()
#        return False
#        
#        


