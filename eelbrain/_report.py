# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import numpy as np

from . import fmtxt
from . import plot
from . import test
from ._data_obj import cellname
from .fmtxt import ms


def format_samples(res):
    if res.samples == -1:
        return "a complete set of %i permutations" % res.n_samples
    elif res.samples is None:
        return "no permutations"
    else:
        return "%i random permutations" % res.n_samples


def format_timewindow(res):
    "Format a description of the time window for a test result"
    uts = res._time_dim
    return 'between %i and %i ms' % (tstart(res.tstart, uts),
                                     tstop(res.tstop, uts))


def tstart(tstart, uts):
    if tstart is None:
        return ms(uts.tmin)
    else:
        return ms(tstart)


def tstop(tstop, uts):
    if tstop is None:
        return ms(uts.tmax + uts.tstep)
    else:
        return ms(tstop)


def sensor_bin_table(section, res, pmin=None):
    if pmin is None:
        caption = "All clusters"
    else:
        caption = "p <= %.s" % pmin

    for effect, cdist in res._iter_cdists():
        ndvar = cdist.masked_parameter_map(pmin)
        if not ndvar.any():
            if effect:
                text = '%s: nothing\n' % effect
            else:
                text = 'Nothing\n'
            section.add_paragraph(text)
            continue
        elif effect:
            caption_ = "%s: %s" % (effect, caption)
        else:
            caption_ = caption
        p = plot.TopomapBins(ndvar, show=False)
        section.add_image_figure(p, caption_)


def sensor_time_cluster(section, cluster, y, model, ds, colors, legend):
    # cluster properties
    tstart_ms = ms(cluster['tstart'])
    tstop_ms = ms(cluster['tstop'])

    # section/title
    title = ("{tstart}-{tstop} p={p}{mark} {effect}"
             .format(tstart=tstart_ms, tstop=tstop_ms,
                     p='%.3f' % cluster['p'],
                     effect=cluster.get('effect', ''),
                     location=cluster.get('location', ''),
                     mark=cluster['sig']).strip())
    while '  ' in title:
        title = title.replace('  ', ' ')
    section = section.add_section(title)

    # description
    paragraph = section.add_paragraph("Id %i" % cluster['id'])
    if 'v' in cluster:
        paragraph.append(", v=%s" % cluster['v'])

    # add cluster image to report
    topo = y.summary(time=(cluster['tstart'], cluster['tstop']))
    cluster_topo = cluster['cluster'].any('time')
    cluster_topo.info['contours'] = {0.5: (1, 1, 0)}
    x = ds.eval(model)
    topos = [[topo[x == cell].summary('case', name=cellname(cell)),
              cluster_topo] for cell in x.cells]
    p = plot.Topomap(topos, axh=3, nrow=1, show=False)
    p.mark_sensors(np.flatnonzero(cluster_topo.x), 'yo')

    caption_ = ["Cluster"]
    if 'effect' in cluster:
        caption_.extend(('effect of', cluster['effect']))
    caption_.append("%i - %i ms." % (tstart_ms, tstop_ms))
    caption = ' '.join(caption_)
    section.add_image_figure(p, caption)
    p.close()

    return cluster_timecourse(section, cluster, y, 'sensor', model, ds, colors,
                              legend)


def source_bin_table(section, res, pmin=None):
    caption = ("All clusters in time bins. Each plot shows all sources "
               "that are part of a cluster at any time during the "
               "relevant time bin. Only the general minimum duration and "
               "source number criterion are applied.")

    for effect, cdist in res._iter_cdists():
        ndvar = cdist.masked_parameter_map(pmin)
        if not ndvar.any():
            if effect:
                text = '%s: nothing\n' % effect
            else:
                text = 'Nothing\n'
            section.add_paragraph(text)
            continue
        elif effect:
            caption_ = "%s: %s" % (effect, caption)
        else:
            caption_ = caption
        im = plot.brain.bin_table(ndvar, surf='inflated')
        section.add_image_figure(im, caption_)


def source_time_clusters(section, clusters, y, ds, model, include,
                          title, colors, legend=None):
    """
    Parameters
    ----------
    ...
    legend : None | fmtxt.Image
        Legend (if shared with other figures).

    Returns
    -------
    legend : fmtxt.Image
        Legend to share with other figures.
    """
    # compute clusters
    if clusters.n_cases == 0:
        section.append("No clusters found.")
        return
    clusters = clusters.sub("p < 1")
    if clusters.n_cases == 0:
        section.append("No clusters with p < 1 found.")
        return
    caption = "Clusters with p < 1"
    table_ = clusters.as_table(midrule=True, count=True, caption=caption)
    section.append(table_)

    # plot individual clusters
    clusters = clusters.sub("p < %s" % include)
    for cluster in clusters.itercases():
        legend = source_time_cluster(section, cluster, y, model, ds, title,
                                     colors, legend)

    return legend


def source_time_cluster(section, cluster, y, model, ds, title, colors, legend):
    # cluster properties
    tstart_ms = ms(cluster['tstart'])
    tstop_ms = ms(cluster['tstop'])

    # section/title
    if title is not None:
        title_ = title.format(tstart=tstart_ms, tstop=tstop_ms,
                              p='%.3f' % cluster['p'],
                              effect=cluster.get('effect', ''),
                              location=cluster.get('location', ''),
                              mark=cluster['sig']).strip()
        while '  ' in title_:
            title_ = title_.replace('  ', ' ')
        section = section.add_section(title_)

    # description
    txt = section.add_paragraph("Id %i, v=%s." % (cluster['id'], cluster['v']))
    if 'p_parc' in cluster:
        txt.append("Corrected across all ROIs: ")
        txt.append(fmtxt.eq('p', cluster['p_parc'], '=', drop0=True, fmt='%s'))
        txt.append('.')

    # add cluster image to report
    brain = plot.brain.cluster(cluster['cluster'].sum('time'),
                               surf='inflated')
    image = plot.brain.image(brain, 'cluster_spatial.png', close=True)
    caption_ = ["Cluster"]
    if 'effect' in cluster:
        caption_.extend(('effect of', cluster['effect']))
    caption_.append("%i - %i ms." % (tstart_ms, tstop_ms))
    caption = ' '.join(caption_)
    section.add_image_figure(image, caption)

    return cluster_timecourse(section, cluster, y, 'source', model, ds, colors,
                              legend)


def cluster_timecourse(section, cluster, y, dim, model, ds, colors,
                       legend=None):
    c_extent = cluster['cluster']
    cid = cluster['id']

    # cluster time course
    idx = c_extent.any('time')
    tc = y[idx].mean(dim)
    p = plot.UTSStat(tc, model, match='subject', ds=ds, legend=None, h=4,
                     colors=colors, show=False)
    # mark original cluster
    for ax in p._axes:
        ax.axvspan(cluster['tstart'], cluster['tstop'], color='r',
                   alpha=0.2, zorder=-2)
    image_tc = p.image('cluster_%i_timecourse.svg' % cid)

    # legend
    if legend is None:
        legend_p = p.plot_legend(show=False)
        legend = legend_p.image("Legend.svg")
        legend_p.close()
    p.close()

    # Barplot
    idx = (c_extent != 0)
    v = y.mean(idx)
    p = plot.Barplot(v, model, 'subject', ds=ds, corr=None, colors=colors,
                     h=4, show=False)
    image_bar = p.image('cluster_%i_barplot.png' % cid)
    p.close()

    # Boxplot
    p = plot.Boxplot(v, model, 'subject', ds=ds, corr=None, colors=colors,
                     h=4, show=False)
    image_box = p.image('cluster_%i_boxplot.png' % cid)
    p.close()

    # compose figure
    section.add_figure("Time course in cluster area, and average value in "
                       "cluster by condition, with pairwise t-tests.",
                       [image_tc, image_bar, image_box, legend])

    # pairwise test table
    res = test.pairwise(v, model, 'subject', ds=ds)
    section.add_figure("Pairwise t-tests of average value in cluster by "
                       "condition", res)

    return legend


def roi_timecourse(doc, ds, label, model, res, colors):
    "Plot ROI time course with cluster permutation test"
    y = ds.info['label_ids'][label]
    label_name = label[:-3].capitalize()
    hemi = label[-2].capitalize()
    title = ' '.join((label_name, hemi))
    caption = "Source estimates in %s (%s)." % (label_name, hemi)
    timecourse(doc, ds, y, model, res, title, caption, colors)


def timecourse(doc, ds, y, model, res, title, caption, colors):
    clusters = res.find_clusters()
    if clusters.n_cases:
        idx = clusters.eval("p.argmin()")
        max_sig = clusters['sig'][idx]
        if max_sig:
            title += max_sig
    section = doc.add_section(title)

    # compose captions
    if clusters.n_cases:
        c_caption = ("Clusters %s based on %s."
                     % (format_timewindow(res), format_samples(res)))
        tc_caption = caption
    else:
        c_caption = "No clusters found %s." % format_timewindow(res)
        tc_caption = ' '.join((caption, c_caption))

    # add UTSStat plot
    p = plot.UTSStat(y, model, match='subject', ds=ds, colors=colors,
                     legend=None, clusters=clusters, show=False)
    ax = p._axes[0]
    if res.tstart is not None:
        ax.axvline(res.tstart, color='k')
    if res.tstop is not None:
        ax.axvline(res.tstop, color='k')
    image = p.image('%s_cluster.png')
    legend_p = p.plot_legend(show=False)
    legend = legend_p.image("Legend.svg")
    section.add_figure(tc_caption, [image, legend])
    p.close()
    legend_p.close()

    # add cluster table
    if clusters.n_cases:
        t = clusters.as_table(midrule=True, caption=c_caption)
        section.append(t)
