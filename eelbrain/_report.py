# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import numpy as np

from . import fmtxt
from . import plot
from . import test
from ._data_obj import cellname
from .fmtxt import ms, Section


def enumeration(items, link='and'):
    "['a', 'b', 'c'] -> 'a, b and c'"
    if len(items) >= 2:
        return (' %s ' % link).join((', '.join(items[:-1]), items[-1]))
    elif len(items) == 1:
        return items[0]
    else:
        raise ValueError("items=%s" % repr(items))


def named_list(items, name='item'):
    "named_list([1, 2, 3], 'number') -> 'numbers (1, 2, 3)"
    if len(items) == 1:
        return "%s (%r)" % (name, items[0])
    else:
        return "%ss (%s)" % (name, ', '.join(map(repr, items)))


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
    return '%s - %s ms' % (tstart(res.tstart, uts), tstop(res.tstop, uts))


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


def sensor_results(res, ds, color):
    report = Section("Results")
    if res._kind == 'cluster':
        p = plot.Topomap(res, show=False)
        report.add_figure("Significant clusters.", p)
        p.close()

        report.add_figure("All clusters.", res.clusters)
    else:
        raise NotImplementedError("Result kind %r" % res._kind)
    return report


def sensor_time_results(res, ds, colors, include=1):
    y = ds.eval(res.Y)
    if res._kind in ('raw', 'tfce'):
        report = Section("Results")
        section = report.add_section("P<=.05")
        sensor_bin_table(section, res, 0.05)
        clusters = res.find_clusters(0.05, maps=True)
        clusters.sort('tstart')
        for cluster in clusters.itercases():
            sensor_time_cluster(section, cluster, y, res._plot_model(), ds,
                                colors, res.match)

        # trend section
        section = report.add_section("Trend: p<=.1")
        sensor_bin_table(section, res, 0.1)

        # not quite there section
        section = report.add_section("Anything: P<=.2")
        sensor_bin_table(section, res, 0.2)
    elif res._kind == 'cluster':
        report = Section("Clusters")
        sensor_bin_table(report, res)
        clusters = res.find_clusters(include, maps=True)
        clusters.sort('tstart')
        for cluster in clusters.itercases():
            sensor_time_cluster(report, cluster, y, res._plot_model(), ds,
                                colors, res.match)
    else:
        raise NotImplementedError("Result kind %r" % res._kind)
    return report


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


def sensor_time_cluster(section, cluster, y, model, ds, colors, match='subject'):
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
    if model:
        x = ds.eval(model)
        topos = [[topo[x == cell].summary('case', name=cellname(cell)),
                  cluster_topo] for cell in x.cells]
    else:
        topos = [[topo, cluster_topo]]
    p = plot.Topomap(topos, axh=3, nrow=1, show=False)
    p.mark_sensors(np.flatnonzero(cluster_topo.x), c='y', marker='o')

    caption_ = ["Cluster"]
    if 'effect' in cluster:
        caption_.extend(('effect of', cluster['effect']))
    caption_.append("%i - %i ms." % (tstart_ms, tstop_ms))
    caption = ' '.join(caption_)
    section.add_image_figure(p, caption)
    p.close()

    cluster_timecourse(section, cluster, y, 'sensor', model, ds, colors, match)


def source_bin_table(section, res, surfer_kwargs, pmin=None):
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
        im = plot.brain.bin_table(ndvar, **surfer_kwargs)
        section.add_image_figure(im, caption_)


def source_time_clusters(section, clusters, y, ds, model, include, title, colors):
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
        source_time_cluster(section, cluster, y, model, ds, title, colors)


def source_time_cluster(section, cluster, y, model, ds, title, colors):
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
        txt.append(fmtxt.eq('p', cluster['p_parc'], 'mcc', '%s', drop0=True))
        txt.append('.')

    # add cluster image to report
    brain = plot.brain.cluster(cluster['cluster'].sum('time'),
                               surf='inflated')
    caption_ = ["Cluster"]
    if 'effect' in cluster:
        caption_.extend(('effect of', cluster['effect']))
    caption_.append("%i - %i ms." % (tstart_ms, tstop_ms))
    caption = ' '.join(caption_)
    section.add_image_figure(brain.image('cluster_spatial'), caption)
    cluster_timecourse(section, cluster, y, 'source', model, ds, colors)


def cluster_timecourse(section, cluster, y, dim, model, ds, colors,
                       match='subject'):
    c_extent = cluster['cluster']
    cid = cluster['id']

    # cluster time course
    idx = c_extent.any('time')
    tc = y[idx].mean(dim)
    p = plot.UTSStat(tc, model, match=match, ds=ds, legend=None, h=4,
                     colors=colors, show=False)
    # mark original cluster
    for ax in p._axes:
        ax.axvspan(cluster['tstart'], cluster['tstop'], color='r',
                   alpha=0.2, zorder=-2)

    if model:
        # legend
        legend_p = p.plot_legend(show=False)
        legend = legend_p.image("Legend")
        legend_p.close()
    else:
        p._axes[0].axhline(0, color='k')
    image_tc = p.image('cluster_%i_timecourse' % cid)
    p.close()

    # Barplot
    idx = (c_extent != 0)
    v = y.mean(idx)
    p = plot.Barplot(v, model, match, ds=ds, corr=None, colors=colors, h=4,
                     show=False)
    image_bar = p.image('cluster_%i_barplot.png' % cid)
    p.close()

    # Boxplot
    p = plot.Boxplot(v, model, match, ds=ds, corr=None, colors=colors, h=4,
                     show=False)
    image_box = p.image('cluster_%i_boxplot.png' % cid)
    p.close()

    if model:
        # compose figure
        section.add_figure("Time course in cluster area, and average value in "
                           "cluster by condition, with pairwise t-tests.",
                           [image_tc, image_bar, image_box, legend])
        # pairwise test table
        res = test.pairwise(v, model, match, ds=ds, corr=None)
        section.add_figure("Pairwise t-tests of average value in cluster by "
                           "condition", res)
    else:
        section.add_figure("Time course in cluster area, and average value in "
                           "cluster.", [image_tc, image_bar, image_box])


def roi_timecourse(doc, ds, label, res, colors):
    "Plot ROI time course with cluster permutation test"
    y = ds.info['label_keys'][label]
    label_name = label[:-3].capitalize()
    hemi = label[-2].capitalize()
    title = ' '.join((label_name, hemi))
    caption = "Source estimates in %s (%s)." % (label_name, hemi)
    doc.append(time_results(res, ds, colors, title, caption))


def time_results(res, ds, colors, title='Results', caption="Timecourse",
                 pairwise_pmax=0.1):
    """Add time course with clusters

    Parameters
    ----------
    res : Result
        Result of the temporal cluster test.
    ds : Dataset
        Data.
    colors : dict
        Cell colors.
    title : str
        Section title.
    """
    clusters = res.find_clusters()
    if clusters.n_cases:
        idx = clusters.eval("p.argmin()")
        max_sig = clusters['sig'][idx]
        if max_sig:
            title += max_sig
    section = Section(title)

    # compose captions
    if clusters.n_cases:
        c_caption = ("Clusters in time window %s based on %s."
                     % (format_timewindow(res), format_samples(res)))
        tc_caption = caption
    else:
        c_caption = "No clusters found %s." % format_timewindow(res)
        tc_caption = ' '.join((caption, c_caption))

    # plotting arguments
    model = res._plot_model()
    sub = res._plot_sub()

    # add UTSStat plot
    p = plot.UTSStat(res.Y, model, None, res.match, sub, ds, colors=colors,
                     legend=None, clusters=clusters, show=False)
    ax = p._axes[0]
    if res.tstart is not None:
        ax.axvline(res.tstart, color='k')
    if res.tstop is not None:
        ax.axvline(res.tstop, color='k')
    image = p.image('%s_cluster.png')
    legend_p = p.plot_legend(show=False)
    legend = legend_p.image("Legend")
    section.add_figure(tc_caption, [image, legend])
    p.close()
    legend_p.close()

    # add cluster table
    if clusters.n_cases:
        t = clusters.as_table(midrule=True, caption=c_caption)
        section.append(t)

    # pairwise plots
    model_ = model
    colors_ = colors
    if pairwise_pmax is not None:
        plots = []
        clusters_ = clusters.sub("p <= %s" % pairwise_pmax)
        clusters_.sort("tstart")
        for cluster in clusters_.itercases():
            cid = cluster['id']
            c_tstart = cluster['tstart']
            c_tstop = cluster['tstop']
            tw_str = "%s - %s ms" % (ms(c_tstart), ms(c_tstop))
            if 'effect' in cluster:
                title = "%s %s%s: %s" % (cluster['effect'], cid, cluster['sig'], tw_str)
                model_ = cluster['effect'].replace(' x ', '%')
                colors_ = colors if model_ == model else None
            else:
                title = "Cluster %s%s: %s" % (cid, cluster['sig'], tw_str)
            y_ = ds[res.Y].summary(time=(c_tstart, c_tstop))
            p = plot.Barplot(y_, model_, res.match, sub, ds=ds, corr=None,
                             show=False, colors=colors_, title=title)
            plots.append(p.image())
            p.close()

        section.add_image_figure(plots, "Value in the time-window of the clusters "
                                 "with uncorrected pairwise t-tests.")

    return section


def result_report(res, ds, title=None, colors=None):
    """Automatically generate section from testnd Result

    Parameters
    ----------
    res : Result
        Test-result.
    ds : Dataset
        Dataset containing the data on which the test was performed.
    """
    sec = Section(title or res._name())

    dims = {dim.name for dim in res._dims}
    sec.append(res.info_list())

    if dims == {'time'}:
        sec.append(time_results(res, ds, colors))
    elif dims == {'sensor'}:
        sec.append(sensor_results(res, ds, colors))
    elif dims == {'time', 'sensor'}:
        sec.append(sensor_time_results(res, ds, colors))
    else:
        raise NotImplementedError("dims=%r" % dims)
    return sec
