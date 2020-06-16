.. currentmodule:: eelbrain

*******
Recipes
*******

.. contents:: Contents
   :local:


^^^^^^^^^^^^^^^^^^^^^
Plots for Publication
^^^^^^^^^^^^^^^^^^^^^

.. seealso::

   * Example on :ref:`exa-customizing-plots`
   * Matplotlib page on `styles <https://matplotlib.org/tutorials/introductory/
     customizing.html>`_

In order to produce multiple plots with consistent style it is useful to set
some :mod:`matplotlib` options globally One way this can be done is by updating
``rcParams`` inside a script, e.g.::

    import matplotlib as mpl

    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.size'] = 8
    # Change all line-widths
    for key in mpl.rcParams:
        if 'linewidth' in key:
            mpl.rcParams[key] *= 0.5
    # The size of saved figures depends on the figure size (w, h) and DPI
    mpl.rcParams['figure.dpi'] = 300
    mpl.rcParams['savefig.dpi'] = 300


Matplotlib's :func:`~matplotlib.pyplot.tight_layout` functionality provides an
easy way for plots to use the available space, and most Eelbrain plots use it
by default. However, when trying to produce multiple plots with identical
scaling it can lead to unwanted discrepancies. Subplot placement can be
specified in absolute scale through the ``margins`` parameter (see
:ref:`general-layout-parameters`).

If a script produces several plots and there is no need to show them on the
screen, they can be kept hidden through the ``show=False`` argument::

    p = plot.UTSStat('uts', 'A', ds=ds, w=5, tight=False, show=False)

When working on more complex figures, it is often desirable to save the legend
separately and combine it in a layout application::

    p = plot.UTSStat('uts', 'A', ds=ds, w=5, tight=False, show=False, legend=False)
    p.save('plot.pdf', transparent=True)
    p.save_legend('legend.pdf', transparent=True)
