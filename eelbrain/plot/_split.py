# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Plot data split for cross-validation"""
from typing import Union

from matplotlib.patches import Rectangle

from .._trf.shared import Splits
from ._base import EelFigure, Layout, LegendMixin
from ._styles import colors_for_oneway


class DataSplit(EelFigure, LegendMixin):

    def __init__(
            self,
            splits: Splits,
            legend: Union[str, int, bool] = 'upper right',
            colors: dict = None,
            **kwargs,
    ):
        """Plot data splits

        Parameters
        ----------
        splits
            The data splits.
        legend
            Matplotlib figure legend location argument or 'fig' to plot the
            legend in a separate figure.
        colors
            Customize colors (uses the following keys: ``'train', 'validate',
            'test'``).
        ...
            Also accepts :ref:`general-layout-parameters`.

        Notes
        -----
        This plot can also be generated from a :class:`BoostingResult` object
        through :meth:`BoostingResult.splits.plot` method.
        """
        attrs = ['train', 'validate', 'test']
        labels = {'train': 'Training', 'validate': 'Validation', 'test': 'Testing'}
        if colors is None:
            colors = colors_for_oneway(attrs, unambiguous=[6, 3, 5])

        layout = Layout(1, 16/9, 2, **kwargs)
        EelFigure.__init__(self, None, layout)
        ax = self.figure.axes[0]

        for x in splits.segments[:-1, 1]:
            ax.axvline(x, color='k')
        handles = {}
        for y, split in enumerate(splits.splits):
            for attr in attrs:
                segments = getattr(split, attr)
                if segments is None:
                    continue
                rect = None
                for x0, x1 in segments:
                    rect = Rectangle((x0, y-0.5), x1-x0, 1, color=colors[attr])
                    ax.add_artist(rect)
                handles[attr] = rect
            # separate splits
            if y:
                ax.axhline(y-0.5, color='white')
        labels = {key: labels[key] for key in handles}
        ax.set_ylabel('Split')
        ax.set_ylim(-0.5, len(splits.splits)-0.5)
        ax.set_xlabel('Sample')
        ax.set_xlim(splits.segments[0, 0], splits.segments[-1, 1])
        LegendMixin.__init__(self, legend, handles, labels)
        self._show()
