'''Eelbrain GUIs'''

def run():
    "Hand over command to the GUI (quit the GUI to return to the terminal)"
    from . import _wxgui
    _wxgui.run()

def select_epochs(*args, **kwargs):
    """GUI for rejecting trials

    Parameters
    ----------
    ds : Dataset | mne.Epochs
        The data for which to select trials. If ds is an mne.Epochs object
        the subsequent parameters up to 'trigger' are irrelevant.
    data : str
        Name of the epochs data in ds.
    accept : str
        Name of the boolean Var in ds to accept or reject epochs.
    blink : str
        Name of the eye tracker data in ds.
    tag : str
        Name of the rejection tag (storing the reason for rejecting a
        specific epoch).
    trigger : str
        Name of the int Var containing the event trigger for each epoch
        (used to assert that when loading a rejection file it comes from
        the same data).
    path : None | str
        Path to the desired rejection file. If the file already exists it
        is loaded as starting values. The extension determines the format
        (``*.pickled`` or ``*.txt``).
    nplots : None | int | tuple of 2 int
        Number of epoch plots per page. Can be an ``int`` to produce a
        square layout with that many epochs, or an ``(n_rows, n_columns)``
        tuple. The default is to use the settings from the last session.
    topo : None | bool
        Show a topomap plot of the time point under the mouse cursor.
        Default (None): use settings form last session.
    mean : None | bool
        Show a plot of the page mean at the bottom right of the page.
        Default (None): use settings form last session.
    vlim : None | scalar
        Limit of the epoch plots on the y-axis. If None, a value is
        determined automatically to show all data.
    plot_range : bool
        In the epoch plots, plot the range of the data (instead of plotting
        all sensor traces). This makes drawing of pages quicker, especially
        for data with many sensors (the default is to use the last setting
        used).
    color : None | matplotlib color
        Color for primary data (default is black).
    lw : scalar
        Linewidth for normal sensor plots.
    mark : None | index for sensor dim
        Sensors to plot as individual traces with a separate color.
    mcolor : matplotlib color
        Color for marked traces.
    mlw : scalar
        Line width for marked sensor plots.
    antialiased : bool
        Perform Antialiasing on epoch plots (associated with a minor speed
        cost).
    pos : None | tuple of 2 int
        Window position on screen. Default (None): use settings form last
        session.
    size : None | tuple of 2 int
        Window size on screen. Default (None): use settings form last
        session.
    allow_interpolation : bool
        Whether to allow interpolating individual channels by epoch (default
        True).

    Notes
    -----
    In addition to the menus, the following keyboard shortcuts are available:

        right-arrow:
            Go to next page.
        left-arrow:
            Go to previous page.
        b:
            Butterfly plot of the currently displayed epoch.
        c:
            Pairwise sensor correlation plot or the current epoch.
        i:
            Schedule the channel nearest to the pointer on the vertical for
            interpolation.
        Shift-i:
            Open dialog to enter channels for interpolation.
        t:
            Topomap plot of the currently displayed time point.
        u:
            Undo.
        shift-u:
            Redo.
    """
    from . import _wxgui
    _wxgui.select_epochs.TerminalInterface(*args, **kwargs)
