
def launch(py2app=False):
    "launches Eelbrain's wxPython terminal"
    import wx
    if wx.GetApp():
        raise RuntimeError("wx.App already running")
        # prevent irregular behavior if called from Eelbrain or pyshell

    import matplotlib as mpl
    mpl.use('WXAgg')

    # configure the logging module so it logs debug messages
    # On OS X they can be viewed in the Console
    import logging
    logging.basicConfig(level=logging.DEBUG)

    from .app import MainApp

    app = MainApp(py2app)
    app.MainLoop()
