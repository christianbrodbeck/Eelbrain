"""
Startup script for wxterm. 

"""
import matplotlib as mpl
mpl.use('WXAgg')

# configure the logging module so it logs debug messages
# On OS X they can be viewed in the Console
import logging
logging.basicConfig(level=logging.DEBUG)

import eelbrain.wxterm.app as _app_module



_app = _app_module.MainApp(globals())
_app.MainLoop()