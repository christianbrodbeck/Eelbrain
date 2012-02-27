"""
Startup script for wxterm. 

"""
import matplotlib as mpl
mpl.use('WXAgg')

import eelbrain.wxterm.app as _app_module



_app = _app_module.MainApp(globals())
_app.MainLoop()