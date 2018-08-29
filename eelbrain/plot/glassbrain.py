# Author: Proloy Das <proloy@umd.edu>

# fix matplotlib backend for osx
from sys import platform
if platform == "darwin":
    import matplotlib
    gui_env = ['WX', 'TkAgg', 'Qt4Agg', 'Qt5Agg', 'WXAgg']
    for gui in gui_env:
        try:
            print("testing %s" %gui)
            matplotlib.use(gui, warn=False, force=True)
            break
        except:
            continue
    print("Using: %s" %matplotlib.get_backend())

from ._glassbrain import (GlassBrain, butterfly)
