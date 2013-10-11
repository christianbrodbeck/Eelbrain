# py2app startup script

import os
import eelbrain.wxterm

os.chdir(os.path.expanduser('~'))
eelbrain.wxterm.launch(True)
