# py2app startup script

import os
import eelbrain._wxterm

os.chdir(os.path.expanduser('~'))
eelbrain._wxterm.launch(True)
