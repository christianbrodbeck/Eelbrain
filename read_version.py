# read version from __init__ without importing it
# stored in the `version` variable
import inspect
from os.path import join, dirname
import re

path = join(dirname(inspect.getfile(inspect.currentframe())),
            'eelbrain', '__init__.py')
with open(path) as fid:
    text = fid.read()
match = re.search("__version__ = '([.\w]+)'", text)
if match is None:
    raise ValueError("No valid version string found in:\n\n" + text)
version = match.group(1)
if version.count('.') != 2 and not version.endswith('dev'):
    raise ValueError("Invalid version string extracted: %r" % version)
