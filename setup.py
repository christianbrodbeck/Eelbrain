"""
setyup.py
=========

This is the setup.py script for Eelbrain.

To create the Alias build on Mac Os X run::

    $ python setup.py py2app -A

After that, Eelbrain.app can be found in the dist/ folder. The application
can be moved to another location without breaking the links, but will always
reflect changes to the source code in eelbrain/, which consequently has to
stay in the same location.

http://packages.python.org/py2app

http://docs.python.org/distutils/index.html

"""
from ez_setup import use_setuptools
use_setuptools()

import re
import sys
from setuptools import setup, find_packages, Extension
import numpy as np

# version must be in X.X.X format, e.g., "0.0.3dev"
with open('eelbrain/__init__.py') as fid:
    text = fid.read()
match = re.search("__version__ = '([.\w]+)'", text)
if match is None:
    raise ValueError("No valid version string found in:\n\n" + text)
version = match.group(1)
if version.count('.') != 2 and not version.endswith('dev'):
    raise ValueError("Invalid version string extracted: %r" % version)

if len(sys.argv) > 1:
    arg = sys.argv[1]
else:
    arg = None

# Cython extensions
ext = [Extension("eelbrain._stats.opt", ["eelbrain/_stats/opt.c"])]

# basic setup arguments
kwargs = dict(
              name='eelbrain',
              version=version,
              description="MEG/EEG analysis tools",
              url="https://pythonhosted.org/eelbrain",
              author="Christian Brodbeck",
              author_email='christianbrodbeck@nyu.edu',
              license='GPL3',
              long_description=open('README.txt').read(),
              install_requires=['tex',
                                'matplotlib >= 1.1',
                                'scipy >= 0.11.0',
                                'numpy >= 1.8',
                                'docutils',
                                'mne >= 0.8'],
              extras_require={'plot.brain': ['nibabel', 'pysurfer']},
              include_dirs=[np.get_include()],
              ext_modules=ext,
              )

# py2app -----------------------------------------------------------------------
py2app_alias_script = """
# Startup script for py2app.
import os
import site
import sys

# extend sys.path
paths = %s
for path in paths:
    if path not in sys.path:
        site.addsitedir(path)


# launch Eelbrain
import eelbrain.wxterm

os.chdir(os.path.expanduser('~'))
eelbrain.wxterm.launch(True)
""" % str(sys.path)

if arg == 'py2app':
    doctypes = [
                {"CFBundleTypeExtensions": ["py"],
                 "CFBundleTypeName": "Python Script",
                 "CFBundleTypeRole": "Editor",
                 "CFBundleTypeIconFile": "icons/system-icons/pydoc.icns",
                 },
                {"CFBundleTypeExtensions": ["pickled"],
                 "CFBundleTypeName": "Pickled Python Object",
                 "CFBundleTypeRole": "Editor",
                 "CFBundleTypeIconFile": "icons/system-icons/pickled.icns",
                 },
                ]

    OPTIONS = {
               'argv_emulation': True,
               'site_packages': True,
               'iconfile': 'icons/system-icons/eelbrain.icns',
               # py2app's iconfile option places the specified file
               # in the new app bundle's Resources directory
#               'packages': 'wx',
#               'resources': ['icons/system-icons/eelbrain.icns',
#                          'resources/License.txt'
#                          ],
               'plist': dict(CFBundleName="Eelbrain",
                             CFBundleShortVersionString=version,
                             CFBundleGetInfoString="Eelbrain " + version,
                             CFBundleExecutable="Eelbrain",
                             CFBundleIdentifier="com.christianmbrodbeck.Eelbrain",
                             CFBundleDocumentTypes=doctypes,
#                             CFBundleIconFile = 'eelbrain.icns',
                             ),
               }

    # create the startup script
    if '-A' in sys.argv:
        script_path = 'scripts/eelbrain_alias.py'
        with open(script_path, 'w') as fid:
            fid.write(py2app_alias_script)
    else:
        script_path = 'scripts/eelbrain_py2app.py'

    kwargs.update(
                  app=[script_path],
                  options={'py2app': OPTIONS},
                  setup_requires=['py2app'])
# cx_freeze??? ---
# elif arg =='build':
#    from cx_Freeze import setup, Executable
#    kwargs.update(executables = [Executable("scripts/eelbrain")])
else:
    # normal & py2exe ---
    kwargs['packages'] = find_packages()
    # py2exe ---
    if arg == 'py2exe':
        # http://wiki.wxpython.org/DistributingYourApplication
        import py2exe, matplotlib
        data_files = matplotlib.get_py2exe_datafiles()

        # DLL files that various Python modules depend on
        data_files.append(('.', [
#                                 "C:\Python27\DLLs\msvcp90.dll",
#                                 "C:\Python27\lib\site-packages\wx\gdiplus.dll",
                                 "C:\python27\scripts\MK2_CORE.DLL",
                                 "C:\python27\scripts\MK2_P4P.DLL",
                                 "C:\python27\scripts\MK2IOMP5MD.DLL",
                                  ]))


        OPTIONS = dict(
#                       skip_archive = True,
#                       bundle_files = 2, # This tells py2exe to bundle everything
                       excludes=['_cairo', '_cocoaagg', '_emf', '_fltkagg',
                                 '_gtkagg', '_qt4agg', '_tkagg'],
                       includes=['matplotlib.numerix.random_array',
                                 'scipy.io.matlab.streams'],
                       dll_excludes=['libgdk-win32-2.0-0.dll',
                                     'libgobject-2.0-0.dll'])
        win_icon = "icons/system-icons/eelbrain.ico"
        kwargs.update(windows=[{"script": 'scripts/eelbrain',
                                   "icon_resources": [(0, win_icon)],
                                   }],
                      data_files=data_files,
                      options={'py2exe': OPTIONS},
                      # com_server=['myserver'],
                      )

    # normal -----------------------------------------------------------------------
    else:
        kwargs['scripts'] = ['scripts/eelbrain']


setup(**kwargs)

