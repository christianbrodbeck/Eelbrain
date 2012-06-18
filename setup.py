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

import sys

#VERSION must be in X.X.X format, e.g., "0.0.3dev"
from eelbrain import __version__ as VERSION

if len(sys.argv) > 1:
    arg = sys.argv[1]
else:
    print ("For more specific instructions, see "
           "http://christianmbrodbeck.github.com/Eelbrain/")
    arg = None


# clear build and dist directories
#print "removing build and dist directories"
#for dir in ['dist', 'build']:
#    print os.path.abspath(dir)
#    if os.path.exists(dir):
#        shutil.rmtree(dir)


kwargs = dict(
              name = 'eelbrain',
              version = VERSION,
              description = "Eelbrain",
              url = 'http://christianmbrodbeck.github.com/Eelbrain/',
              author = "Christian M. Brodbeck",
              author_email = 'christianmbrodbeck@gmail.com',
              license = 'GPL3',
              long_description = open('README.txt').read(),
              )


if arg == 'py2app':  #####  #####  #####  #####  #####  #####  #####  #####  #####  #####
    from setuptools import setup
    
    doctypes = [
                {"CFBundleTypeExtensions": ["py"],
                 "CFBundleTypeName": "Python Script",
                 "CFBundleTypeRole": "Editor",
#                 "CFBundleTypeIconFile": "../icons/system-icons/pydoc.icns",
                 },
                ]
    
    OPTIONS = {
               'iconfile': 'icon/eelbrain.icns',
               # py2app's iconfile option places the specified file
               # in the new app bundle's Resources directory
#               'packages': 'wx',
#               'resources': ['icons/system-icons/eelbrain.icns',
#                          'resources/License.txt'
#                          ],
               'plist': dict(CFBundleName = "Eelbrain",
                             CFBundleShortVersionString = VERSION,
                             CFBundleGetInfoString = "Eelbrain "+VERSION,
                             CFBundleExecutable = "Eelbrain",
                             CFBundleIdentifier = "com.christianmbrodbeck.Eelbrain",
                             CFBundleDocumentTypes = doctypes,
#                             CFBundleIconFile = 'eelbrain.icns',
                             ),
               'argv_emulation': True
               }

    kwargs.update(
#                  app=['run_eelbrain_py2app.py'],
                  app=['eelbrain.py'],
#                  data_files=[('.', ['icons/system-icons/eelbrain.icns'])],
#                  app=['scripts/eelbrain'], # tries relative import of eelbrain
                  options = {'py2app': OPTIONS},
                  setup_requires=['py2app'])
#elif arg =='build':  #####  #####  #####  #####  #####  #####  #####  #####  #####  #####  
#    from cx_Freeze import setup, Executable
#    kwargs.update(executables = [Executable("scripts/eelbrain")])
else: #####  py2exe & normal  #####  #####  #####  #####  #####  #####  #####  #####  #####
    kwargs.update( # FIXME:
                  packages = ['eelbrain',
                              'eelbrain.analyze', 
                              'eelbrain.fmtxt', 
                              'eelbrain.load', 
                              'eelbrain.plot', 
                              'eelbrain.psyphys', 
                              'eelbrain.psyphys.fileio', 
                              'eelbrain.save', 
                              'eelbrain.ui', 
                              'eelbrain.utils',
                              'eelbrain.vessels',
                              'eelbrain.wxgui',
                              'eelbrain.wxgui.psyphys',
                              'eelbrain.wxterm',
                              'eelbrain.wxutils',
                              ],
                  )
    
    if arg == 'py2exe':  #####  #####  #####  #####  #####  #####  #####  #####  #####
        from distutils.core import setup
        import py2exe, matplotlib
        data_files = matplotlib.get_py2exe_datafiles()
        
        # DLL files that various Python modules depend on
        data_files.append(('.', [#"C:\Python27\DLLs\msvcp90.dll",
                                 #"C:\Python27\lib\site-packages\wx\gdiplus.dll",
                                 "C:\python27\scripts\MK2_CORE.DLL",
                                 "C:\python27\scripts\MK2_P4P.DLL",
                                 "C:\python27\scripts\MK2IOMP5MD.DLL",
                                  ]))
        
        
        OPTIONS = dict(#skip_archive = True,
                       #bundle_files = 2, # This tells py2exe to bundle everything
                       excludes = ['_cairo', '_cocoaagg', '_emf', '_fltkagg', 
                                   '_gtkagg', '_qt4agg', '_tkagg'], #'
                       includes = ['matplotlib.numerix.random_array',
                                    'scipy.io.matlab.streams'],
                       dll_excludes = ['libgdk-win32-2.0-0.dll',
                                        'libgobject-2.0-0.dll'])
        win_icon = "icons/system-icons/eelbrain.ico"
        kwargs.update(windows = [{"script": 'scripts/eelbrain',
                                   "icon_resources": [(0, win_icon)],
                                   }],
                      data_files = data_files,
                      options = {'py2exe': OPTIONS},
                      #com_server=['myserver'],
                      )
    
    else: # NORMAL sdist  #####  #####  #####  #####  #####  #####  #####  #####  #####
#        from distutils2.core import setup
        from distutils.core import setup
        kwargs.update(scripts = ['eelbrain.py'],
                      requires = ['numpy', 'scipy', 'matplotlib', 'wxPython'],
#                      ['tex', 'bioread'], # optional
                      )


setup(**kwargs)

