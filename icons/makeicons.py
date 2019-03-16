"Combine icons into a Python file"
import os
from wx.tools.img2py import img2py

python_file = "../eelbrain/_wxgui/icons.py"

# individual files
files = {}  # name: file

# folders that are included entirely
folders = [
    "actions",
    "brain",
    "copy",
    "documents",
    "plot",
]

# tango files (in `tango` subfolder)
tango = {
    'actions': [
        'document-open',
        'document-save',
        'document-save-as',
        'edit-clear',
        'edit-redo',
        'edit-undo',
        'go-down',
        'go-first',
        'go-last',
        'go-next',
        'go-previous',
        'go-up',
        'media-seek-forward',
        'media-seek-backward',
        'system-log-out',
        'view-refresh',
    ],
    'apps': [
        'help-browser',
        'utilities-terminal'
    ],
    'mimetypes': ['x-office-presentation'],
    'places': ['start-here'],
    'status': ['image-missing'],
}


for folder in folders:
    for name in os.listdir(folder):
        if name.endswith('.png'):
            path = os.path.join(folder, name)
            iconname = '/'.join((folder, name[:-4]))
            files[iconname] = path

for basename, names in tango.iteritems():
    for name in names:
        path = os.path.join('tango', basename, name + '.png')
        iconname = '/'.join(('tango', basename, name))
        files[iconname] = path


kwargs = dict(compressed=True,
              catalog=True,
              functionCompatible=False,
              functionCompatibile=-1)


img2py('system-icons/eelbrain/eelbrain256.png', python_file, append=False,
       imgName='eelbrain256', icon=False, **kwargs)

img2py('system-icons/eelbrain/eelbrain32.png', python_file, append=True,
       imgName='eelbrain', icon=True, **kwargs)

for name, image_file in files.iteritems():
    img2py(image_file, python_file, append=True,
           imgName=name, icon=False, **kwargs)

print("Done")
