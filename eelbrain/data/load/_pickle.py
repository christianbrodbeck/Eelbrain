from cPickle import Unpickler
from importlib import import_module

from ... import ui


def map_paths(self, module_name, class_name):
    "Subclass to handle changes in module paths"
    if module_name == 'eelbrain.vessels.data':
        module_name = 'eelbrain.data'
        class_names = {'var':'Var', 'factor':'Factor', 'ndvar':'NDVar',
                       'datalist':'Datalist', 'dataset':'Dataset'}
        class_name = class_names[class_name]

    module = import_module(module_name)
    return getattr(module, class_name)


def unpickle(file_path=None):
    """Load pickled Python objects from a file.

    Almost like ``cPickle.load(open(file_path))``, but also loads object saved
    with older versions of Eelbrain, and allows using a system file dialog to
    select a file.

    Parameters
    ----------
    file_path : None | str
        Path to a pickled file. If None, a system file dialog will be used. If
        the user cancels the file dialog, a RuntimeError is raised.
    """
    if file_path is None:
        filetypes = [("Pickles (*.pickled)", '*.pickled'), ("All files", '*')]
        file_path = ui.ask_file("Select File to Unpickle", "Select a pickled "
                                "file to unpickle", filetypes)
    if file_path is False:
        raise RuntimeError("User canceled")

    with open(file_path, 'r') as fid:
        unpickler = Unpickler(fid)
        unpickler.find_global = map_paths
        obj = unpickler.load()

    return obj
