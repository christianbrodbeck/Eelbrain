# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from pathlib import Path
from pickle import dump, HIGHEST_PROTOCOL, Unpickler
from itertools import chain
import os

from .._data_obj import Dataset, NDVar, Var, SourceSpaceBase, ismodelobject
from .._types import PathArg
from .._utils import tqdm, ui


class EelUnpickler(Unpickler):

    def find_class(self, module, name):
        "Backwards-compatibility for changes in module paths"
        if module.startswith('eelbrain.'):
            if module == 'eelbrain.vessels.data':
                module = 'eelbrain._data_obj'
                class_names = {'var': 'Var', 'factor': 'Factor', 'ndvar': 'NDVar',
                               'datalist': 'Datalist', 'dataset': 'Dataset'}
                name = class_names[name]
            elif module.startswith('eelbrain.data.'):
                if module.startswith('eelbrain.data.load'):
                    module = module.replace('.data.load', '.load')
                elif module.startswith('eelbrain.data.stats'):
                    module = module.replace('.data.stats', '._stats')
                elif module.startswith('eelbrain.data.data_obj'):
                    module = module.replace('.data.data_obj', '._data_obj')
                else:
                    raise NotImplementedError("%r / %r" % (module, name))

        return Unpickler.find_class(self, module, name)


def pickle(obj, dest: PathArg = None, protocol: int = HIGHEST_PROTOCOL):
    """Pickle a Python object.

    Parameters
    ----------
    obj : object
        Python object to save.
    dest : Path
        Path to destination where to save the file. If no destination is
        provided, a file dialog is shown. If a destination without extension is
        provided, ``.pickle`` is appended.
    protocol : int
        Pickle protocol (default is ``HIGHEST_PROTOCOL``). For pickles that can
        be opened in Python 2, use ``protocol<=2``.
    """
    if dest is None:
        filetypes = [("Pickled Python Objects (*.pickle)", '*.pickle')]
        dest = ui.ask_saveas("Pickle Destination", "", filetypes)
        if dest is False:
            raise RuntimeError("User canceled")
        else:
            print(f'dest={dest!r}')
    else:
        dest = Path(dest).expanduser()
        if not dest.suffix:
            dest = dest.with_suffix('.pickle')

    try:
        with open(dest, 'wb') as fid:
            dump(obj, fid, protocol)
    except SystemError as exception:
        if exception.args[0] == 'error return without exception set':
            if os.path.exists(dest):
                os.remove(dest)
            raise IOError("An error occurred while pickling. This could be due to an attempt to pickle an array (or NDVar) that is too big. Try saving several smaller arrays.")
        else:
            raise


def unpickle(path: PathArg = None):
    """Load pickled Python objects from a file.

    Almost like ``pickle.load(open(path))``, but also loads object saved
    with older versions of Eelbrain, and allows using a system file dialog to
    select a file.

    Parameters
    ----------
    path : Path
        Path to a pickled file. If omitted, a system file dialog is shown.
        If the user cancels the file dialog, a RuntimeError is raised.

    Notes
    -----
    If you see ``ValueError: unsupported pickle protocol: 4``, the pickle file
    was saved with Python 3; in order to make pickles backwards-compatible, use
    :func:`~eelbrain.save.pickle` with ``protocol=2``.
    """
    if path is None:
        filetypes = [("Pickles (*.pickle|*.pickled)", '*.pickle?'), ("All files", '*')]
        path = ui.ask_file("Select File to Unpickle", "Select a file to unpickle", filetypes)
        if path is False:
            raise RuntimeError("User canceled")
        else:
            print(f"unpickle {path}")
    else:
        path = Path(path).expanduser()
        if not path.exists():
            for ext in ('.pickle', '.pickled'):
                new_path = path.with_suffix(ext)
                if new_path.exists():
                    path = new_path
                    break

    with open(path, 'rb') as fid:
        unpickler = EelUnpickler(fid, encoding='latin1')
        try:
            return unpickler.load()
        except EOFError:
            raise EOFError(f"Corrupted file, writing may have been interrupted: {path}")


def update_subjects_dir(obj, subjects_dir, depth=0):
    """Update NDVar SourceSpace.subjects_dir attributes

    Examine elements of ``obj`` recursively and replace ``subjects_dir`` on all
    NDVars with SourceSpace dimension that are found.

    Parameters
    ----------
    obj : object
        Object to examine.
    subjects_dir : str
        New values for subjects_dir.
    depth : int
        Recursion depth for examining attributes (default 0, i.e. only apply to
        ``obj`` without recursion). Negative number for exhaustive search.

    Notes
    -----
    The following elements are searched:

      - Attributes of objects that have a ``__dict__``.
      - ``dict`` values.
      - list/tuple items.
    """
    if isinstance(obj, SourceSpaceBase):
        if obj.subjects_dir != subjects_dir:
            obj.subjects_dir = subjects_dir
            if obj._subjects_dir is not None:
                obj._subjects_dir = subjects_dir
    elif isinstance(obj, NDVar):
        for dim in obj.dims:
            if isinstance(dim, SourceSpaceBase):
                if dim.subjects_dir == subjects_dir:
                    break
                dim.subjects_dir = subjects_dir
                if dim._subjects_dir is not None:
                    dim._subjects_dir = subjects_dir
        else:
            for v in obj.info.values():
                update_subjects_dir(v, subjects_dir, depth)
    elif depth:
        if isinstance(obj, Var):
            values = obj.info.values()
        elif isinstance(obj, Dataset):
            values = chain(obj.info, obj.values())
        elif ismodelobject(obj):
            return
        else:
            if hasattr(obj, '__dict__'):
                values = obj.__dict__.values()
            else:
                values = ()

            if isinstance(obj, dict):
                values = chain(values, obj.values())
            elif isinstance(obj, (tuple, list)):
                values = chain(values, obj)

        for v in values:
            update_subjects_dir(v, subjects_dir, depth - 1)


def convert_pickle_protocol(
        root: PathArg = None,
        to_protocol: int = 4,
        pattern: str = '**/*.pickle',
):
    """Re-save all pickle files with a specific protocol

    Parameters
    ----------
    root
        Root directory to look for pickle files.
    to_protocol
        Protocol to re-save with.
    pattern
        Filename pattern used to find pickle files.

    Notes
    -----
    Python 3.8 introduced a new pickle protocol 5, which Python 3.7 can's read.
    This function converts such files in order to make them readable with older
    version of Python.
    """
    if root is None:
        root = ui.ask_dir("Select folder", "Select folder to search for pickle files")
        if root is False:
            raise RuntimeError("User canceled")
        else:
            print(f"Searching {root}")
    root = Path(root)
    for path in tqdm(root.glob(pattern), "Converted", unit=' files'):
        obj = unpickle(path)
        pickle(obj, path, to_protocol)
