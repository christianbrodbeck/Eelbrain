import logging as _logging

try:
    import wx
    from wx_ui import *
except ImportError:
    _logging.warning("wx unavailable; using shell ui")
    from terminal_ui import *

__all__ = ['ask', 'ask_color', 'ask_dir', 'ask_saveas',
           'copy_file'
           'message', 
           'progress', 
           'test_targetpath',
           ]


def test_targetpath(path):
    """
    Returns True if path is a valid path to write to, False otherwise. If the
    directory does not exist, the user is asked whether it should be created.
    
    """
    if not path:
        return False
    
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        msg = ("The directory %r does not exist. Should it be created?" % dirname)
        if ask("Create Directory?", msg):
            os.makedirs(dirname)
    
    return os.path.exists(dirname)
