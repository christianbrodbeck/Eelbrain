"""
Module to deploy GUI elements depending on the current environment.

"""

import logging
import os

try:
    import wx
    import wx_ui 
except ImportError:
    logging.info("wx unavailable; using tk_ui")
    wx_ui = False


__all__ = ['ask', 'ask_color', 'ask_dir', 'ask_saveas',
           'copy_file'
           'message', 
           'progress', 
           'test_targetpath',
           ]

def get_ui():
    if wx_ui and bool(wx.GetApp()):
        return wx_ui
    else:
        import tk_ui
        return tk_ui


def ask_saveas(title = "Save File",
               message = "Please Pick a File Name", 
               ext = [('pickled', "pickled Python object")]):
    return get_ui().ask_saveas(title, message, ext)



def ask_dir(title = "Select Folder",
            message = "Please Pick a Folder",
            must_exist = True):
    return get_ui().ask_dir(title, message, must_exist)



def ask_file(title = "Pick File",
             message = "Please Pick a File", 
             ext = [('*', "all files")],
             directory='',
             mult=False):
    return get_ui().ask_file(title, message, ext, directory, mult)



def ask(title = "Overwrite File?",
        message = "Duplicate filename. Do you want to overwrite?",
        cancel=False,
        default=True, # True=YES, False=NO, None=Nothing
        ):
    return get_ui().ask(title, message, cancel, default)



def ask_color(default=(0,0,0)):
    return get_ui().ask_color(default)


def show_help(obj):
    return get_ui().show_help(obj)


def message(title, message="", icon='i'):
    return get_ui().message(title, message, icon)


def progress(*args, **kwargs):
    return get_ui().progress(*args, **kwargs)


def copy_file(path):
    return get_ui().copy_file(path)


def copy_text(text):
    return get_ui().copy_text(text)



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
