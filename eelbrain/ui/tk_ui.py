'''
Created on Jun 17, 2012


http://tkinter.unpythonic.net/wiki/tkFileDialog


@author: Christian M Brodbeck
'''
import tkFileDialog
import tkMessageBox
from Tkinter import Tk



def ask_saveas(title, message, ext):
    return tkFileDialog.asksaveasfile(title=title, message=message)


def ask_dir(title = "Select Folder",
            message = "Please Pick a Folder",
            must_exist = True):
    return tkFileDialog.askdirectory(title=title, mustexist=must_exist)


def ask(title = "Overwrite File?",
        message = "Duplicate filename. Do you want to overwrite?",
        cancel=False,
        default=True, # True=YES, False=NO, None=Nothing
        ):
    return tkMessageBox.askyesno(title, message)


def copy_file(path):
    raise NotImplementedError


def copy_text(text):
    # http://stackoverflow.com/a/4203897/166700
    r = Tk()
    r.withdraw()
    r.clipboard_clear()
    r.clipboard_append(text)
    r.destroy()


def message(title, message="", icon='i'):
    if icon in 'i?':
        tkMessageBox.showinfo(title, message)
    elif icon == '!':
        tkMessageBox.showwarning(title, message)
    elif icon == 'error':
        tkMessageBox.showerror(title, message)
    else:
        raise ValueError("Invalid icon argument: %r" % icon)


def progress(*args, **kwargs):
    pass


def show_help(obj):
    print getattr(obj, '__doc__', 'no docstring')
