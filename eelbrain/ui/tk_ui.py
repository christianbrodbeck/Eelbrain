'''
Created on Jun 17, 2012


http://tkinter.unpythonic.net/wiki/tkFileDialog


@author: Christian M Brodbeck
'''
import logging
import tkFileDialog
from tkFileDialog import Open, SaveAs
import tkMessageBox
from Tkinter import Tk



def ask_saveas(title, message, filetypes, defaultDir=None, defaultFile=None):
    dlg = SaveAs(title=title, message=message, filetypes=filetypes)
    filename = dlg.show()
    return filename


def ask_dir(title="Select Folder",
            message="Please Pick a Folder",
            must_exist=True):
    return tkFileDialog.askdirectory(title=title, mustexist=must_exist)


def ask_file(title, message, filetypes, directory, mult):
    dlg = Open(title=title, filetypes=filetypes, initialdir=directory,
               multiple=mult)
    out = dlg.show()
    return out


def ask(title="Overwrite File?",
        message="Duplicate filename. Do you want to overwrite?",
        cancel=False,
        default=True,  # True=YES, False=NO, None=Nothing
        ):
    return tkMessageBox.askyesno(title, message)


def ask_str(message, title, default=''):
    return Tk.tkSimpleDialog.askstring(title, message, initialvalue=default)


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


class progress_monitor:
    def __init__(self, i_max=None,
                 title="Task Progress",
                 message="Wait and pray!",
                 cancel=True):
        self.i = -1
        self.i_max = i_max
        self.title = title
        self._message = message
        self.advance(message)

    def _log(self):
        msg = ("Progress %r: %s of %s; %r" % (self.title, self.i, self.i_max,
                                              self._message))
        logging.info(msg)

    def advance(self, new_msg=None):
        self.i += 1
        if new_msg:
            self._message = new_msg

        self._log()

    def message(self, new_msg):
        self._message = new_msg
        self._log()

    def terminate(self):
        pass


def show_help(obj):
    print getattr(obj, '__doc__', 'no docstring')
