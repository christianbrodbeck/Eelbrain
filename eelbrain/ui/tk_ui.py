'''
Created on Jun 17, 2012

@author: Christian M Brodbeck
'''
import tkFileDialog



def ask_saveas(title, message, ext):
    return tkFileDialog.asksaveasfile(title=title, message=message)


def ask_dir(title = "Select Folder",
            message = "Please Pick a Folder",
            must_exist = True):
    return tkFileDialog.askdirectory(title=title, mustexist=must_exist)

