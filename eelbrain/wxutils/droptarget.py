import logging

import wx



class FilenameDropTarget(wx.FileDropTarget):
    """
    File drop target: http://wiki.wxpython.org/DragAndDrop
    
    (!) apparently the FileDropTarget can only be used once; when I saved the 
    instance as an Editor attribute and SetDropTarget it to a second editor
    this caused a segmentation fault.
    
    """
    def __init__(self, text_target):
        wx.FileDropTarget.__init__(self)
        self.text_target = text_target
    
    def OnDropFiles(self, x, y, filenames):
        msg = "DROP! %r" % filenames
        logging.info(msg)
        if len(filenames) == 1:
            filenames = "'%s'"%(filenames[0])
        self.text_target.ReplaceSelection(str(filenames))


class TextDropTarget(wx.TextDropTarget):
    def __init__(self, text_target):
        wx.TextDropTarget.__init__(self)
        self.text_target = text_target
    
    def OnDropText(self, x, y, text):
        msg = "DROP! %r" % text
        logging.info(msg)
        self.text_target.ReplaceSelection(text)


class StringDropTarget(wx.DropTarget):
    """
    DropTarget for multiple data types based on:
    http://www.wiki.wxpython.org/DragAndDrop#wxDataObjectComposite
    
    """
    def __init__(self, target):
        wx.DropTarget.__init__(self)
        self.target = target
        
        self.do = wx.DataObjectComposite()  # the dataobject that gets filled with the appropriate data
        self.filedo = wx.FileDataObject()
        self.textdo = wx.TextDataObject()
#        self.bmpdo = wx.BitmapDataObject()
        self.do.Add(self.filedo)
        self.do.Add(self.textdo)
#        self.do.Add(self.bmpdo)
        self.SetDataObject(self.do)
        
    def OnData(self, x, y, d):#, data):
        if self.GetData():
            df = self.do.GetReceivedFormat().GetType()
#            data = self.GetData()
            
            if df in [wx.DF_UNICODETEXT, wx.DF_TEXT]:
                text = self.textdo.GetText()
            elif df == wx.DF_FILENAME:
                filenames = self.filedo.GetFilenames()
                if len(filenames) == 1:
                    filenames = '%r' % filenames[0]
                text = str(filenames)
            
            msg = "OnData! %r" % text   
            logging.info(msg)
            self.target.ReplaceSelection(text)
        return d
    


def set_for_strings(window):
    drop_target = StringDropTarget(window)
    window.SetDropTarget(drop_target)
    

