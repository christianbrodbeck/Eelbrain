"""
Frame for displaying the tree structure of datasets in an experiment

TODO: use PropertyGrid

"""

import logging
import os

import wx
import wx.lib.mixins.treemixin

import ID
from eelbrain.wxutils import Icon



class ExperimentFrame(wx.Frame):
    def __init__(self, e, parent, varname='-', size=(250, 500), **kwargs): #parent, 
        if isinstance(varname, basestring):
            title = varname#e.name
        else:
            title = '-'
        if e.path:
            title += ' ({0})'.format(os.path.split(e.path)[-1])
        wx.Frame.__init__(self, parent, size=size, title=title, **kwargs)
        self.e = e
        self.varname = varname
        self.shell = parent
        
        # elements
        self.tree = DatasetTree(e, self)
        
        # toolbar
        tb = self.CreateToolBar(wx.TB_HORIZONTAL)
        tb.SetToolBitmapSize(size=(32,32))
        tb.AddLabelTool(wx.ID_SAVE, "Save", Icon("tango/actions/document-save"))
        #tb.AddLabelTool(ID.DATASET_IMPORT, "Import", 
        #                Icon("datasets/import"))
        #self.Bind(wx.EVT_TOOL, self.OnImport, id=ID.DATASET_IMPORT)
        tb.AddLabelTool(ID.DATASET_ATTACH, "Attach", 
                        Icon("actions/attach"))
        self.Bind(wx.EVT_TOOL, self.OnDatasetAttach, id=ID.DATASET_ATTACH)
#        tb.AddSeparator()
#        tb.AddLabelTool(wx.ID_REFRESH, "Refresh", Icon("tango/actions/view-refresh"))
#        self.Bind(wx.EVT_TOOL, self.tree.Refresh, id=wx.ID_REFRESH)
        tb.Realize()
        self.toolbar = tb
        self.Bind(wx.EVT_TOOL, self.OnSave, id=wx.ID_SAVE)
        # events
        self.Bind(wx.EVT_ACTIVATE, self.tree.OnActivate)
        self.Bind(wx.EVT_CLOSE, self.OnClose)
        # Finish
        self.Show()
    # commands
    def OnDatasetAttach(self, event=None):
        dataset = self.tree.GetSelectedDataset()
        if dataset:
            self.shell.attach(dataset, e_varname=self.varname, 
                                     internal_call=True)
        else:
            self.shell.attach(self.e, e_varname=self.varname,
                                     internal_call=True)
    #def OnImport(self, event=None):
    #    pass
    def OnSave(self, event=None):
        self.e.save()
        if isinstance(self.varname, basestring):
            varname = self.varname
        else:
            varname = '-'
        self.shell.shell_message(varname+'.save("%s")'%self.e.path, 
                                        ascommand=True, internal_call=True)
    def OnClose(self, event=None):
        logging.debug("Exp OnClose")
#        self.shell.deframe_experiment(self)
        event.Skip()




class DatasetTree(wx.lib.mixins.treemixin.VirtualTree, wx.TreeCtrl):
    def __init__(self, e, parent, **kwargs):
        wx.TreeCtrl.__init__(self, parent, #**kwargs)
                             style=wx.TR_HIDE_ROOT|wx.TR_HAS_BUTTONS, **kwargs) #|wx.TR_EDIT_LABELS|wx.TR_HAS_BUTTONS|wx.TR_DEFAULT_STYLE
        self.parent = parent
        self.e = e
        # ICONS f tree
        il = wx.ImageList(32,32)
        self.icons = {}
        for name in ["unknown", "uts", "utstopo", "event", 'spec',
                     "import", "import-wav", "import-egi", "import-aup",
                     "import-eph"]:
            self.icons[name] = il.Add(Icon("datasets/"+name))
        self.AssignImageList(il)
        # tree
        self.ExpandAll()#(root)
        self.Show()
        self.Bind(wx.EVT_LEFT_DCLICK, self.OnDoubleClick)
        # ?
        self.Bind(wx.EVT_TREE_ITEM_EXPANDED, self.OnItemExpanded)#, self.tree)
        #self.Bind(wx.EVT_TREE_ITEM_COLLAPSED, self.OnItemCollapsed, self.tree)
        #self.Bind(wx.EVT_TREE_SEL_CHANGED, self.OnSelChanged, self.tree)
        #self.Bind(wx.EVT_TREE_BEGIN_LABEL_EDIT, self.OnBeginEdit, self.tree)
        #self.Bind(wx.EVT_TREE_END_LABEL_EDIT, self.OnEndEdit, self.tree)
        #self.Bind(wx.EVT_TREE_ITEM_ACTIVATED, self.OnActivate, self.tree)
        self.RefreshItems()
        self.ExpandAll()#(root)
    def GetSelectedDataset(self):
        treeId = self.GetSelection()
        index = self.GetIndexOfItem(treeId)
        return self.dataset_for_index(index)
    ## Virtual Tree
    def OnActivate(self, event=None):
#        logging.debug(" Tree Activate")
        self.RefreshItems()
        self.ExpandAll()
    def OnItemExpanded(self, event=None):
        item = event.GetItem()
#        logging.debug(" OnItemExpanding "+str(item))
        self.RefreshChildrenRecursively(item, self.GetIndexOfItem(item))
        #self.RefreshItems()
    def Refresh(self, event=None):
        self.RefreshItems()
        self.ExpandAll()
    def dataset_for_index(self, index):
        if len(index) == 0:
            return self.e
        else:
            d = self.e.children[index[0]]
            for i in index[1:]:
                d = d.children[i]
            return d
    def OnGetChildrenCount(self, index):
        d = self.dataset_for_index(index)
        return len(d.children)
    def OnGetItemText(self, index):#, column):
        d = self.dataset_for_index(index)
        name = "{name} ({cl})".format(name=d.name, cl=d.__class__.__name__)
        return name
    def OnGetItemImage(self, index, which):
        data_type = self.dataset_for_index(index).properties['data_type']
        #print "which:", which, " data_type:", data_type
        if data_type in self.icons:
            icon = self.icons[data_type]
        elif isinstance(data_type, basestring) and data_type[:6] == 'import':
            icon = self.icons['import']
        else:
            icon = self.icons['unknown']
        return icon
    # idle
    """
    def AddDataset_toNode_(self, dataset, parentitem):
        newitem_id = self.AppendItem(parentitem, dataset.name)
        self.SetItemPyData(newitem_id, dataset)
        icon = self.icons['dataset']
        self.SetItemImage(newitem_id, icon, wx.TreeItemIcon_Normal)
        #self.Expand(newitem_id)
        #self.SelectItem(newitem_id, True)
        for child in dataset.children:
            self.AddDataset_toNode_(child, newitem_id)
    """
    # Tree Cmds
    def OnDoubleClick(self, event=None):
        logging.info("experiment_frame DoubleClick!")
        dataset = self.GetSelectedDataset()
        if dataset:
            try:
                v = self.parent.shell.gui_shellfunc(dataset)
                self.parent.shell.global_namespace['_v'] = v
            except Exception as inst:
                self.parent.shell.shell_message(str(inst), internal_call=True)
                raise
                
#                msg = "print(e{ne}.d{nd})".format(ne='', nd=dataset.id)
#                self.parent.shell.shell_message(msg, ascommand=True, internal_call=True)
#                text = dataset.__str__() # '\n' +
#                self.parent.shell.shell_message(text, internal_call=True)
        #self.parent.OnDatasetAttach()


#class VarList():

'''
## MARK: unused

class DatasetTreeBook(wx.Treebook):
    def __init__(self, e, parent, id=-1): #parent, 
        wx.Treebook.__init__(self, parent, id, style=wx.BK_DEFAULT)
        self.parent = parent
        self.e = e
        # images f tree
        il = wx.ImageList(32,32)
        il.Add(Icon("mimetypes.application-x-executable"))
        il.Add(Icon("datasets.base"))
        self.AssignImageList(il)
        
        self.Bind(wx.EVT_TREEBOOK_PAGE_CHANGED, self.OnPageChanged)
        self.Bind(wx.EVT_TREEBOOK_PAGE_CHANGING, self.OnPageChanging)
        
        self._updatePanels()
        
        # This is a workaround for a sizing bug on Mac...
        wx.FutureCall(100, self.AdjustSize)
    def AdjustSize(self):
        #print self.GetTreeCtrl().GetBestSize()
        self.GetTreeCtrl().InvalidateBestSize()
        self.SendSizeEvent()
        
    # create the tree structure
    def _updatePanels(self):
        p = ExperimentPanel(self, -1)
        self.AddPage(p, "Experiment", imageId=0)
        pos = 1
        for dataset in self.e.importers:
            self._makePanel(dataset, parent_page, pos)
            pos += 1
            
    def _makePanel(self, dataset, pos):
        # get the appropriate panel
        p = DatasetPanel(self, pos, -1)
        self.InsertSubPage(pos, p, dataset.name, imageId=1)
            
    def OnPageChanged(self, event):
        event.Skip()
    def OnPageChanging(self, event):
        event.Skip()
        
        

class ExperimentPanel(wx.Panel):
    pass


class DatasetPanel(wx.Panel):
    def __init__(self, parent, pos, *args, **kwargs):
        wx.Panel.__init__(self, parent, *args, **kwargs)
        self.pos = pos
'''
