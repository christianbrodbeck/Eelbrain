"""Embedd Mayavi in Eelbrain

for testing:

src = datasets.get_mne_sample(src='ico', sub=[0])['src']
brain = plot.brain.brain(src.source, mask=False,hemi='lh',views='lat')
"""
from traits.trait_base import ETSConfig
ETSConfig.toolkit = 'wx'

from logging import getLogger

from mayavi.core.ui.api import SceneEditor, MlabSceneModel
import numpy as np
from traits.api import HasTraits, Instance
from traitsui.api import View, Item, HGroup, VGroup
from tvtk.api import tvtk
from tvtk.pyface.toolkit import toolkit_object
import wx

from .._wxgui.frame import EelbrainFrame
from .._wxutils import Icon


SCENE_NAME = 'scene_%i'

# undecorated scene
Scene = toolkit_object('scene:Scene')


class MayaviView(HasTraits):

    view = Instance(View)

    def __init__(self, width, height, n_rows, n_columns):
        HasTraits.__init__(self)

        n_scenes = n_rows * n_columns
        if n_scenes < 1:
            raise ValueError("n_rows=%r, n_columns=%r" % (n_rows, n_columns))

        self.scenes = tuple(MlabSceneModel() for _ in xrange(n_scenes))
        for i, scene in enumerate(self.scenes):
            self.add_trait(SCENE_NAME % i, scene)

        if n_rows == n_columns == 1:
            self.view = View(Item(SCENE_NAME % 0,
                                  editor=SceneEditor(scene_class=Scene),
                                  resizable=True, show_label=False),
                             width=width, height=height, resizable=True)
        else:
            rows = []
            for row in xrange(n_rows):
                columns = []
                for column in xrange(n_columns):
                    i = row * n_columns + column
                    item = Item(SCENE_NAME % i,
                                editor=SceneEditor(scene_class=Scene),
                                resizable=True, show_label=False)
                    columns.append(item)
                rows.append(HGroup(*columns))
            self.view = View(VGroup(*rows))

        self.figures = [scene.mayavi_scene for scene in self.scenes]


class BrainFrame(EelbrainFrame):

    def __init__(self, parent, brain, title, width, height, n_rows, n_columns):
        EelbrainFrame.__init__(self, parent, wx.ID_ANY, title)

        # toolbar
        tb = self.CreateToolBar(wx.TB_HORIZONTAL)
        tb.SetToolBitmapSize(size=(32, 32))
        tb.AddLabelTool(wx.ID_SAVE, "Save", Icon("tango/actions/document-save"))
        self.Bind(wx.EVT_TOOL, self.OnSaveAs, id=wx.ID_SAVE)
        self.Bind(wx.EVT_UPDATE_UI, self.OnUpdateUISave, id=wx.ID_SAVE)
        tb.Realize()

        self.mayavi_view = MayaviView(width, height, n_rows, n_columns)
        # Use traits to create a panel, and use it as the content of this
        # wx frame.
        self.ui = self.mayavi_view.edit_traits(parent=self,
                                               view=self.mayavi_view.view,
                                               kind='subpanel')
        self.panel = self.ui.control
        # Hide the toolbar (the edit_traits command assigns scene_editor)
        for scene in self.mayavi_view.scenes:
            scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()

        self.panel.SetSize((width + 2, height + 2))
        self.Fit()

        self.figure = self.mayavi_view.figures

        self._brain = brain
        self.Bind(wx.EVT_CLOSE, self.OnClose)  # remove circular reference

    def CanCopy(self):
        return True

    def Copy(self):
        ss = self._brain.screenshot('rgba', True)
        ss = np.round(ss * 255).astype(np.uint8)
        h, w, _ = ss.shape
        image = wx.ImageFromDataWithAlpha(
            w, h, ss[:,:,:3].tostring(), ss[:,:,3].tostring())
        bitmap = image.ConvertToBitmap()
        if not wx.TheClipboard.Open():
            getLogger('eelbrain').debug("Failed to open clipboard")
            return
        try:
            wx.TheClipboard.SetData(bitmap)
        finally:
            wx.TheClipboard.Close()
            wx.TheClipboard.Flush()

    def OnClose(self, event):
        self._brain._frame_is_alive = False
        self._brain = None
        event.Skip()

    def OnSave(self, event):
        self.OnSaveAs(event)

    def OnSaveAs(self, event):
        default_file = '%s.png' % self.GetTitle().replace(': ', ' - ')
        dlg = wx.FileDialog(self, "If no file type is selected below, it is "
                                  "inferred from the extension.",
                            defaultFile=default_file,
                            wildcard="Any (*.*)|*.*|PNG (*.png)|*.png",
                            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        if dlg.ShowModal() == wx.ID_OK:
            self._brain.save_image(dlg.GetPath(), 'rgba', True)
        dlg.Destroy()

    def OnUpdateUISave(self, event):
        event.Enable(True)

    def OnUpdateUISaveAs(self, event):
        event.Enable(True)
