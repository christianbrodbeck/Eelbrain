import eelbrain.psyphys as pp
import eelbrain.wxgui.psyphys as ppgui

e = pickle.load(open(u'/Users/christian/Data/tutorial_scr.eelbrain'))

v = ppgui.list(e.skin_conductance, e.SCRs, e.event)

e.SCRs.p.color.set((1, 0.1, 0.0))
# (press the update button in the viewer's toolbar to see the color change)