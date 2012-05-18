import eelbrain.psyphys as pp
import eelbrain.wxgui.psyphys as ppgui

e = pickle.load(open(u'/Users/christian/Data/tutorial_scr.eelbrain'))

vlist = ppgui.list(e.skin_conductance, e.SCRs, e.event2_enum)

e.SCRs.p.color.set((1, 0.1, 0.0))
# (press the update button in the viewer's toolbar to see the color change)


# display a smaller time window by selecting a subset of the events
trial = e.variables['trial']
vlist.set_zoom(2, trial==[0, 1])
# the first number is an index into the datasets displayed by vlist - to see 
# them in order look at the representation of the viewer object (>>> vlist).
# trial==[0, 1] then changes the viewer to display only events with value 0 
# or 1 on the trial variable

# reset the zoom with:
# >>> vlist.set_zoom()
