"""
Illustrates data import. Data available from:
http://christianmbrodbeck.github.com/Eelbrain/psyphys-tutorial.html
 
"""
import eelbrain.psyphys as pp
import eelbrain.wxgui.psyphys as ppgui


e = pp.Experiment()

# define import settings
i = pp.importer.txt(e)
i.p.source.set(u'/Users/christian/Data/simulated_scr')
i.p.samplingrate = 200
i.p.channels[0] = 'event', 'evt'
i.p.channels[1] = 'skin_conductance', 'uts'
i.p.vars_from_names[:3] = 'subject'

# import the data
e.saveas(u'/Users/christian/Data/tutorial_scr')
i.get()

# set condition based on event magnitude
attach(e.variables)
e.variables.new_parasite(magnitude, 'condition', 'dict', {4:'control', 5:'test'})

# extract SCRs
d = pp.op.physio.SCR(e.skin_conductance, name='SCRs')

# add 'trial' variable
d = pp.op.evt.Enum(e.event, 'event2_enum') 
d.p.var = 'trial'
d.p.count = 'magnitude'

# save the experiment
e.save()