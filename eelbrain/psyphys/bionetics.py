"""
Bionetics
=========

Adding a derived dataset::

    >>> new = DerivedDataset(parent, name=None, **kwargs)

kwargs: cache='hd'  - 'ram' 'hd' None


TODO: better way to predict name (when I change the import script, e.d3 
suddenly becomes e.d6, so analysis scripts don't match anymore).


the following are lazy creation attributes, meaning that an @property attri-
bute looks for a stored item._attribute instance, and computes it if it can not
find it.

dataset.properties
    properties (depend on parent dataset and processing settings)

dataset.compiled
    (items used repeatedly during segment processing)

segment.data
    the data?




"""
import cPickle as pickle 
import os, shutil

from eelbrain import ui
from vars import VarMothership
from eelbrain.utils._basic_ops_ import test_attr_name


# extension used to save experiments
_extension = 'eelbrain'

def isexperiment(item):
    return item.__class__.__name__ == 'Experiment'

#def isDataset(item):
#    return all(hasattr(item, attr) for attr in ('experiment', 'variables', 
#                                                'properties'))


class Experiment(object):
    def __init__(self, name="Experiment", path=None):
        """
        An Experiment instance is the central container for working with data.
         
        """
        self.children = []
        self.variables = VarMothership()
        self.name = name
        self.path = path
#        self.cachePath = None
        self._cacheIDs = set()
        self._itemIDs = set()
        self._item_names = {}
#        self.collectors = {}
    # managing ExperimentItems
    def rename_item(self, item, name):
        "rename an ExperimentItem"
        if hasattr(self, name):
            raise ValueError("Experiment already has an attribute named %r" % name)
        else:
            test_attr_name(name)
        
        delattr(self, item.name)
        setattr(self, name, item)
        setattr(item, 'name', name)
    def _register_item(self, item, name):
        "adds item to experiment and returns unique id"
        # check name
        test = name.format(i=0, c='x', p='x')
        test_attr_name(test)
        
        # format name
        if '{c}' in name:
            name = name.format(i='{i}', c=item.__class__.__name__)
        if '{i}' in name:
            i = 0
            while hasattr(self, name.format(i=i)):
                i += 1
            name = name.format(i=i)
        else:
            if hasattr(self, name):
                raise ValueError("Experiment already has attribute named %r"
                                 % name)
        
        setattr(self, name, item)

        # get ID
        ID = max(set([-1]).union(self._itemIDs)) + 1
        self._itemIDs.add(ID)
        self._item_names[ID] = name
        
        return ID, name
    def _del_item(self, item):
        if len(item.children) > 0:
            raise ValueError("Can only delete items at the bottom of the "
                             "hierarchy")
        ID = item.id
        name = self._item_names.pop(ID)
        delattr(self, name)
        self._itemIDs.remove(ID)
        if item in self.children:
            self.children.remove(item)
#    @property
#    def varlist(self):
#        self.variables.varlist
    def __repr__(self):
        txt = "<Experiment:  name=%r, path=%r>" % (self.name, self.path)
        return txt
    def __str__(self):
        lines = []
        for c in self.children:
            lines += c._get_tree_repr_()
        return '\n'.join(lines)
        
#        lines = ["Eelbrain Experiment:", " > Variables:"]
#        vstr = "    {n} ({dt})"
#        for var in self.variables.commanders:
#            lines.append(vstr.format(n=var.name, dt=var.dtype))
#        lines.append(" > Importers:")
#        dstr = "    {name} ({dt})"
#        for i in self.children:
#            lines.append(dstr.format(name=i.name, dt=i['data_type']))
#        return '\n'.join(lines)
        
        # DATASETS
#        txt = ["Datasets:"]
#        i = 0
#        h = 0
#        for d in self.children:
#            i = d.addShortcutToExperiment(i, h, txt)
#        # VARIABLES
#        txt.append( "Variables:" )
#        vars = []
#        for i, v in enumerate(self.variables):
#            s=v.shortcut
#            if s in vars:
#                s+=str(i)
#            vars.append(s)
#            exec("self.v%s = v"%s)
#            txt.append( "    v%s: %s"%(s, v.name) )
#        return '\n'.join(txt)
    # File Ops
    def save(self):
        """
        save the experiment in the path used before (Experiment.path). If it 
        has not been saved before, ask for a new path.
        
        """
        if self.path is None:
            self.saveas()
        else:
            for d in self.children:
                d.close()
            with open(self.path, 'w') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
    def saveas(self, path=None):
        "Save the experiment. if path is None, a system file dialog is used."
        if not path:
            msg = ("Pick a filename to pickle your data. A folder named"+\
                   " '<name>.cache' will be created to store data files.")
            path = ui.ask_saveas(title="Save Pickled EEG",
                                 message=msg, 
                                 ext = [(_extension, "pickled eelbrain experiment")])
        if path:
            if not path.endswith(os.path.extsep + _extension):
                path = os.path.extsep.join((path, _extension))
            
            # if the experiment already has a cache, copy it
            if self.path is not None:
                old_cache = self._cache_path
                new_cache = path + '.cache'
                if os.path.exists(old_cache):
                    shutil.copytree(old_cache, new_cache)
            
            # save the experiment
            self.path = path
            self.save()
        else:
            ui.msg("Saving aborted")
    @property
    def _cache_path(self):
        return self.path + '.cache'
    def _get_memmap_id(self):
        if self.path == None:
            print "need to save experiment before caching"
            self.save()
        if not os.path.exists(self.path + '.cache'):
            os.mkdir(self.path + '.cache')
        if len(self._cacheIDs) == 0:
            ID = 0
        else:
            ID = list( set(range(max(self._cacheIDs)+2)).difference(self._cacheIDs) )[0]
        self._cacheIDs.add(ID)
        return ID
    def _free_memmap_id(self, ID):
        self._cacheIDs.remove(ID)


