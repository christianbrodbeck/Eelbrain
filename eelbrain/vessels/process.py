'''
Created on Feb 24, 2012

@author: christian
'''

import mdp as _mdp

import data as _data


def rm_pca(ds, rm=[], source='MEG', target='MEG'):
    """
    Perform PCA and remove certain components. Use gui.pca to find components
    initially. Algorithm from the gui!
    
    """
    if not rm:
        raise ValueError("No components selected")
    if isinstance(source, basestring):
        source = ds[source]
    
    rm = sorted(rm)
    n_comp = max(rm) + 1
    
    # do the pca
    pca = _mdp.nodes.PCANode(output_dim=n_comp)
    for epoch in source.data:
        pca.train(epoch)
    pca.stop_training()
    
    # remove the components
    n_epochs, n_t, n_sensors = source.data.shape
    data = source.data.copy() # output data
    
    # take serialized data views for working with the PCANode
    new_data = data.view()
    old_data = source.data.view()
    
    # reshape the views
    new_data.shape = (n_epochs * n_t, n_sensors)
    old_data.shape = (n_epochs * n_t, n_sensors)
    
    # project the components and remove
    proj = pca.execute(old_data)
    for i in xrange(proj.shape[1]):
        if i not in rm:
            proj[:,i] = 0 
    rm_comp_data = pca.inverse(proj)
    new_data -= rm_comp_data
    
    # create the output new ndvar 
    dims = source.dims
    properties = source.properties
    ds[target] = _data.ndvar(dims, data, properties, name=target)

