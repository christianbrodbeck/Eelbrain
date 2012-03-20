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
    
    # project into the pca space
    n_epochs, n_t, n_sensors = source.data.shape
    old_data = source.data.reshape((n_epochs * n_t, n_sensors))
    proj = pca.execute(old_data)
    
    # flatten retained components
    for i in xrange(proj.shape[1]):
        if i not in rm:
            proj[:,i] = 0 
    
    # remove the components
    rm_comp_data = pca.inverse(proj)
    new_data = source.data - rm_comp_data.reshape(source.data.shape)
    
    # create the output new ndvar 
    dims = source.dims
    properties = source.properties
    ds[target] = _data.ndvar(dims, new_data, properties, name=target)

