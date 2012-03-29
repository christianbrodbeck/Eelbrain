'''
Created on Feb 24, 2012

@author: christian
'''

import numpy as _np
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
    data = source.get_data(('time', 'sensor'))
    
    # do the pca
    pca = _mdp.nodes.PCANode(output_dim=n_comp)
    for epoch in data:
        pca.train(epoch)
    pca.stop_training()
    
    # project into the pca space
    n_epochs, n_t, n_sensors = data.shape
    old_data = data.reshape((n_epochs * n_t, n_sensors))
    proj = pca.execute(old_data)
    
    # flatten retained components
    for i in xrange(proj.shape[1]):
        if i not in rm:
            proj[:,i] = 0 
    
    # remove the components
    rm_comp_data = pca.inverse(proj)
    new_data = data - rm_comp_data.reshape(data.shape)
    
    # create the output new ndvar 
    dims = source.get_dims(('time', 'sensor'))
    properties = source.properties
    ds[target] = _data.ndvar(dims, new_data, properties, name=target)


def mark_by_threshold(dataset, DV='MEG', threshold=2e-12, above=True, below=False, 
                      target='reject'):
    """
    Marks epochs based on a threshold criterion (any sensor exceeding the 
    threshold at any time) 
    
    above: True, False, None
        How to mark segments that exceed the threshold: True->good; 
        False->bad; None->don't change
    below:
        Same as ``above`` but for segments that do not exceed the threshold
    threshold : float
        The threshold value.
    target : factor or str
        Factor (or its name) in which the result is stored. If ``var`` is 
        a string and the dataset does not contain that factor, it is 
        created.
    
    """
    if DV is None:
        DV = dataset.default_DV
        if DV is None:
            raise ValueError("No valid DV")
    if isinstance(DV, basestring):
        DV = dataset[DV]
    
    # get the factor on which to store results
    if _data.isfactor(target) or _data.isvar(target):
        assert len(target) == dataset.N
    elif isinstance(target, basestring):
        if target in dataset:
            target = dataset[target]
        else:
            x = _np.zeros(dataset.N, dtype=bool)
            target = _data.var(x, name=target)
            dataset.add(target)
    else:
        raise ValueError("target needs to be a factor")
    
    # do the thresholding
    if _data.isndvar(DV):
        for ID in xrange(dataset.N):
            data = DV.get_data(('time', 'sensor'), ID)
            v = _np.max(_np.abs(data))
            
            if v > threshold:
                if above is not None:
                    target[ID] = above
            elif below is not None:
                target[ID] = below
    else:
        for ID in xrange(dataset.N):
            v = DV[ID]
            
            if v > threshold:
                if above is not None:
                    target[ID] = above
            elif below is not None:
                target[ID] = below




