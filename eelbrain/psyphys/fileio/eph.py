'''
Created on Dec 8, 2010

@author: christian
'''

import os, logging

import numpy as np


def read(path, mrk=True):
    """
    Returns data (numpy array) and properties (dictionary)
    
    if mrk is True, the function looks for a path+'.mkr' file. If one is found,
    properties contains 2 more entries, 'mrk_hd' (the header) and 'mrk' (the
    markers)
    
    """
    # read data file
    with open(path) as f:
        hdr = f.readline().split()
        
        electrodes = int(hdr[0])
        samples = int(hdr[1])
        samplingrate = int(hdr[2])
    
        data = np.fromfile(f, sep=' ').reshape((samples, electrodes))
        properties = dict(samplingrate=samplingrate)
    
    # look for mrk file
    mpath = os.extsep.join((path, 'mrk'))
    if os.path.exists(mpath):
        # read mrk file
        markers = read_mrk(mpath)
        properties['mrk'] = markers
    
    return data, properties


def read_mrk(path, cleanup=False):
    """
    reads a .eph.mrk file.
    
    returns list of markers [(t1, t2, name), ...] 
    
    if cleanup is True, events with identical t1 and t2 are merged
    (using the label of the event earliest in the file) 
     
    """    
    with open(path) as f:
        line1 = f.readline()
        markers = [] # (t1, t2, name)
        for line in f:
            items = line.split()
            t1 = int(items[0])
            t2 = int(items[1])
            l_start = line.index('"')+1
            l_end = line.index('"', l_start)
            label = line[l_start:l_end]
            if cleanup and len(markers)>0:
                last = markers[-1]
                if t1 ==last[0] and t2==last[1]:
                    continue
            markers.append((t1, t2, label))

    return markers
            



def write(path, data, samplingrate, mrk=False, fmt='%.7g'):
    """
    Writes a .eph file.
    
    mrk must be of the same format as provided by the eph.read function:    
    [(t1, t2, name), ...]
    
    fmt is used by numpy.savetxt 
    
    """
#    electrodes = properties['electrodes']
    samples, electrodes = data.shape
    
    with open(path, 'w') as f:
        hdr = [str(item) for item in [electrodes, samples, samplingrate]]
        line1 = '\t'.join(hdr)
        f.write(line1 + '\r\n')
        np.savetxt(f, data, fmt=fmt, newline='\r\n')
    
    if mrk:
        line = '\t%i\t%i\t"%s"\r\n'
        mpath = os.extsep.join((path, 'mrk'))
        with open(mpath, 'w') as f:
            f.write("TL02\r\n")
            for mark in mrk:
                f.write(line % mark)


#def create_mrk(path, epoch_length, overwrite=False):
#    """
#    creates an mrk file dividing the eph in equal epochs 
#     - path can be file or folder (creates mrk for all files ending in .eph)
#     - epoch length: length in frames of an epoch
#     - overwite: overwrite if a '....eph.mrk' file already exists
#    
#    """
#    pass


def _rebaseline(inpath, outpath, old_length, start, end, new_baseline_length=None):
    """
    start, end, new_baseline_length: all in frames
    
    new_baseline_length=None -> no baseline correction
    
    """
    for filename in os.listdir(inpath):
        if filename.endswith('.eph'):
            data, props = read(os.path.join(inpath, filename))
            ns = data.shape[1]
            
            data_ep = data.reshape((-1, old_length, ns))
            n_epochs = len(data_ep)
            logging.debug('%s, %s epochs'%(filename, n_epochs))
            
            data_out = data_ep[:,start:end]
            if new_baseline_length:
                baseline = data_out[:,:new_baseline_length].mean(1)[:,None,:]
                data_out -= baseline
            
            data_out = data_out.reshape((-1, ns))
            
            mrk = []
            new_len = end - start
            for i in range(n_epochs):
                t = i * new_len
                name = "epoch_%s"%i
                mrk.append((t, t, name))
            
            p_out = os.path.join(outpath, filename)
            write(p_out, data_out, props['samplingrate'], mrk=mrk)


def _avg(inpath, outpath, epoch_length, overwrite=False):
    """
    for each eph file in the inpath, reads the file, computes the average 
    epoch, and saves it with the same name in the outpath directory 
    
    """
    for filename in os.listdir(inpath):
        if filename.endswith('.eph'):
            p_out = os.path.join(outpath, filename)
            if overwrite or (not os.path.exists(p_out)):
                data, props = read(os.path.join(inpath, filename), mrk=False)
                nt, ns = data.shape
                
                if nt % epoch_length != 0:
                    raise ValueError("%s not divisible by epoch length"%filename)
                
                data = data.reshape((-1, epoch_length, ns))
                data = data.mean(0)
                
                nt, ns = data.shape
                assert nt == epoch_length
                
                write(p_out, data, props['samplingrate'])

def _gavg(files, out_path):
    """
    files: list of file names (each file containing one epoch)
    outpath: path to save average epoch
    
    """
    data = None
    for i, path in enumerate(files):
        data1, props1 = read(path, mrk=False)
        if data is None:
            shape = data1.shape
            newshape = (len(files), ) + shape
            data = np.empty(newshape)
        else:
            if shape != data1.shape:
                msg = "Episode %s does not match shape"%path
                raise ValueError(msg)
        
        data[i] = data1
    
    data = data.mean(0)
    write(out_path, data, props1['samplingrate'])



def _concatenate_epochs(source_folder, target_path, overwrite=False):
    """
    reads all eph files in source_folder, combines them, and writes eph and 
    marker files to target_path
    
    """
    if not overwrite:
        if os.path.exists(target_path):
            raise ValueError("%s already exists"%target_path)
    
    data = []
    mrk = []
    sr = None
    i_epoch = 0
    i_sample = 0
    
    for fname in os.listdir(source_folder):
        if fname.endswith('eph'):
            _data, props = read(os.path.join(source_folder, fname), mrk=False)
            data.append(_data)
            if sr == None:
                sr = props['samplingrate']
            else:
                assert sr == props['samplingrate'], ("Samplingrate mismatch "
                                                     "between files!")
            mrk.append((i_sample, i_sample, "epoch_%i"%i_epoch))
            i_epoch += 1
            i_sample += len(_data)
    
    out_data = np.vstack(data)
    write(target_path, out_data, sr, mrk=mrk)