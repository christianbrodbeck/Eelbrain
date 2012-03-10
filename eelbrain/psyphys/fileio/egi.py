import struct
#import array as pyarray
#import time
import logging

import numpy as np

#from eelbrain import ui


    
    
def egi_goodsegments(f, segment_length, good=['good']):
    """
    segment_length in ms
    """
    goodSegments=[]
    lines = f.readlines()
    for line in lines[1:]:
        l=line.split()
        if l[3] in good:
            i = int(l[1])
            goodSegments.append(i/segment_length)
    return goodSegments


def readegi_hdr(f, forceversion=None, voc=False):
    #file://localhost/Volumes/MyBook/MATLAB/eeglab7_1_4_15b/functions/sigprocfunc/readegihdr.m
    #
    #http://docs.python.org/library/struct.html
    #
    
    version, = struct.unpack('>I', f.read(4)) #(fid,1,'integer*4')
    if forceversion != None:
        version = forceversion
    if not 2 <= version <= 7:
        raise IOError('EGI Simple Binary Versions 2-7 supported only. Got %s.'%type(version))

    year, month, day, hour, minute, second = struct.unpack('>6H', f.read(12))   #fread(fid,1,'integer*2');
    millisecond, = struct.unpack('>I', f.read(4))
    
    head={'version': version,
            'samples': 0,
            'n_segments': 0,
            'segment_length': 0,
            'n_cells': 0,
            'eventcode': [] }
    if version in [2,3]:
        dtype = np.int32
    elif version in [4,5]:
        dtype = np.float32
    elif version in [6,7]:
        dtype = np.float64
    head['dtype'] = dtype
    
    (head['samplingrate'],
            head['n_sensors'],
            head['gain'],
            head['bits'],
            head['range']) = struct.unpack('>5H', f.read(10))
    if version in [2,4,6]:
        head['samples'], = struct.unpack('>I', f.read(4))
        head['segmented']=False
    elif version in [3,5,7]:    #segmented
        head['segmented']=True
        head['n_cells'], = struct.unpack('>H', f.read(2))
        if head['n_cells']:
            head['cells']=[]
            for i in range(head['n_cells']):
                cellname_len = f.read(1) # fread(fid,1,'uchar')
                #print '??? %s %s ord: %s'%(catname_len, type(catname_len), ord(catname_len))
                head['cells'].append(f.read( ord(cellname_len) )) #char(fread(fid,catname_len,'uchar'))
        head['n_segments'], = struct.unpack('>H', f.read(2))
        head['segment_length'], = struct.unpack('>I', f.read(4))
    else:
        raise IOError('Invalid EGI version')


    head['eventtypes'], = struct.unpack('>H', f.read(2))

    if head['eventtypes']==0:
        head['eventcode']=[] #[None]*4 #  [1,1:4] = None
    else:
        for i in range(head['eventtypes']):
            head['eventcode'].append( f.read(4) )#fread(fid,[1,4],'uchar')

    if voc:
        print 'header:\n', head
    return head


def extract_egi_segments(filename, dataset, cellvar, filevars, RawSegment, 
                                downsample=False, desired_segments=[], forceversion=None):
    """ file://localhost/Volumes/MyBook/MATLAB/eeglab7_1_4_15b/functions/sigprocfunc/readegi.m
    extracts segments form a given EGI binary no events file
    
    OLD! REPLACED BY segments_reader BELOW
    
    """
    with open(filename, mode='rb') as f:
        # get our header structure
        head = readegi_hdr(f, forceversion)
        assert head['segmented']

        # each type of event has a dedicated "channel"
        if head['eventtypes']:
            logging.warning(" File contains %s event channels. They are ignored."%(head['eventtypes']))

        if desired_segments==[]:
            desired_segments = range(head['n_segments'])
        

        # get datatype from version number
        v = head['version']
        if v in [2,3]:
            datatype = '>l' #int32 #'>h' # ???: h or H #'integer*2'
        elif v in [4,5]: 
            datatype = '>f4' #'>f' #'float32'
        elif v in [6,7]:
            datatype = '>f8' #'>d' #'float64'
        
        datatype_bitsize = struct.calcsize(datatype)
        datatype = np.dtype(datatype)
        
        # FIXME: need to take care of this eventually...
        """ 
        EventData = [];
        if head['eventtypes'] > 0:
            EventData = TrialData[:, head['n_sensors']:]
            TrialData = TrialData[:, :head['n_sensors']]

        # convert from A/D units to microvolts
        if head['bits'] != 0 and head['range'] != 0:
            TrialData = (head['range']/(2**head['bits']))*TrialData
        """

        # prepare shape args
        FrameVals = head['n_sensors'] + head['eventtypes']
        datashape = (head['segment_length'], FrameVals)
        datalen = FrameVals * head['segment_length']
        block_length = 6 + datalen * datatype_bitsize
        # for downsampling
        if downsample:
            datashape_1 = (head['segment_length'], head['n_sensors']) 
            datashape_2 = (datashape_1[0]/downsample,) + (downsample,) + datashape_1[1:]
        
        # read in epoch data
        segments = []
        f_data_start = f.tell()
        for i in desired_segments: #range(head['n_segments']):
            f.seek(f_data_start + i*block_length)
            segcatind, = struct.unpack('>H', f.read(2)) # ???: signed or unsigned?
            segtime, = struct.unpack('>I', f.read(4)) # ???
            data = np.fromfile(f, dtype=datatype, count=datalen).reshape(datashape)
            if head['eventtypes']:
                data = data[:,:head['n_sensors']]
            if downsample:
                data = data.reshape(datashape_2).mean(1)
            # create the segment
            s = RawSegment(dataset).initWithData_(data)
            s[cellvar] = segcatind
            for var,v in filevars:
                s[var]=v 
            s.ts=segtime
            segments.append(s)
    return segments






def segments_reader(filename, queue, desired_segments=[], forceversion=None):
    """ file://localhost/Volumes/MyBook/MATLAB/eeglab7_1_4_15b/functions/sigprocfunc/readegi.m
    extracts segments form a given EGI binary no events file and puts them into queue
    
    lp6: Done. Import took 265.145678997 s
    
    as [data, cellvar] list.
    when finished puts [False, False]
    """
    logging.debug(" segments_reader {0}".format(filename))
    #logging.debug( '...' + filename.split('/')[-1],
    with open(filename, mode='rb') as f:
        # get our header structure
        head = readegi_hdr(f, forceversion)
        assert head['segmented']

        # each type of event has a dedicated "channel"
        if head['eventtypes']:
            logging.warning(" File contains %s event channels. They are ignored."%(head['eventtypes']))

        if desired_segments == []:
            desired_segments = range(head['n_segments'])
        

        # get datatype from version number
        v = head['version']
        if v in [2,3]:
            datatype = '>l' #int32 #'>h' # ???: h or H #'integer*2'
        elif v in [4,5]: 
            datatype = '>f4' #'>f' #'float32'
        elif v in [6,7]:
            datatype = '>f8' #'>d' #'float64'
        
        datatype_bitsize = struct.calcsize(datatype)
        datatype = np.dtype(datatype)
        
        # FIXME: need to take care of this eventually...
        #""" 
        #EventData = [];
        #if head['eventtypes'] > 0:
        #    EventData = TrialData[:, head['n_sensors']:]
        #    TrialData = TrialData[:, :head['n_sensors']]

        ## convert from A/D units to microvolts
        #if head['bits'] != 0 and head['range'] != 0:
        #    TrialData = (head['range']/(2**head['bits']))*TrialData
        #"""

        # prepare shape args
        FrameVals = head['n_sensors'] + head['eventtypes']
        datashape = (head['segment_length'], FrameVals)
        datalen = FrameVals * head['segment_length']
        block_length = 6 + datalen * datatype_bitsize
        #print type(datalen), type(block_length)
        # read in epoch data
        f_data_start = f.tell() # returns Long
        for i in desired_segments: #range(head['n_segments']):
            f.seek(f_data_start + int(i*block_length))
            # FIXME: desired_segments comes as float list -- should beint???
            segcatind, = struct.unpack('>H', f.read(2)) # ???: signed or unsigned?
            segtime, = struct.unpack('>I', f.read(4)) # ???
            data = np.fromfile(f, dtype=datatype, count=datalen).reshape(datashape)
            if head['eventtypes']:
                data = data[:,:head['n_sensors']]
            queue.put((data, [segtime, segcatind]))
        queue.put((False, False))
        #print "  \t%s"%(time.time() - t_start) #).rjust(70)










    
#def readegi(filename=None, dataChunks=[], forceversion=None):
#    """
#    Direct translation of the matlab function after:
#    file://localhost/Volumes/MyBook/MATLAB/eeglab7_1_4_15b/functions/sigprocfunc/readegi.m
#    """
#    if filename==None:
#        filename = ui.ask_file(title="Pick Egi File")
#    
#    with open(filename, mode='rb') as f:
#
#        # get our header structure
#        print 'Importing binary EGI data file ...'
#        head = readegi_hdr(f, forceversion)
#
#        # do we have segmented data?
#        if head['version'] in [3, 5, 7]: #[2,4,6]: ???:wtf
#            segmented = True
#        else:
#            segmented = False
#        ## my HACK
#        #if head['n_segments']>0:
#
#
#        # each type of event has a dedicated "channel"
#        FrameVals = head['n_sensors'] + head['eventtypes']
#
#        if segmented:
#            if dataChunks==[]:
#                desiredSegments = range(head['n_segments'])
#            else:
#                desiredSegments = dataChunks
#
#            nsegs = len(desiredSegments)
#
##            readexpected = FrameVals*head['segment_length']*nsegs
##        else:
##            if dataChunks == []:
##                desiredFrames = range(head['samples']) #[1:head.samples] = dataChunks
##            else:
##                desiredFrames = dataChunks
#
#            #nframes = len(desiredFrames)
#            #readexpected = FrameVals*nframes
#
#        # get datatype from version number
#        v = head['version']
#        if v in [2,3]:
#            datatype = np.dtype('>l') #int32 #'>h' # ???: h or H #'integer*2'
#        elif v in [4,5]: 
#            datatype = np.dtype('>f4') #'>f' #'float32'
#        elif v in [6,7]:
#            datatype = np.dtype('>f8') #'>d' #'float64'
#        else:
#            raise IOError('Unknown data format')
#
#        # read in epoch data
##        readtotal = 0
#        j=0
#        SegmentCatIndex = []
#        SegmentStartTime = []
#        if segmented:
#            datashape = (head['segment_length'], FrameVals)
#            datalen = FrameVals * head['segment_length']
#            for i in range(head['n_segments']):
#                segcatind, = struct.unpack('>H', f.read(2)) # ???: signed or unsigned?
#                segtime, = struct.unpack('>I', f.read(4)) # ???
#
#                #[tdata, count] = fread(fid,[FrameVals,head.segment_length],datatype);
#                # http://www.johnny-lin.com/cdat_tips/tips_fileio/bin_array.html
#                #
#                #tdata = pyarray.array(datatype)
#                #tdata.read(f, datalen)
#                #data = array(tdata).reshape(datashape)
#                data = np.fromfile(f, dtype=datatype, count=datalen).reshape(datashape)
#                f.close()
#                return head, data, segtime, segcatind
#                # check if this segment is one of our desired ones
#                if i in desiredSegments:
#                    j+=1;
#                    SegmentCatIndex.append(segcatind)
#                    SegmentStartTime.append(segtime)
#         
#           
##                    readtotal += count
#
#                if j >= nsegs:
#                    break
#
#        else:
#            pass
#        """
#            # read unsegmented data
#
#            % if dataChunks is empty, read all frames
#            if isempty(dataChunks)
#
#                [TrialData, readtotal] = fread(fid, [FrameVals,head.samples],datatype);     
#          
#            else   % grab only the desiredFrames
#                   % This could take a while...
#
#                for i=1:head.samples,
#
#                  [tdata, count] = fread(fid, [FrameVals,1],datatype);
#          
#                  % check if this segment is a keeper
#                  if ismember(i,desiredFrames),
#                      j=j+1;
#                      TrialData(:,j) = tdata;
#                      readtotal = readtotal + count;
#                  end
#
#                  if (j >= nframes), break; end;
#            """
#
#    #if readtotal != readexpected:
#    #     logging.warning( 'Number of values read not equal to the number expected.')
#
##    EventData = [];
#    if head['eventtypes'] > 0:
#        raise NotImplementedError
##        EventData = TrialData[:, head['n_sensors']:]
##        TrialData = TrialData[:, :head['n_sensors']]
#
#    # convert from A/D units to microvolts
#    if head['bits'] != 0 and head['range'] != 0:
#        raise NotImplementedError
##        TrialData = (head['range']/(2**head['bits']))*TrialData
#
#    # convert event codes to char
#    # ---------------------------
#    #head.eventcode = char(head.eventcode);
#
#
#    #--------------------------- isvector() ---------------
#    #function retval = isvector(V)
#    #s = size(V);
#    #retval = ( length(s) < 3 ) & ( min(s) <= 1 );
#
