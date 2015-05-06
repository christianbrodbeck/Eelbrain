__author__ = 'lauragwilliams'

from eelbrain import *
import mne
import numpy as np
import os
import datetime


def average_activity_R(e,time,parc,label=None,average_ms='average',group='all',temp_dir = 'temp/'):

    """
    Compute average activity over a spatial window; either averaged over time (average_ms = "average") or for each ms of a window (average_ms = "ms").


    Parameters:
    ----------

    e               :               experiment class object
    time            :               tuple, the window for analysis in secs for example: (0.15, 0.2)
    parc            :               parcellation for the spatial window
    label           :               label within the parc of interest (if None, takes first label from parc)
    average_ms      :               ('average','ms'), determine if the output is an average over the time window, or gets activity for each ms within the timewindow
    group           :               the group to analyse (default = all)
    temp_dir        :               where the files will be saved (default is to create a "temp" folder in current dir)

    """

    e.set(group = group)

    dses = []
    activities = []

    if label == None:

        parc_labels = mne.read_labels_from_annot('fsaverage',parc,subjects_dir=e.get('mri-sdir'))
        label = parc_labels[0]

    for subject in ['A0125']: # keep this one subject for now, while we test it out

        ds = e.load_epochs_stc(subject)

        # get the size of the data for use in the by -ms set
        decim = e.epochs.get('epoch').get('decim')
        n_epochs = len(ds[ds.keys()[0]])
        n_ms = int(((time[1]-time[0])*1000)/decim)

        shape = (n_ms,n_epochs)

        empty_array = np.zeros(shape=shape)

        for i in xrange(len(ds[ds.keys()[0]])):

            item = ds['item'][i]

            ds_sub = ds.sub("item == '%s'" % item)
            src = ds_sub['src']
            src.source.set_parc(parc)
            src_region = src.sub(source=label.name)

            ds_sub['src'] = src_region
            timecourse = ds_sub['src'].mean('source')
            timecourse_sub = timecourse.sub(time=time)

            # if average, get average activity over time
            if average_ms == 'average':

                average = timecourse_sub.mean('time')
                activities.append(average)

            # if ms, get the activity ms by ms within the timewindow
            elif average_ms == 'ms':

                activities = empty_array

                data = timecourse_sub.x[0]

                # insert activity for each epoch, and for each ms
                for n in xrange(len(data)):
                    activities[n,i] = data[n]

        # for each time-point, add a column to the dataset with the activity for each epoch
        if average_ms == 'average':

            ds['dSPM_av'] = activities

        elif average_ms == 'ms':

            for i in xrange(len(activities)):

                n = i * decim

                ds['dSPM_%i' %n] = Var(activities[i])

        # create a temporary folder, and save each subject's datafile in there, so we can remove it from RAM

        if not os.path.isdir(temp_dir):
            os.makedirs(temp_dir)

        # get the date to append to the filename
        today = datetime.date.today()

        ds.save_txt(path=temp_dir + today + '_ds_%s_%s_%s_%s' % (subject,average_ms,label.name,time), delim=',')

        print subject

        # clean out ds
        ds = 0


def load_temp_ds(temp_dir='temp/',save=False,file_path=None):
    """

    Loads ds from temp folder


    Parameters:
    ----------

    temp_dir        :               location of ds
    save            :               bool, whether to save the .csv to file or not
    file_path       :               if save == True, specify destination
    """

    dses = []

    for file in os.listdir(temp_dir):
        path = temp_dir + file
        ds = load.txt.tsv(path, names=True, delimiter=',', ignore_missing=True)
        dses.append(ds)

    average_ds = combine((dses))

    return average_ds

    if save == True:

        average_ds.save_txt(file_path + 'averaged_%s_%s_%s.csv' % (label.name,time,average_ms),delim=',')

    return average_ds