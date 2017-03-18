from surfer import Brain

DST = '../images/brain_%s.png'
subjects_dir = '/Volumes/Seagate/refpred/mri'
surfaces = ('inflated',
            'inflated_avg',
            'inflated_pre',
            # 'orig',
            # 'pial',
            'smoothwm',
            'sphere',
            'white')


for surf in surfaces:
    brain = Brain('fsaverage', 'lh', surf, size=400, background='white',
                  subjects_dir=subjects_dir)
    brain.save_image(DST % surf, 'rgba', True)
