import re


COMBINATION_PARC = 'combination'
FS_PARC = 'subject_parc'  # Parcellation that come with every MRI-subject
FSA_PARC = 'fsaverage_parc'  # Parcellation that comes with fsaverage
SEEDED_PARC = 'seeded'
SEEDED_PARC_RE = re.compile('(\w+)-(\d+)$')


class Parcellation(object):
    make = False
    morph_from_fsaverage = False

    def __init__(self, name, views=None):
        self.name = name
        self.views = views

    def as_dict(self):
        return NotImplemented


class CombinationParcellation(Parcellation):
    "Recombine labels from an existingparcellation"
    make = True

    def __init__(self, name, base, labels, views=None):
        Parcellation.__init__(self, name, views)
        self.base = base
        self.labels = labels

    def as_dict(self):
        return {'kind': COMBINATION_PARC, 'base': self.base,
                'labels': self.labels}


class EelbrainParcellation(Parcellation):
    "Parcellation that has special make rule"
    make = True

    def __init__(self, name, morph_from_fsaverage, views=None):
        Parcellation.__init__(self, name, views)
        self.morph_from_fsaverage = morph_from_fsaverage

    def as_dict(self):
        return {'kind': 'eelbrain_parc'}


class FreeSurferParcellation(Parcellation):
    "Parcellation that comes with FreeSurfer"

    def as_dict(self):
        return {'kind': FS_PARC}


class FSAverageParcellation(Parcellation):
    "Parcellation that comes with FSAverage"
    morph_from_fsaverage = True

    def as_dict(self):
        return {'kind': FSA_PARC}


class SeededParcellation(Parcellation):
    "Parcellation that is grown from seed vertices"
    make = True

    def __init__(self, name, seeds, mask=None, surface='white', views=None):
        Parcellation.__init__(self, name, views)
        self.seeds = seeds
        self.mask = mask
        self.surface = surface

    def as_dict(self):
        return {'kind': SEEDED_PARC, 'seeds': self.seeds,
                'surface': self.surface, 'mask': self.mask}


PARC_CLASSES = {
    COMBINATION_PARC: CombinationParcellation,
    FS_PARC: FreeSurferParcellation,
    FSA_PARC: FSAverageParcellation,
    SEEDED_PARC: SeededParcellation,
}
