"""Various functions from migrating from older to newer MneExperiment versions"""
from os.path import join


def update_rej_files():
    """Update rej-file path (remove session name, add subject name)"""
    old_temp = join('{rej-dir}', '{session}_{sns_kind}_{epoch}-{rej}.pickled')
