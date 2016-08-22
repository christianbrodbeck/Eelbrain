import cProfile
from argparse import ArgumentParser
import pstats

from eelbrain import datasets
from eelbrain._stats.boosting import boosting


CACHE = '.boosting.profile'
statement = """res = boosting(ds['y'], ds['x1'], 0, 1)"""


if __name__ == '__main__':
    # option parser
    parser = ArgumentParser(description="See "
                            "https://docs.python.org/2/library/profile.html")
    parser.add_argument("--sort", default='tottime',
                        help="Sort the profile entries according to CRITERION")
    parser.add_argument("-n", type=int, default=20,
                        help="Display NUMBER entries from the profile.")
    args = parser.parse_args()

    # run profile
    ds = datasets._get_continuous()
    cProfile.run(statement, CACHE)

    # display stats
    p = pstats.Stats(CACHE)
    p.strip_dirs()
    p.sort_stats(args.sort)
    p.print_stats(args.n)
