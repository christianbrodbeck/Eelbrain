# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""
Reuires https://github.com/maenu/scholar.py::

    pip install https://github.com/maenu/scholar.py/archive/master.zip

"""
from argparse import ArgumentParser
from pathlib import Path
from string import ascii_lowercase
import re

from pybtex.database import BibliographyData, parse_bytes, parse_file
from scholar.scholar import ScholarQuerier, ScholarSettings, SearchScholarQuery


DIR = Path(__file__).absolute().parent
CACHE = DIR / 'cache.bib'
DST = DIR.parent / 'publications.bib'
IGNORE = """
vo2014cytotoxicity
takeilnatriureticpeptideisolatedfromeelbrain
""".split()
BIORXIV_OBSOLETE = """
326785
633834
""".split()
ACRONYMS = ['EEG', 'MEG', 'MRI']


def download():
    querier = ScholarQuerier()
    settings = ScholarSettings()
    settings.set_citation_format(ScholarSettings.CITFORM_BIBTEX)
    querier.apply_settings(settings)
    query = SearchScholarQuery()
    query.set_phrase("eelbrain")
    query.set_timeframe(2012, None)
    query.set_include_patents(False)
    # download entries
    bib = BibliographyData()
    start = 0
    while True:
        querier.send_query(query)
        if len(querier.articles) == 0:
            break
        # extract citation data
        for article in querier.articles:
            querier.get_citation_data(article)
            # parse entry
            data = parse_bytes(article.citation_data, 'bibtex')
            assert len(data.entries) == 1
            for key, entry in data.entries.items():
                # make sure URL is present
                if 'url' not in entry.fields:
                    url = article.attrs['url'][0]
                    if url:
                        entry.fields['url'] = url
                # store
                bib.add_entry(key, entry)
        # next page
        start += 10
        query.set_start(start)
    # write to file
    CACHE.write_bytes(bib.to_bytes('bibtex').replace(br'\\&', br'\&'))


def parse():
    src_bib = parse_file(CACHE, 'bibtex')
    bib = parse_file(DST, 'bibtex')
    for entry in src_bib.entries.values():
        if entry.key in IGNORE:
            continue
        elif entry.type != 'article':
            continue
        journal = entry.fields['journal'].lower()
        if journal == 'biorxiv' and entry.fields['pages'] in BIORXIV_OBSOLETE:
            continue
        elif entry.key in bib.entries:
            if journal == bib.entries[entry.key].fields['journal'].lower():
                continue
        # fix title
        for repl in ACRONYMS:
            entry.fields['title'] = re.sub(repl, '{' + repl + '}', entry.fields['title'])
        # URL is redundant with DOI
        if 'doi' in entry.fields:
            entry.fields.pop('url', None)

        key = entry.key
        for c in ascii_lowercase:
            if key in bib.entries:
                key = f'{entry.key}_{c}'
            else:
                break
        bib.add_entry(key, entry)
    DST.write_bytes(bib.to_bytes('bibtex').replace(br'\\&', br'\&'))


if __name__ == '__main__':
    parser = ArgumentParser(description="Find publications that used Eelbrain on Google Scholar")
    parser.add_argument('task', choices=('download', 'parse'))
    args = parser.parse_args()
    if args.task == 'download':
        download()
    else:
        parse()
