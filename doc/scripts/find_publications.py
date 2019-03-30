# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""
Reuires https://github.com/maenu/scholar.py
"""
from pathlib import Path
from string import ascii_lowercase
import re

from pybtex.database import parse_bytes, parse_file
from scholar import ScholarQuerier, ScholarSettings, SearchScholarQuery


DST = Path(__file__).absolute().parents[1] / 'publications.bib'
IGNORE = """
vo2014cytotoxicity
takeilnatriureticpeptideisolatedfromeelbrain
matchintemporal
brodbeck2018transformation
""".split()
ACRONYMS = ['EEG', 'MEG', 'MRI']

querier = ScholarQuerier()
settings = ScholarSettings()
settings.set_citation_format(ScholarSettings.CITFORM_BIBTEX)
querier.apply_settings(settings)
query = SearchScholarQuery()
query.set_phrase("eelbrain")
query.set_timeframe(2012, None)
query.set_include_patents(False)


bib = parse_file(DST, 'bibtex')
start = 0
while True:
    querier.send_query(query)
    if len(querier.articles) == 0:
        break
    # extract articles
    for article in querier.articles:
        querier.get_citation_data(article)
        # convert to pybtex entry
        data = parse_bytes(article.citation_data, 'bibtex')
        assert len(data.entries) == 1
        for entry in data.entries.values():
            if entry.key in IGNORE:
                continue
            elif entry.type != 'article':
                continue
            elif entry.key in bib.entries:
                if entry.fields['journal'] == bib.entries[entry.key].fields['journal']:
                    continue
            # fix title
            for repl in ACRONYMS:
                entry.fields['title'] = re.sub(repl, '{' + repl + '}', entry.fields['title'])
            # add info
            if 'url' not in entry.fields:
                url = article.attrs['url'][0]
                if url:
                    entry.fields['url'] = url

            key = entry.key
            for c in ascii_lowercase:
                if key in bib.entries:
                    key = f'{entry.key}_{c}'
                else:
                    break
            bib.add_entry(key, entry)
    # next page
    start += 10
    query.set_start(start)

DST.write_bytes(bib.to_bytes('bibtex'))
