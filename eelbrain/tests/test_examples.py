# generated by eelbrain/scripts/make_example_tests.py
import importlib.util
import logging
import os
from pathlib import Path
import re

import mne
import pytest

from eelbrain import configure, conftest
from eelbrain.testing import hide_plots, working_directory


DATASETS = {
    'mne_sample': bool(mne.datasets.sample.data_path(download=False))
}

# find examples
examples_dir = Path(__file__).parents[2] / 'examples'
examples = list(examples_dir.glob('*/*.py'))


@hide_plots
@pytest.mark.parametrize("path", examples)
def test_example(tmp_path, path: Path):
    "Run the example script at ``filename``"
    # check for flags
    text = path.read_text()
    if re.findall("^# skip test:", text, re.MULTILINE):
        return
    # check for required modules
    required_modules = re.findall(r"^# requires: (\w+)$", text, re.MULTILINE)
    for module in required_modules:
        try:
            importlib.import_module(module)
        except ImportError:
            pytest.skip(f"required module {module} not available")
    if conftest.SKIP_MAYAVI and 'mayavi' in required_modules:
        pytest.skip("requires mayavi")
    # check for required datasets
    required_datasets = re.findall(r"^# dataset: (\w+)$", text, re.MULTILINE)
    for dataset in required_datasets:
        if not DATASETS[dataset]:
            raise pytest.skip(f"required dataset {dataset} not available")
    # set up context
    configure(show=False)
    with working_directory(tmp_path):
        temp_dir = Path(tmp_path)
        # link files
        for file in path.parent.glob('*.*'):
            if file.name.startswith(('.', '_')):
                continue
            elif file.name in text:
                os.link(file, temp_dir / file.name)
        # reduce computational load
        text = text.replace("n_samples = 1000", "n_samples = 2")
        # prepare example script
        exa_path = temp_dir / path.name
        exa_path.write_text(text)
        logging.info(" Executing %s/%s", path.parent.name, path.name)
        # execute example
        spec = importlib.util.spec_from_file_location(exa_path.stem, exa_path)
        example_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(example_module)
