# Helper to download data for Alice example
import hashlib
import shutil
from pathlib import Path
from urllib.request import urlretrieve
import zipfile

from pooch import file_hash

from .._types import PathArg


def get_alice_path(
        path: PathArg = Path("~/Data/Alice"),
):
    md5 = hashlib.md5()

    path = Path(path).expanduser().resolve()
    if path.exists():
        return path
    path.mkdir(exist_ok=True, parents=True)
    urls = [
        ['https://drum.lib.umd.edu/bitstream/handle/1903/27591/stimuli.zip', '4336a47bef7d3e63239c40c0623dc186'],
        # ['https://drum.lib.umd.edu/bitstream/handle/1903/27591/eeg.0.zip', 'd63d96a6e5080578dbf71320ddbec0a0'],
        ['https://drum.lib.umd.edu/bitstream/handle/1903/27591/eeg.1.zip', 'bdc65f168db4c0f19bb0fed20eae129b'],  # S15-S34
        # ['https://drum.lib.umd.edu/bitstream/handle/1903/27591/eeg.2.zip', '3fb33ca1c4640c863a71bddd45006815'],
    ]

    for url, hash in urls:
        temp_file_path, header = urlretrieve(url)
        # with open(temp_file_path, "rb") as f:
        #     for chunk in iter(lambda: f.read(1048576), b""):
        #         md5.update(chunk)
        # file_hash = md5.hexdigest()
        # if file_hash != hash:
        #     raise RuntimeError(f'Hash mismatch for {url}: {file_hash} != {hash}')
        with zipfile.ZipFile(temp_file_path, 'r') as f:
            f.extractall(path)
        Path(temp_file_path).unlink()
    return path
