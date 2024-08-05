from pathlib import Path
from typing import Sequence, Tuple, Union


# https://matplotlib.org/stable/users/explain/colors/colors.html
ColorArg = Union[str, Sequence[float], Tuple[str, float]]
PathArg = Union[Path, str]
