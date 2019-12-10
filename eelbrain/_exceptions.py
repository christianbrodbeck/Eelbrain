"""Exceptions used throughout Eelbrain"""


class DefinitionError(Exception):
    "MneExperiment definition error"


class DimensionMismatchError(Exception):
    "Trying to align NDVars with mismatching dimensions"

    @classmethod
    def from_dims_list(cls, message, dims_list):
        unique_dims = []
        for dims in dims_list:
            if any(dims == dims_ for dims_ in unique_dims):
                continue
            else:
                unique_dims.append(dims)
        desc = '\n'.join(map(str, unique_dims))
        return cls(f'{message}\n{desc}')


class WrongDimension(Exception):
    "Dimension that is supported"


class IncompleteModel(Exception):
    "Function requires a fully specified model"


class OldVersionError(Exception):
    "Trying to load a file from a version that is no longer supported"


class ZeroVariance(Exception):
    "Trying to do test on data with zero variance"
