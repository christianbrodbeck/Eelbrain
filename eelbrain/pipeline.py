from ._utils import _deprecated_alias

from ._experiment.mne_experiment import MneExperiment
from ._experiment.preprocessing import RawSource, RawFilter, RawICA, RawMaxwell, RawReReference, RawApplyICA
from ._experiment.epochs import ContinuousEpoch, EpochCollection, PrimaryEpoch, SecondaryEpoch, SuperEpoch
from ._experiment.groups import Group, SubGroup
from ._experiment.parc import SubParc, CombinationParc, FreeSurferParc, FSAverageParc, SeededParc, IndividualSeededParc
from ._experiment.test_def import ANOVA, TTestOneSample, TTestIndependent, TTestRelated, TContrastRelated, TwoStageTest, ROITestResult, ROI2StageResult
from ._experiment.variable_def import EvalVar, GroupVar, LabelVar

# backwards compatibility
TTestInd = _deprecated_alias('TTestInd', TTestIndependent, '0.34')
TTestRel = _deprecated_alias('TTestRel', TTestRelated, '0.34')
TContrastRel = _deprecated_alias('TContrastRel', TContrastRelated, '0.34')
