"""Neural network subpackage for STAVKI models."""

from .multitask import MultiTaskModel
from .goals_regressor import GoalsRegressor

__all__ = ["MultiTaskModel", "GoalsRegressor"]
