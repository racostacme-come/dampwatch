"""DampWatch: auditable linear structural dynamics."""

__version__ = "0.1.0"

from .dynamics import (
    EnergyBudget,
    History,
    Parameters,
    System,
    amplification_matrix,
    energy_budget,
    exact_response,
    integrate,
)

__all__ = [
    "EnergyBudget",
    "History",
    "Parameters",
    "System",
    "amplification_matrix",
    "energy_budget",
    "exact_response",
    "integrate",
]
