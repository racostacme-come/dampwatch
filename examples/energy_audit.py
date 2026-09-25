"""A damped oscillator with a closed discrete energy budget."""

import numpy as np

from dampwatch import System, energy_budget, integrate

system = System(mass=[[2.0]], damping=[[0.4]], stiffness=[[8.0]])
history = integrate(system, [1.0], [0.0], dt=0.01, steps=1000, rho_inf=1.0)
budget = energy_budget(system, history)
defect = np.max(np.abs(budget.defect))
print(f"Final mechanical energy: {budget.mechanical[-1]:.8f} J")
print(f"Accumulated physical dissipation: {budget.dissipated[-1]:.8f} J")
print(f"Maximum discrete energy defect: {defect:.3e} J")
assert defect < 1e-11
