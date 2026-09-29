"""Regenerate the manuscript figure from committed numerical CSV outputs."""

import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent


def read(name):
    with (ROOT / "results" / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def values(rows, key):
    if not rows:
        raise ValueError(f"No recorded rows selected for {key}")
    return np.array([float(row[key]) for row in rows])


plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(6.35, 2.35), layout="constrained")
rows = read("convergence.csv")
axes[0].loglog(values(rows, "dt_s"), values(rows, "relative_state_error"), "o-")
axes[0].set(title="State refinement", xlabel="Time step (s)", ylabel="Relative state error")
rows = read("free_response.csv")
for key, label in (
    ("rho05_low_mode_energy_J", "low mode / initial"),
    ("rho05_high_mode_energy_J", "high mode / initial"),
):
    e = values(rows, key)
    axes[1].semilogy(values(rows, "time_s"), np.maximum(e / e[0], 1e-16), label=label)
axes[1].set(title="Filtering at rho = 0.5", xlabel="Time (s)", ylabel="Modal energy fraction")
axes[1].legend(fontsize=7)

for ax in axes:
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    if ax.get_xscale() == "log":
        points = np.unique(ax.lines[0].get_xdata())
        if 2 <= len(points) <= 6:
            ax.set_xticks(points, labels=[f"{x:.3g}" for x in points])
    ax.grid(alpha=0.2, which="both")
fig.savefig(ROOT / "paper" / "figure.pdf")
plt.close(fig)
