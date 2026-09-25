"""Deterministic synthetic experiment and analytical acceptance checks."""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from .dynamics import System, amplification_matrix, energy_budget, exact_response, integrate


def two_mass_system(*, damped: bool = False) -> System:
    """1 kg primary mass, 0.15 kg attachment, 4 and 600 N/m springs.

    Ground -- soft spring -- primary -- stiff spring -- attachment.
    Optional Rayleigh damping C = 0.06 M + 0.0001 K in SI units.
    """
    mass = np.diag([1.0, 0.15])
    stiffness = np.array([[604.0, -600.0], [-600.0, 600.0]])
    damping = 0.06 * mass + 0.0001 * stiffness if damped else np.zeros((2, 2))
    return System(mass, damping, stiffness)


def _csv(path: Path, header: list[str], rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def run_study(output: str | Path, *, plot: bool = True) -> dict:
    """Write a repeatable report; fail if numerical acceptance checks fail.

    Existing report files at output are replaced. No timing claims or random
    seeds are involved. Numeric data are reproducible within roundoff across
    supported platforms; image bytes can vary with plotting-library versions.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    system = two_mass_system()
    eigenvalues, modes = eigh(system.stiffness, system.mass)
    omega = np.sqrt(eigenvalues)
    # Fix eigenvector sign for platform-independent modal coordinates.
    for j in range(2):
        if modes[0, j] < 0:
            modes[:, j] *= -1
    q0 = np.array([1.0, 0.6]) / omega
    u0, v0 = modes @ q0, np.zeros(2)
    histories = {
        rho: integrate(system, u0, v0, dt=0.05, steps=240, rho_inf=rho) for rho in (1.0, 0.5)
    }
    time = histories[1.0].time
    exact_u, exact_v = exact_response(system, u0, v0, time)
    budgets = {rho: energy_budget(system, history) for rho, history in histories.items()}
    modal_u = {
        rho: history.displacement @ system.mass @ modes for rho, history in histories.items()
    }
    modal_v = {
        rho: history.velocity @ system.mass @ modes for rho, history in histories.items()
    }
    modal_energy = {
        rho: 0.5 * (modal_v[rho] ** 2 + omega**2 * modal_u[rho] ** 2) for rho in histories
    }
    initial_energy = float(budgets[1.0].mechanical[0])
    exact_energy = 0.5 * (
        np.einsum("ni,ij,nj->n", exact_u, system.stiffness, exact_u)
        + np.einsum("ni,ij,nj->n", exact_v, system.mass, exact_v)
    )
    _csv(
        output / "free_response.csv",
        [
            "time_s",
            "exact_primary_m",
            "rho1_primary_m",
            "rho05_primary_m",
            "rho1_energy_J",
            "rho05_energy_J",
            "rho05_low_mode_energy_J",
            "rho05_high_mode_energy_J",
        ],
        zip(
            time,
            exact_u[:, 0],
            histories[1.0].displacement[:, 0],
            histories[0.5].displacement[:, 0],
            budgets[1.0].mechanical,
            budgets[0.5].mechanical,
            modal_energy[0.5][:, 0],
            modal_energy[0.5][:, 1],
            strict=True,
        ),
    )

    damped = two_mass_system(damped=True)
    forced = integrate(
        damped, u0, v0, dt=0.02, steps=600, force=lambda t: [0.8 * np.sin(1.2 * t), 0.0]
    )
    forced_budget = energy_budget(damped, forced)
    _csv(
        output / "forced_budget.csv",
        ["time_s", "mechanical_J", "physical_dissipation_J", "external_work_J", "defect_J"],
        zip(
            forced.time,
            forced_budget.mechanical,
            forced_budget.dissipated,
            forced_budget.work,
            forced_budget.defect,
            strict=True,
        ),
    )

    # Independent reference at common sample times removes changing-grid bias.
    sample_time = np.linspace(0, 2, 101)
    reference_u, reference_v = exact_response(damped, u0, v0, sample_time)
    reference_norm = np.sqrt(2 * initial_energy)
    convergence = []
    previous = None
    for dt in (0.004, 0.002, 0.001, 0.0005):
        history = integrate(damped, u0, v0, dt=dt, steps=round(2 / dt), rho_inf=0.5)
        indices = np.rint(sample_time / dt).astype(int)
        du = history.displacement[indices] - reference_u
        dv = history.velocity[indices] - reference_v
        error = float(
            np.max(
                np.sqrt(
                    np.einsum("ni,ij,nj->n", du, damped.stiffness, du)
                    + np.einsum("ni,ij,nj->n", dv, damped.mass, dv)
                )
            )
            / reference_norm
        )
        order = None if previous is None else float(np.log2(previous / error))
        convergence.append({"dt_s": dt, "relative_state_error": error, "order": order})
        previous = error
    _csv(
        output / "convergence.csv",
        ["dt_s", "relative_state_error", "order"],
        ([r["dt_s"], r["relative_state_error"], r["order"]] for r in convergence),
    )

    frequencies = np.logspace(-3, 2, 160)
    spectra = {
        rho: np.array(
            [max(abs(np.linalg.eigvals(amplification_matrix(x, rho)))) for x in frequencies]
        )
        for rho in (0.0, 0.5, 0.8, 1.0)
    }
    _csv(
        output / "spectral_radius.csv",
        ["omega_dt", "rho0", "rho05", "rho08", "rho1"],
        zip(frequencies, *spectra.values(), strict=True),
    )
    high_limit = float(max(abs(np.linalg.eigvals(amplification_matrix(1e7, 0.5)))))
    retention = modal_energy[0.5][-1] / modal_energy[0.5][0]
    metrics = {
        "undamped_rho1_max_relative_energy_drift": float(
            np.max(np.abs(budgets[1.0].mechanical / initial_energy - 1))
        ),
        "exact_max_relative_energy_drift": float(
            np.max(np.abs(exact_energy / initial_energy - 1))
        ),
        "rho05_low_mode_energy_retention": float(retention[0]),
        "rho05_high_mode_energy_retention": float(retention[1]),
        "forced_rho1_max_normalized_budget_defect": float(
            np.max(np.abs(forced_budget.defect)) / initial_energy
        ),
        "rho05_high_frequency_spectral_radius": high_limit,
        "max_sampled_spectral_radius": float(max(np.max(x) for x in spectra.values())),
        "finest_observed_order": convergence[-1]["order"],
    }
    checks = {
        "undamped_energy_conserved": metrics["undamped_rho1_max_relative_energy_drift"] < 1e-10,
        "reference_energy_conserved": metrics["exact_max_relative_energy_drift"] < 1e-9,
        "forced_discrete_budget_closes": metrics["forced_rho1_max_normalized_budget_defect"]
        < 1e-10,
        "high_mode_filtered": retention[1] < 1e-6,
        "low_mode_retained": retention[0] > 0.98,
        "second_order_recovered": 1.9 < convergence[-1]["order"] < 2.1,
        "high_frequency_limit_matches": abs(high_limit - 0.5) < 1e-4,
        "sampled_spectral_stability": metrics["max_sampled_spectral_radius"] < 1 + 2e-12,
    }
    report = {
        "schema_version": 1,
        "experiment": "two-mass stiffness contrast and discrete energy audit",
        "units": "SI: kg, m, s, N, J",
        "model": {
            "mass": system.mass.tolist(),
            "stiffness": system.stiffness.tolist(),
            "damped_C": damped.damping.tolist(),
            "omega_rad_s": omega.tolist(),
            "initial_displacement_m": u0.tolist(),
            "initial_velocity_m_s": v0.tolist(),
        },
        "free_study": {"dt_s": 0.05, "steps": 240, "rho_inf": [1.0, 0.5]},
        "forced_study": {
            "dt_s": 0.02,
            "steps": 600,
            "rho_inf": 1.0,
            "force_N": "[0.8*sin(1.2*t), 0]",
        },
        "metrics": metrics,
        "convergence": convergence,
        "checks": {key: bool(value) for key, value in checks.items()},
        "interpretation": "Signed budget defect includes discretization and quadrature error; "
        "it is not a monotone generalized-alpha algorithmic energy.",
    }
    (output / "validation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            f"Validation failed: {', '.join(failed)}; see {output / 'validation.json'}"
        )
    if plot:
        _plot(
            output,
            histories,
            modal_energy,
            q0,
            omega,
            budgets,
            exact_u,
            forced,
            forced_budget,
            convergence,
            frequencies,
            spectra,
        )
    return report


def _plot(
    output,
    histories,
    modal_energy,
    q0,
    omega,
    budgets,
    exact_u,
    forced,
    forced_budget,
    convergence,
    frequencies,
    spectra,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), layout="constrained")
    fig.suptitle("DampWatch | Where vibration energy goes", fontsize=21, weight="bold")
    time = histories[1.0].time
    colors = {1.0: "#206c9f", 0.5: "#d86630"}
    axes[0, 0].plot(time, exact_u[:, 0], color="#202e37", lw=1.5, label="Exact reference")
    for rho, history in histories.items():
        axes[0, 0].plot(
            time,
            history.displacement[:, 0],
            color=colors[rho],
            alpha=0.8,
            lw=1,
            label=rf"$\rho_\infty={rho}$",
        )
        axes[0, 1].plot(
            time, budgets[rho].mechanical, color=colors[rho], label=rf"$\rho_\infty={rho}$"
        )
    axes[0, 0].set(
        title="A | Primary mass: phase accuracy still matters",
        xlabel="Time [s]",
        ylabel="Displacement [m]",
    )
    axes[0, 1].axhline(
        budgets[1.0].mechanical[0], color="#202e37", ls=":", label="Exact energy"
    )
    axes[0, 1].set(
        title="B | Free undamped system: numerical energy loss",
        xlabel="Time [s]",
        ylabel="Mechanical energy [J]",
    )
    for j, label in enumerate(("Low mode", "High mode")):
        axes[1, 0].semilogy(
            time,
            np.maximum(modal_energy[0.5][:, j] / (0.5 * omega[j] ** 2 * q0[j] ** 2), 1e-16),
            label=label,
        )
    axes[1, 0].set(
        title=r"C | Selective filtering at $\rho_\infty=0.5$",
        xlabel="Time [s]",
        ylabel="Mode energy / initial mode energy",
        ylim=(1e-16, 2),
    )
    for rho, radius in spectra.items():
        axes[1, 1].semilogx(frequencies, radius, label=rf"$\rho_\infty={rho}$")
    axes[1, 1].set(
        title="D | Full three-state amplification spectrum",
        xlabel=r"$\omega\Delta t$",
        ylabel="Spectral radius",
        ylim=(0, 1.05),
    )
    b = forced_budget
    axes[2, 0].plot(forced.time, b.mechanical - b.mechanical[0], label=r"$E-E_0$")
    axes[2, 0].plot(
        forced.time, b.work - b.dissipated, ls="--", label=r"$W-D$", color="#d86630"
    )
    axes[2, 0].set(
        title=r"E | Forced damped budget at $\rho_\infty=1$",
        xlabel="Time [s]",
        ylabel="Energy change [J]",
    )
    dt = np.array([r["dt_s"] for r in convergence])
    error = np.array([r["relative_state_error"] for r in convergence])
    axes[2, 1].loglog(dt, error, "o-", color="#206c9f", label="State error")
    axes[2, 1].loglog(
        dt, error[-1] * (dt / dt[-1]) ** 2, ":", color="#d86630", label="Second order"
    )
    axes[2, 1].set(
        title="F | Refinement against matrix exponential",
        xlabel=r"$\Delta t$ [s]",
        ylabel="Relative energy-norm state error",
    )
    for ax in axes.flat:
        ax.grid(alpha=0.18)
        ax.legend(fontsize=9)
    fig.savefig(output / "damping_audit.png", dpi=160)
    plt.close(fig)
