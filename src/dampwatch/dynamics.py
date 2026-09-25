"""Dense, constant-matrix, linear generalized-alpha integration.

Weights use x_(n+1-alpha) = (1-alpha)*x_(n+1) + alpha*x_n.
Loads are interpolated from endpoint samples with the same alpha_f weight.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cho_factor, cho_solve, eigvalsh, expm

Array = NDArray[np.float64]
Force = Callable[[float], ArrayLike]


def _array(value: ArrayLike, name: str) -> Array:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    result = np.array(value, dtype=float, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _vector(value: ArrayLike, size: int, name: str) -> Array:
    result = _array(value, name)
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    return result


@dataclass(frozen=True)
class System:
    """Real symmetric matrices: M positive definite, K and C semidefinite.

    Inputs are copied and made read-only. Tiny roundoff-level asymmetry is
    symmetrized. Semidefiniteness allows a relative 1e-12 eigenvalue tolerance.
    Units must be mutually consistent; no unit conversion is performed.
    """

    mass: ArrayLike
    damping: ArrayLike
    stiffness: ArrayLike

    def __post_init__(self) -> None:
        size = None
        for name in ("mass", "damping", "stiffness"):
            value = _array(getattr(self, name), name)
            if value.ndim != 2 or value.shape[0] == 0 or value.shape[0] != value.shape[1]:
                raise ValueError(f"{name} must be a nonempty square matrix")
            if size is not None and value.shape != (size, size):
                raise ValueError("all matrices must have the same shape")
            size = value.shape[0]
            scale = max(float(np.max(np.abs(value))), np.finfo(float).tiny)
            if np.max(np.abs(value - value.T)) > 1e-12 * scale:
                raise ValueError(f"{name} must be symmetric")
            value = 0.5 * (value + value.T)
            if name == "mass":
                try:
                    cho_factor(value)
                except np.linalg.LinAlgError as error:
                    raise ValueError("mass must be positive definite") from error
            elif eigvalsh(value)[0] < -1e-12 * scale:
                raise ValueError(f"{name} must be positive semidefinite")
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def size(self) -> int:
        """Number of unconstrained displacement degrees of freedom."""
        return self.mass.shape[0]


@dataclass(frozen=True)
class Parameters:
    """Chung-Hulbert parameters for an asymptotic spectral radius in [0, 1]."""

    rho_inf: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.rho_inf) or not 0 <= self.rho_inf <= 1:
            raise ValueError("rho_inf must be finite and in [0, 1]")

    @property
    def alpha_m(self) -> float:
        return (2 * self.rho_inf - 1) / (self.rho_inf + 1)

    @property
    def alpha_f(self) -> float:
        return self.rho_inf / (self.rho_inf + 1)

    @property
    def gamma(self) -> float:
        return 0.5 + self.alpha_f - self.alpha_m

    @property
    def beta(self) -> float:
        return 0.25 * (1 + self.alpha_f - self.alpha_m) ** 2


@dataclass
class History:
    """Endpoint states; state/load arrays have shape (steps + 1, ndof)."""

    time: Array
    displacement: Array
    velocity: Array
    acceleration: Array
    force: Array


@dataclass
class EnergyBudget:
    """Endpoint energy, accumulated midpoint damping/work, and signed defect.

    defect = mechanical - mechanical[0] + dissipated - work.
    A negative defect indicates additional energy loss in this discrete audit.
    For rho_inf < 1 this is not a guaranteed monotone algorithmic energy.
    """

    mechanical: Array
    dissipated: Array
    work: Array
    defect: Array


class _Stepper:
    def __init__(self, system: System, dt: float, parameters: Parameters):
        self.s, self.h, self.p = system, dt, parameters
        p = parameters
        effective = (
            (1 - p.alpha_m) * system.mass
            + (1 - p.alpha_f) * p.gamma * dt * system.damping
            + (1 - p.alpha_f) * p.beta * dt**2 * system.stiffness
        )
        self.factor = cho_factor(effective)

    def step(self, u: Array, v: Array, a: Array, f0: Array, f1: Array):
        s, h, p = self.s, self.h, self.p
        up = u + h * v + h**2 * (0.5 - p.beta) * a
        vp = v + h * (1 - p.gamma) * a
        rhs = (
            (1 - p.alpha_f) * f1
            + p.alpha_f * f0
            - p.alpha_m * (s.mass @ a)
            - s.damping @ ((1 - p.alpha_f) * vp + p.alpha_f * v)
            - s.stiffness @ ((1 - p.alpha_f) * up + p.alpha_f * u)
        )
        anew = cho_solve(self.factor, rhs)
        return up + p.beta * h**2 * anew, vp + p.gamma * h * anew, anew


def integrate(
    system: System,
    displacement: ArrayLike,
    velocity: ArrayLike,
    *,
    dt: float,
    steps: int,
    rho_inf: float = 1.0,
    force: Force | None = None,
) -> History:
    """Integrate from t=0 on a fixed grid; compute consistent initial acceleration.

    force(t) must return one finite real vector, and is sampled once per endpoint.
    The effective matrix is factored once. Zero steps returns the initial state.
    """
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    if not np.isfinite(dt * steps):
        raise ValueError("final time must be finite")
    p = Parameters(rho_inf)
    time = np.arange(steps + 1, dtype=float) * dt
    shape = (steps + 1, system.size)
    u, v, a = (np.empty(shape) for _ in range(3))
    u[0] = _vector(displacement, system.size, "displacement")
    v[0] = _vector(velocity, system.size, "velocity")
    f = np.zeros(shape)
    if force is not None:
        for i, t in enumerate(time):
            f[i] = _vector(force(float(t)), system.size, "force")
    a[0] = cho_solve(
        cho_factor(system.mass), f[0] - system.damping @ v[0] - system.stiffness @ u[0]
    )
    stepper = _Stepper(system, dt, p)
    for i in range(steps):
        u[i + 1], v[i + 1], a[i + 1] = stepper.step(u[i], v[i], a[i], f[i], f[i + 1])
    if not all(np.all(np.isfinite(x)) for x in (u, v, a)):
        raise FloatingPointError("integration produced nonfinite states; check scaling")
    return History(time, u, v, a, f)


def energy_budget(system: System, history: History) -> EnergyBudget:
    """Audit endpoint work and physical damping with midpoint quadrature.

    Accepts a History produced by integrate; changing its arrays is the caller's
    responsibility. Force work uses mean endpoint force dotted with displacement
    increment. Damping uses dt * mean_velocity.T @ C @ mean_velocity.
    """
    u, v = history.displacement, history.velocity
    mechanical = 0.5 * (
        np.einsum("ni,ij,nj->n", u, system.stiffness, u)
        + np.einsum("ni,ij,nj->n", v, system.mass, v)
    )
    vmid = 0.5 * (v[1:] + v[:-1])
    fmid = 0.5 * (history.force[1:] + history.force[:-1])
    dissipated = np.r_[
        0.0,
        np.cumsum(np.diff(history.time) * np.einsum("ni,ij,nj->n", vmid, system.damping, vmid)),
    ]
    work = np.r_[0.0, np.cumsum(np.einsum("ni,ni->n", fmid, np.diff(u, axis=0)))]
    return EnergyBudget(
        mechanical, dissipated, work, mechanical - mechanical[0] + dissipated - work
    )


def exact_response(
    system: System,
    displacement: ArrayLike,
    velocity: ArrayLike,
    times: ArrayLike,
    *,
    force: ArrayLike | None = None,
) -> tuple[Array, Array]:
    """Independent matrix-exponential reference for constant force (default zero).

    Uses an augmented state [u, v, 1], allowing rigid modes and non-proportional
    damping without inverting K. Each requested time is evaluated independently.
    'Exact' means exact in time up to floating-point matrix-exponential error.
    """
    n = system.size
    u0 = _vector(displacement, n, "displacement")
    v0 = _vector(velocity, n, "velocity")
    t = _array(times, "times")
    if t.ndim != 1 or np.any(t < 0):
        raise ValueError("times must be a one-dimensional nonnegative array")
    f = np.zeros(n) if force is None else _vector(force, n, "force")
    operator = np.zeros((2 * n + 1, 2 * n + 1))
    operator[:n, n : 2 * n] = np.eye(n)
    factor = cho_factor(system.mass)
    operator[n : 2 * n, :n] = -cho_solve(factor, system.stiffness)
    operator[n : 2 * n, n : 2 * n] = -cho_solve(factor, system.damping)
    operator[n : 2 * n, -1] = cho_solve(factor, f)
    initial = np.r_[u0, v0, 1.0]
    values = np.array([expm(operator * ti) @ initial for ti in t]).reshape(-1, 2 * n + 1)
    return values[:, :n], values[:, n : 2 * n]


def amplification_matrix(omega_dt: float, rho_inf: float = 1.0) -> Array:
    """Full 3x3 map on [u, dt*v, dt**2*a] for an undamped oscillator.

    All three independent basis states are propagated, including the parasitic
    acceleration mode. This avoids falsely estimating rho_inf from a 2x2 map.
    """
    if not np.isfinite(omega_dt) or omega_dt < 0:
        raise ValueError("omega_dt must be finite and nonnegative")
    if omega_dt > np.sqrt(np.finfo(float).max):
        raise ValueError("omega_dt is too large to square")
    system = System([[1.0]], [[0.0]], [[omega_dt**2]])
    stepper = _Stepper(system, 1.0, Parameters(rho_inf))
    zero = np.zeros(1)
    columns = []
    for column in np.eye(3):
        columns.append(np.concatenate(stepper.step(*column.reshape(3, 1), zero, zero)))
    return np.column_stack(columns)
