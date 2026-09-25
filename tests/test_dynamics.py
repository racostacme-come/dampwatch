import numpy as np
import pytest
from numpy.testing import assert_allclose

from dampwatch import (
    Parameters,
    System,
    amplification_matrix,
    energy_budget,
    exact_response,
    integrate,
)


def oscillator(damping=0.0):
    return System([[2.0]], [[damping]], [[8.0]])


@pytest.mark.parametrize("rho", [0.0, 0.4, 0.8, 1.0])
def test_second_order_against_analytic_oscillator(rho):
    errors = []
    for steps in (100, 200, 400):
        result = integrate(oscillator(), [1.0], [0.0], dt=2 / steps, steps=steps, rho_inf=rho)
        errors.append(np.max(np.abs(result.displacement[:, 0] - np.cos(2 * result.time))))
    assert np.all(np.log2(np.array(errors[:-1]) / errors[1:]) > 1.9)


@pytest.mark.parametrize("rho", [0.0, 0.4, 0.8, 1.0])
def test_weighted_equilibrium_and_kinematics(rho):
    s = System([[2, 0.2], [0.2, 1]], [[0.3, 0.1], [0.1, 0.4]], [[8, -2], [-2, 3]])
    h = 0.017
    r = integrate(
        s,
        [0.1, -0.3],
        [0.2, 0.5],
        dt=h,
        steps=60,
        rho_inf=rho,
        force=lambda t: [np.sin(3 * t), 2 * t**2],
    )
    p = Parameters(rho)
    u, v, a = r.displacement, r.velocity, r.acceleration

    def weighted(x, alpha):
        return (1 - alpha) * x[1:] + alpha * x[:-1]

    residual = (
        weighted(a, p.alpha_m) @ s.mass
        + weighted(v, p.alpha_f) @ s.damping
        + weighted(u, p.alpha_f) @ s.stiffness
        - weighted(r.force, p.alpha_f)
    )
    assert_allclose(residual, 0, atol=2e-14)
    assert_allclose(
        u[1:], u[:-1] + h * v[:-1] + h**2 * ((0.5 - p.beta) * a[:-1] + p.beta * a[1:])
    )
    assert_allclose(v[1:], v[:-1] + h * ((1 - p.gamma) * a[:-1] + p.gamma * a[1:]))


def test_average_acceleration_energy_conservation():
    s = oscillator()
    r = integrate(s, [1], [0], dt=0.1, steps=3000)
    assert_allclose(energy_budget(s, r).mechanical, 4, atol=3e-12)


def test_forced_damped_discrete_energy_identity():
    s = System([[2, 0.1], [0.1, 1]], [[0.8, -0.1], [-0.1, 0.3]], [[8, -1], [-1, 3]])
    r = integrate(
        s,
        [0.5, -0.2],
        [0, 0.4],
        dt=0.025,
        steps=400,
        force=lambda t: [np.sin(t), np.cos(2 * t)],
    )
    audit = energy_budget(s, r)
    assert_allclose(audit.defect, 0, atol=2e-13)
    assert np.all(np.diff(audit.dissipated) >= 0)
    assert audit.dissipated[-1] > 0.1
    assert abs(audit.work[-1]) > 0.1


@pytest.mark.parametrize("damping", [0, 1.5, 8, 12])
def test_exact_response_matches_closed_form_in_all_damping_regimes(damping):
    t = np.linspace(0, 3, 50)
    u, v = exact_response(oscillator(damping), [1], [0], t)
    z = damping / 4
    if z < 2:
        w = np.sqrt(4 - z * z)
        expected = np.exp(-z * t) * (np.cos(w * t) + z / w * np.sin(w * t))
        expected_v = -4 / w * np.exp(-z * t) * np.sin(w * t)
    elif z == 2:
        expected = np.exp(-2 * t) * (1 + 2 * t)
        expected_v = -4 * t * np.exp(-2 * t)
    else:
        w = np.sqrt(z * z - 4)
        expected = np.exp(-z * t) * (np.cosh(w * t) + z / w * np.sinh(w * t))
        expected_v = -4 / w * np.exp(-z * t) * np.sinh(w * t)
    assert_allclose(u[:, 0], expected, atol=3e-14)
    assert_allclose(v[:, 0], expected_v, atol=3e-14)


def test_rigid_body_constant_force_reference_and_integration():
    s = System([[2]], [[0]], [[0]])
    r = integrate(s, [1], [0.5], dt=0.03, steps=100, rho_inf=0.3, force=lambda t: [4])
    u, v = exact_response(s, [1], [0.5], r.time, force=[4])
    assert_allclose(u[:, 0], 1 + 0.5 * r.time + r.time**2, atol=2e-14)
    assert_allclose(v[:, 0], 0.5 + 2 * r.time, atol=2e-14)
    assert_allclose(r.displacement, u, atol=2e-13)
    assert_allclose(r.velocity, v, atol=2e-13)


def test_nonproportional_damping_constant_load_convergence():
    s = System([[2, 0.2], [0.2, 1]], [[0.6, 0.2], [0.2, 0.3]], [[8, -2], [-2, 3]])
    errors = []
    for steps in (50, 100, 200):
        r = integrate(
            s,
            [0.5, -0.1],
            [0.2, 0.3],
            dt=2 / steps,
            steps=steps,
            rho_inf=0.5,
            force=lambda t: [0.4, -0.8],
        )
        u, v = exact_response(s, [0.5, -0.1], [0.2, 0.3], r.time, force=[0.4, -0.8])
        errors.append(max(np.max(np.abs(r.displacement - u)), np.max(np.abs(r.velocity - v))))
    assert np.all(np.log2(np.array(errors[:-1]) / errors[1:]) > 1.85)


@pytest.mark.parametrize("rho", [0.0, 0.3, 0.7, 1.0])
def test_spectral_stability_and_high_frequency_limit(rho):
    radii = [
        max(abs(np.linalg.eigvals(amplification_matrix(x, rho))))
        for x in np.logspace(-3, 2, 60)
    ]
    assert max(radii) <= 1 + 2e-12
    # At rho=1 all three eigenvalues coalesce near -1 for extreme omega*dt;
    # evaluating the limit numerically there is ill-conditioned. Its radius
    # is identically one, already checked above at moderate frequencies.
    frequency = 100 if rho == 1 else 1e7
    limit = max(abs(np.linalg.eigvals(amplification_matrix(frequency, rho))))
    assert abs(limit - rho) < 1e-4


def test_zero_steps_and_empty_reference():
    r = integrate(oscillator(), [1], [2], dt=0.1, steps=0)
    assert_allclose(r.acceleration, [[-4]])
    assert_allclose(energy_budget(oscillator(), r).defect, [0])
    assert exact_response(oscillator(), [1], [0], [])[0].shape == (0, 1)


def test_system_owns_readonly_copy():
    m = np.eye(2)
    s = System(m, m, m)
    m[0, 0] = 3
    assert s.mass[0, 0] == 1
    with pytest.raises(ValueError):
        s.mass[0, 0] = 4


@pytest.mark.parametrize(
    "mass,damping,stiffness",
    [
        ([[0]], [[0]], [[1]]),
        ([[-1]], [[0]], [[1]]),
        ([[1]], [[-1]], [[1]]),
        ([[1]], [[0]], [[-1]]),
        ([[np.nan]], [[0]], [[1]]),
        ([[1]], [[0]], [[1j]]),
        ([], [], []),
        ([[1, 2]], [[0]], [[1]]),
        (np.eye(2), np.eye(1), np.eye(2)),
        (np.eye(2), np.eye(2), [[1, 1], [0, 1]]),
    ],
)
def test_bad_systems_rejected(mass, damping, stiffness):
    with pytest.raises(ValueError):
        System(mass, damping, stiffness)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dt": 0},
        {"dt": -1},
        {"dt": np.inf},
        {"steps": -1},
        {"steps": 2.5},
        {"steps": True},
        {"rho_inf": -0.1},
        {"rho_inf": 1.1},
        {"rho_inf": np.nan},
        {"force": lambda t: [np.inf]},
        {"force": lambda t: [1, 2]},
    ],
)
def test_bad_integration_inputs_rejected(kwargs):
    options = {"dt": 0.1, "steps": 2} | kwargs
    with pytest.raises(ValueError):
        integrate(oscillator(), [1], [0], **options)


@pytest.mark.parametrize("times", [[-1], [[0]], [np.nan]])
def test_bad_reference_times(times):
    with pytest.raises(ValueError):
        exact_response(oscillator(), [1], [0], times)


@pytest.mark.parametrize("frequency", [-1, np.nan, np.inf, 1e200])
def test_bad_amplification_frequency(frequency):
    with pytest.raises(ValueError):
        amplification_matrix(frequency)


def test_load_is_sampled_once_at_every_endpoint():
    calls = []

    def force(t):
        calls.append(t)
        return [0]

    r = integrate(oscillator(), [1], [0], dt=0.1, steps=4, force=force)
    assert_allclose(calls, r.time)


@pytest.mark.parametrize("rho", [0.0, 0.5, 1.0])
def test_manufactured_time_dependent_load(rho):
    s = System([[2, 0.2], [0.2, 1]], [[0.6, 0.2], [0.2, 0.3]], [[8, -2], [-2, 3]])
    frequency = np.array([1.3, 2.7])

    def force(t):
        u = np.sin(frequency * t)
        v = frequency * np.cos(frequency * t)
        a = -(frequency**2) * u
        return s.mass @ a + s.damping @ v + s.stiffness @ u

    errors = []
    for steps in (100, 200, 400):
        r = integrate(s, [0, 0], frequency, dt=2 / steps, steps=steps, rho_inf=rho, force=force)
        expected = np.sin(r.time[:, None] * frequency)
        errors.append(np.max(np.abs(r.displacement - expected)))
    assert np.all(np.log2(np.array(errors[:-1]) / errors[1:]) > 1.9)


@pytest.mark.parametrize("rho", [0.0, 0.5, 1.0])
def test_static_equilibrium_is_preserved(rho):
    r = integrate(
        oscillator(0.3), [0.5], [0], dt=0.3, steps=100, rho_inf=rho, force=lambda t: [4.0]
    )
    assert_allclose(r.displacement, 0.5, atol=2e-15)
    assert_allclose(r.velocity, 0, atol=2e-15)
