# Validation record

Run date: 2026-09-25. Original synthetic data; no course data used.

## Local verification completed

Windows, Python 3.14.5: 59 tests passed in the development installation and again
after installing the built wheel. The import path was explicitly checked to be
inside site-packages. Python statement coverage was 97.76% (262/268 statements);
this does not cover NumPy/SciPy internals. Ruff lint and format checks passed.
The isolated source and wheel builds passed; distribution contents were inspected
for expected source/tests/reports and absence of virtual environments or Git data.
The installed console command, standalone example and `pip check` passed. All
eight experiment acceptance checks passed. The example's maximum energy defect
was 1.243e-14 J.

The numerical core was merged only after its 51 tests and lint checks passed.
The validation branch adds six forcing/equilibrium cases and two report/CLI tests.
Remote CI outcomes can be inspected in the repository Actions history; no review
or remote test success is inferred from this local record.

## Numerical evidence

- Closed-form undamped oscillators establish second-order displacement convergence
  at rho=0, 0.4, 0.8, 1. Manufactured coupled sinusoidal solutions test variable
  forcing at rho=0, 0.5, 1.
- Full weighted equilibrium and both kinematic updates are checked for a coupled,
  non-proportionally damped system with time-dependent forcing.
- An undamped 3,000-step trajectory conserves energy at rho=1. The independent
  forced, damped midpoint energy identity is checked on two coupled DOFs.
- Matrix exponentials match independent underdamped, critically damped, overdamped,
  and rigid-body closed forms. Constant-load, non-proportional damping convergence
  and stationary equilibrium are tested separately.
- Spectral stability is sampled across omega*h in [1e-3, 100]; the asymptotic
  limit is checked at 1e7 for rho<1. The three-state map includes the parasitic mode.
- Eight report acceptance checks are enforced. Repeated runs produce identical
  parsed reports on the same environment; CSV reload and PNG generation are tested.
  Cross-platform values need only agree to tolerance.

The six-panel PNG was visually inspected for readable axes, legends, labels and
unclipped panels. Tiny modal energy is clipped at 1e-16 for display; numerical
values below roundoff significance are not a physical claim.

## Development findings

An initial full-map stability test used omega*h up to 1e4 at rho=1 with a 2e-12
radius tolerance. Three nearly coincident eigenvalues near -1 made the numerical
radius overshoot by 4.8e-9. The test now uses frequencies up to 100 for that
invariant; analytical radius one and long-time energy conservation provide separate
evidence. For rho<1 the limit remains tested at 1e7. The integrator was unchanged.

The initial CSV reload test failed because `loadtxt` cannot parse the intentionally
blank first convergence order. It now uses `genfromtxt`, preserving that missing
value as NaN. Ruff found long lines and a lambda assignment; these were fixed.
Development failures were resolved before merging.

## Scope

See README for reproduction commands/equations and `results/validation.json` for
values. The dependency snapshot records local versions. Workflow conclusions,
rather than configuration alone, establish remote CI success. Actual build/install
and CI outcomes are recorded in the completion record after checks finish.

No nonlinear, constrained, sparse, impulsive-load, ill-conditioned or large-system
performance validation is claimed. There is no custom C++ component. Matrix
exponential is independent of the recurrence, but uses the same model matrices;
scalar closed forms check its signs and damping conventions. Spectral sampling is
numerical evidence, not a proof over all inputs.
