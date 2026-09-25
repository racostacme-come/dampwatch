import json

import numpy as np

from dampwatch.cli import main
from dampwatch.study import run_study


def test_complete_report_and_numeric_reproducibility(tmp_path):
    first = run_study(tmp_path / "first", plot=True)
    second = run_study(tmp_path / "second", plot=False)
    assert first == second
    assert all(first["checks"].values())
    assert (tmp_path / "first" / "damping_audit.png").stat().st_size > 10000
    assert json.loads((tmp_path / "first" / "validation.json").read_text()) == first
    for name in ("free_response", "forced_budget", "convergence", "spectral_radius"):
        a = np.genfromtxt(tmp_path / "first" / f"{name}.csv", skip_header=1, delimiter=",")
        b = np.genfromtxt(tmp_path / "second" / f"{name}.csv", skip_header=1, delimiter=",")
        # The first convergence order is intentionally blank (NaN when loaded).
        np.testing.assert_allclose(a, b)


def test_cli(tmp_path, capsys):
    assert main(["--output", str(tmp_path), "--no-plot"]) == 0
    assert "finest_observed_order" in capsys.readouterr().out
    assert (tmp_path / "validation.json").exists()
