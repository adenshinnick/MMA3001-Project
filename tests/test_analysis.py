"""Tests for the comparison and sensitivity studies."""
from pathlib import Path

import numpy as np
import pytest

from rocket6dof.analysis import (ScaledDrag, apogee_by_method, fly, observed_order,
                                 rail_exit_state, variant)
from rocket6dof.config import load_rocket

ROCKET = load_rocket(Path(__file__).parent.parent / "data" / "rocket" / "l1_rocket.toml")


def test_observed_order():
    h = np.array([0.4, 0.2, 0.1])
    np.testing.assert_allclose(observed_order(h, 3.0 * h**4), [4.0, 4.0])


def test_variant_scales_inputs():
    r = variant(ROCKET, cd_factor=1.1, mass_factor=0.9, thrust_factor=1.05)
    assert r.dry_mass == pytest.approx(0.9 * ROCKET.dry_mass)
    assert r.motor.total_impulse == pytest.approx(1.05 * ROCKET.motor.total_impulse)
    assert r.drag_model(0.3) == pytest.approx(1.1 * ROCKET.drag_model(0.3))
    assert isinstance(r.drag_model, ScaledDrag)


def test_more_drag_means_lower_apogee():
    low = fly(ROCKET, {"cd_factor": 1.1}, full=False)["apogee"]
    nominal = fly(ROCKET, {}, full=False)["apogee"]
    assert low < nominal


def test_integrators_agree_and_euler_is_first_order():
    """RK4 and SciPy's RK45 agree on the apogee to well under a millimetre at
    fine settings (an independent check of RK4), and halving explicit Euler's
    step roughly halves its error (first order)."""
    t0, x0 = rail_exit_state(ROCKET)
    ref = apogee_by_method(ROCKET, "rk4", 0.005, t0, x0)[0]
    assert apogee_by_method(ROCKET, "rk45", 1e-9, t0, x0)[0] == pytest.approx(ref, abs=1e-4)
    e1 = abs(apogee_by_method(ROCKET, "euler", 0.004, t0, x0)[0] - ref)
    e2 = abs(apogee_by_method(ROCKET, "euler", 0.002, t0, x0)[0] - ref)
    assert 1.7 < e1 / e2 < 2.3
