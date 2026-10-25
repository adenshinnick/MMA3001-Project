"""Tests for the comparison and uncertainty studies."""
from pathlib import Path

import numpy as np
import pytest

from rocket6dof.analysis import (ScaledDrag, apogee_fixed_step, apogee_rk45, fly, monte_carlo,
                                 numpy_cross_baseline, observed_order, sample_inputs, variant)
from rocket6dof.config import load_rocket

ROCKET = load_rocket(Path(__file__).parent.parent / "data" / "rocket" / "l1_rocket.toml")


def test_rk4_and_scipy_rk45_agree():
    """Two independent integrators must give the same apogee."""
    rk4, _, _ = apogee_fixed_step(ROCKET, 0.02)
    rk45, _, _ = apogee_rk45(ROCKET, rtol=1e-10)
    assert rk4 == pytest.approx(rk45, abs=1e-3)


def test_observed_order():
    h = np.array([0.4, 0.2, 0.1])
    np.testing.assert_allclose(observed_order(h, 3.0 * h**4), [4.0, 4.0])


def test_variant_scales_inputs():
    r = variant(ROCKET, cd_factor=1.1, mass_factor=0.9, thrust_factor=1.05)
    assert r.dry_mass == pytest.approx(0.9 * ROCKET.dry_mass)
    assert r.motor.total_impulse == pytest.approx(1.05 * ROCKET.motor.total_impulse)
    assert r.drag_model(0.3, 1e7) == pytest.approx(1.1 * ROCKET.drag_model(0.3, 1e7))
    assert isinstance(r.drag_model, ScaledDrag)


def test_more_drag_means_lower_apogee():
    low = fly(ROCKET, {"cd_factor": 1.1}, full=False)["apogee"]
    nominal = fly(ROCKET, {}, full=False)["apogee"]
    assert low < nominal


def test_monte_carlo_inputs_are_repeatable():
    assert sample_inputs(5, seed=1) == sample_inputs(5, seed=1)
    assert sample_inputs(5, seed=1) != sample_inputs(5, seed=2)


def test_parallel_runs_give_the_same_answers():
    samples = sample_inputs(2, seed=3)
    serial = monte_carlo(ROCKET, samples, workers=1)
    parallel = monte_carlo(ROCKET, samples, workers=2)
    assert [r["apogee"] for r in serial] == [r["apogee"] for r in parallel]


def test_optimised_cross_product_changes_nothing():
    """The performance change (np.cross -> cross3) must not change the answer."""
    fast = apogee_fixed_step(ROCKET, 0.04)[0]
    with numpy_cross_baseline():
        slow = apogee_fixed_step(ROCKET, 0.04)[0]
    assert fast == pytest.approx(slow, abs=1e-9)
