"""Tests for the comparison and sensitivity studies."""
from pathlib import Path

import numpy as np
import pytest

from rocket6dof.analysis import ScaledDrag, fly, observed_order, variant
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
