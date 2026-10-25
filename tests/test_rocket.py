import numpy as np
import pytest

from rocket6dof.motor import Motor
from rocket6dof.rocket import Parachute, Rocket

# Motor: 0.2 kg at ignition, 0.1 kg after burnout, 0.2 m long.
MOTOR = Motor("M", [1.0, 2.0], [100.0, 0.0], 0.1, 0.2, diameter=0.03, length=0.2)
J_DRY = np.diag([0.001, 0.05, 0.05])


def test_cg_moves_forward_as_propellant_burns():
    """Dry CG at 0.4 m; motor from 0.6 to 0.8 m, so its CG is at 0.7 m."""
    r = Rocket(1.0, J_DRY, MOTOR, dry_cg=0.4, motor_station=0.6)
    assert r.cg(0.0) == pytest.approx((1.0 * 0.4 + 0.2 * 0.7) / 1.2)
    assert r.cg(5.0) == pytest.approx((1.0 * 0.4 + 0.1 * 0.7) / 1.1)


def test_inconsistent_options_are_rejected():
    with pytest.raises(ValueError):
        Rocket(1.0, J_DRY, MOTOR, motor_station=0.6)            # needs dry_cg


def test_parachute():
    assert Parachute(1.2, 0.75).area == pytest.approx(np.pi * 0.36)
    with pytest.raises(ValueError):
        Parachute(0.0, 0.75)
