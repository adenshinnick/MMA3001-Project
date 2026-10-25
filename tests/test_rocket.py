import numpy as np
import pytest

from rocket6dof.motor import Motor
from rocket6dof.rocket import Parachute, Rocket

# A motor with known numbers: 0.2 kg at ignition, 0.1 kg after burnout, 0.2 m long, 0.03 m wide.
MOTOR = Motor("M", [1.0, 2.0], [100.0, 0.0], 0.1, 0.2, diameter=0.03, length=0.2)
J_DRY = np.diag([0.001, 0.05, 0.05])


def make_rocket():
    # Dry CG at 0.4 m; motor from 0.6 to 0.8 m, so its CG is at 0.7 m.
    return Rocket(1.0, J_DRY, MOTOR, dry_cg=0.4, motor_station=0.6)


def test_cg_moves_forward_as_propellant_burns():
    r = make_rocket()
    assert r.cg(0.0) == pytest.approx((1.0 * 0.4 + 0.2 * 0.7) / 1.2)
    assert r.cg(5.0) == pytest.approx((1.0 * 0.4 + 0.1 * 0.7) / 1.1)


def test_inertia_by_hand():
    """At burnout: motor 0.1 kg cylinder; parallel-axis terms to the combined CG."""
    r = make_rocket()
    x_cg = (0.4 + 0.07) / 1.1
    I_motor_t = 0.1 * (3 * 0.015**2 + 0.2**2) / 12
    expected_t = 0.05 + 1.0 * (0.4 - x_cg) ** 2 + I_motor_t + 0.1 * (0.7 - x_cg) ** 2
    J = r.inertia_at(5.0)
    assert J[0, 0] == pytest.approx(0.001 + 0.5 * 0.1 * 0.015**2)
    assert J[1, 1] == pytest.approx(expected_t)
    assert J[2, 2] == pytest.approx(expected_t)


def test_inertia_is_constant_without_motor_position():
    r = Rocket(1.0, J_DRY, MOTOR)
    np.testing.assert_array_equal(r.inertia_at(0.0), J_DRY)


def test_inconsistent_options_are_rejected():
    with pytest.raises(ValueError):
        Rocket(1.0, J_DRY, MOTOR, motor_station=0.6)            # needs dry_cg
    with pytest.raises(ValueError):
        Rocket(1.0, J_DRY, Motor("M", [1.0, 2.0], [100.0, 0.0], 0.1, 0.2),
               dry_cg=0.4, motor_station=0.6)                     # motor size unknown


def test_parachute():
    assert Parachute(1.2, 0.75).area == pytest.approx(np.pi * 0.36)
    with pytest.raises(ValueError):
        Parachute(0.0, 0.75)
