import numpy as np
import pytest

from rocket6dof.dynamics import POS, VEL
from rocket6dof.motor import Motor
from rocket6dof.quaternions import quat_from_axis_angle
from rocket6dof.rocket import Rocket
from rocket6dof.simulation import simulate

G0 = 9.80665
GRAVITY = np.array([0.0, 0.0, -G0])
Y = np.array([0.0, 1.0, 0.0])
J = np.diag([0.01, 0.5, 0.5])

# Full thrust from ignition (the rocket lifts off immediately).
CONST_MOTOR = Motor("CONST", [0.0, 2.0, 2.001], [100.0, 100.0, 0.0], 0.1, 0.3)
# Thrust starts at zero, so the rocket must wait on the pad.
TRIANGLE_MOTOR = Motor("TRI", [1.0, 2.0], [100.0, 0.0], 0.05, 0.15)


def nose_at_elevation(elevation_deg):
    """Attitude with the nose pointing east, elevation_deg above horizontal."""
    return quat_from_axis_angle(Y, -np.radians(elevation_deg))


def coast_from_burnout(result):
    """Burnout time, position and velocity, and the exact vacuum-coast times
    from burnout to apogee and to landing."""
    burnout = result["events"]["burnout"]
    r_b = burnout["state"][POS]
    v_b = burnout["state"][VEL]
    t_to_apogee = v_b[2] / G0
    t_to_landing = (v_b[2] + np.sqrt(v_b[2] ** 2 + 2 * G0 * r_b[2])) / G0
    return burnout["t"], r_b, v_b, t_to_apogee, t_to_landing


def test_rejects_bad_settings():
    rocket = Rocket(1.0, J, CONST_MOTOR)
    with pytest.raises(ValueError):
        simulate(rocket, nose_at_elevation(90.0), h=0.0)
    with pytest.raises(ValueError):
        simulate(rocket, nose_at_elevation(90.0), rail_length=-1.0)


@pytest.mark.parametrize("elevation_deg", [90.0, 80.0])
def test_events_match_exact_vacuum_coast(elevation_deg):
    """After burnout only gravity acts, so apogee and landing follow exactly
    from the burnout state."""
    result = simulate(Rocket(1.0, J, CONST_MOTOR), nose_at_elevation(elevation_deg))
    assert result["status"] == "landed"

    t_b, r_b, v_b, t_up, t_down = coast_from_burnout(result)
    assert t_b == pytest.approx(CONST_MOTOR.burn_time, abs=1e-12)

    apogee = result["events"]["apogee"]
    assert apogee["t"] == pytest.approx(t_b + t_up, rel=1e-9)
    expected_apogee = r_b + v_b * t_up + 0.5 * GRAVITY * t_up**2
    np.testing.assert_allclose(apogee["position"], expected_apogee, rtol=1e-9, atol=1e-9)

    landing = result["events"]["landing"]
    assert landing["t"] == pytest.approx(t_b + t_down, rel=1e-9)
    expected_landing = r_b + v_b * t_down + 0.5 * GRAVITY * t_down**2
    np.testing.assert_allclose(landing["position"], expected_landing, rtol=1e-9, atol=1e-6)
    assert result["x"][-1, 2] == pytest.approx(0.0, abs=1e-9)


def test_rocket_waits_on_pad_until_thrust_exceeds_weight():
    """Thrust starts at zero, so the rocket must sit still, not sink, until
    thrust exceeds weight at about 0.11 s."""
    result = simulate(Rocket(1.0, J, TRIANGLE_MOTOR), nose_at_elevation(90.0))
    t, x = result["t"], result["x"]

    waiting = t < 0.1
    np.testing.assert_array_equal(x[waiting, 2], 0.0)  # height stays exactly 0
    np.testing.assert_array_equal(x[waiting, 5], 0.0)  # vertical speed stays exactly 0
    assert np.all(x[:, 2] >= -1e-9)                    # never below the ground
    assert result["events"]["rail_exit"]["t"] > 0.11


def test_weak_motor_never_leaves_the_rail():
    """A motor weaker than the rocket's weight must leave it on the pad."""
    weak = Motor("WEAK", [1.0, 2.0], [5.0, 0.0], 0.01, 0.05)
    result = simulate(Rocket(1.0, J, weak), nose_at_elevation(90.0), t_max=5.0)
    assert result["status"] == "t_max reached"
    assert "rail_exit" not in result["events"]
    np.testing.assert_array_equal(result["x"][:, 2], 0.0)