import numpy as np
import pytest

from rocket6dof.dynamics import OMEGA, POS, QUAT, VEL, make_rhs, make_state
from rocket6dof.integrators import integrate_rk4
from rocket6dof.motor import Motor
from rocket6dof.quaternions import quat_from_axis_angle
from rocket6dof.rocket import Rocket

G0 = 9.80665  # typed here so the tests are independent of the code
Y = np.array([0.0, 1.0, 0.0])

# A motor with a constant 100 N thrust for 2 s. A thrust curve must end at
# 0 N, so it falls to zero over a final 0.001 s. Its total impulse, by hand:
# 100 N * 2 s + 0.5 * 100 N * 0.001 s = 200.05 N s.
THRUST = 100.0
TOTAL_IMPULSE = 200.05
PROPELLANT_MASS = 0.1
MOTOR_MASS = 0.3
MOTOR = Motor("CONST", [0.0, 2.0, 2.001], [THRUST, THRUST, 0.0], PROPELLANT_MASS, MOTOR_MASS)

DRY_MASS = 1.0
J = np.diag([0.01, 0.5, 0.5])


def make_rocket():
    return Rocket(DRY_MASS, J, MOTOR)


def nose_at_elevation(elevation_deg):
    """Attitude with the nose pointing east, elevation_deg above horizontal."""
    return quat_from_axis_angle(Y, -np.radians(elevation_deg))


def test_make_state_layout():
    x = make_state([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [2.0, 0.0, 0.0, 0.0], [7.0, 8.0, 9.0])
    np.testing.assert_allclose(x[POS], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(x[VEL], [4.0, 5.0, 6.0])
    np.testing.assert_allclose(x[QUAT], [1.0, 0.0, 0.0, 0.0])  # normalised
    np.testing.assert_allclose(x[OMEGA], [7.0, 8.0, 9.0])


def test_rocket_rejects_non_positive_dry_mass():
    with pytest.raises(ValueError):
        Rocket(0.0, J, MOTOR)


def test_free_fall_after_burnout_is_a_parabola():
    """After burnout only gravity acts, so the path is the exact parabola."""
    f = make_rhs(make_rocket())
    x0 = make_state([0.0, 0.0, 100.0], [5.0, 0.0, 20.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    t, x = integrate_rk4(f, x0, 10.0, 13.0, 0.01)  # starts well after burnout

    T = t[-1] - t[0]
    np.testing.assert_allclose(
        x[-1, POS], [5.0 * T, 0.0, 100.0 + 20.0 * T - 0.5 * G0 * T**2], atol=1e-9
    )
    np.testing.assert_allclose(x[-1, VEL], [5.0, 0.0, 20.0 - G0 * T], atol=1e-9)


@pytest.mark.parametrize("elevation_deg", [90.0, 80.0, 45.0])
def test_constant_thrust_matches_rocket_equation(elevation_deg):
    """Constant thrust T along a fixed direction u, plus gravity:

        v(t) = c ln(m0/m) u - g t z
        r(t) = (c/mdot) [(m0 - m) - m ln(m0/m)] u - 0.5 g t^2 z

    with m = m0 - mdot t and c = T / mdot.
    """
    q0 = nose_at_elevation(elevation_deg)
    x0 = make_state([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], q0, [0.0, 0.0, 0.0])
    t, x = integrate_rk4(make_rhs(make_rocket()), x0, 0.0, 1.5, 0.01)  # inside the burn

    T = t[-1]
    m0 = DRY_MASS + MOTOR_MASS
    mdot = PROPELLANT_MASS * THRUST / TOTAL_IMPULSE  # constant burn rate [kg/s]
    m = m0 - mdot * T
    c = THRUST / mdot                                # effective exhaust velocity [m/s]
    el = np.radians(elevation_deg)
    u = np.array([np.cos(el), 0.0, np.sin(el)])      # nose direction
    z_hat = np.array([0.0, 0.0, 1.0])

    v_expected = c * np.log(m0 / m) * u - G0 * T * z_hat
    r_expected = (c / mdot) * ((m0 - m) - m * np.log(m0 / m)) * u - 0.5 * G0 * T**2 * z_hat

    np.testing.assert_allclose(x[-1, VEL], v_expected, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(x[-1, POS], r_expected, rtol=1e-9, atol=1e-9)


def test_roll_does_not_change_the_flight_path():
    """Rolling about its own long axis must not change where a rocket goes,
    because the thrust stays along that axis."""
    q0 = nose_at_elevation(80.0)
    still = make_state([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], q0, [0.0, 0.0, 0.0])
    rolling = make_state([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], q0, [3.0, 0.0, 0.0])

    f = make_rhs(make_rocket())
    _, x_still = integrate_rk4(f, still, 0.0, 1.5, 0.01)
    _, x_rolling = integrate_rk4(f, rolling, 0.0, 1.5, 0.01)

    np.testing.assert_allclose(x_rolling[-1, POS], x_still[-1, POS], atol=1e-8)
    assert x_rolling[-1, OMEGA][0] == pytest.approx(3.0)  # roll rate unchanged