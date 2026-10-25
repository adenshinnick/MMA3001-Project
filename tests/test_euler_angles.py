import numpy as np
import pytest

from rocket6dof.euler_angles import (ENU_TO_NED, euler_rates, euler_to_rotation,
                                     quat_to_euler, rotation_to_euler)
from rocket6dof.integrators import integrate_rk4
from rocket6dof.quaternions import quat_derivative, quat_from_axis_angle, quat_to_rotation_matrix
from rocket6dof.simulation import launch_attitude


def test_round_trip():
    angles = np.array([0.3, -0.4, 1.2])
    np.testing.assert_allclose(rotation_to_euler(euler_to_rotation(angles)), angles, atol=1e-12)


def test_rail_elevation_is_the_pitch_angle():
    assert np.degrees(quat_to_euler(launch_attitude(80.0, 90.0))[1]) == pytest.approx(80.0)


def test_euler_angles_agree_with_quaternion_away_from_gimbal_lock():
    """Integrating the Euler-angle rates and the quaternion with the same body
    rates must give the same attitude when pitch stays away from +/-90 deg."""
    omega = np.array([0.5, 0.2, -0.1])
    q0 = quat_from_axis_angle([1.0, 2.0, 3.0], 0.4)
    _, q = integrate_rk4(lambda t, q: quat_derivative(q, omega), q0, 0.0, 2.0, 1e-3)
    _, e = integrate_rk4(lambda t, e: euler_rates(e, omega), quat_to_euler(q0), 0.0, 2.0, 1e-3)
    np.testing.assert_allclose(euler_to_rotation(e[-1]),
                               ENU_TO_NED @ quat_to_rotation_matrix(q[-1]), atol=1e-9)


def test_euler_rates_blow_up_near_vertical():
    """Gimbal lock: a small pitch rate near theta = 90 deg needs huge roll and
    yaw rates. The quaternion derivative stays bounded by |omega| / 2."""
    omega = np.array([0.0, 0.0, 0.1])
    rates = euler_rates([0.0, np.radians(89.999), 0.0], omega)
    assert abs(rates[2]) > 1000.0
    q = launch_attitude(90.0)
    assert np.linalg.norm(quat_derivative(q, omega)) == pytest.approx(0.05)
