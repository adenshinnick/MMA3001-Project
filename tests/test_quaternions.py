import pytest
import numpy as np


from rocket6dof.integrators import integrate_rk4
from rocket6dof.quaternions import (
    quat_derivative,
    quat_from_axis_angle,
    quat_multiply,
    quat_normalise,
    quat_to_rotation_matrix,
    rotate_vector,
)

X = np.array([1.0,0.0,0.0])
Y = np.array([0.0, 1.0, 0.0])
Z = np.array([0.0, 0.0, 1.0])

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])

# A rotation with no special symmetry, for tests where any rotation will do.
Q_GENERAL = quat_from_axis_angle([1.0, 2.0, 3.0], 0.7)

def test_identity_changes_nothing():
    np.testing.assert_allclose(quat_to_rotation_matrix(IDENTITY), np.eye(3), atol=1e-15)
    np.testing.assert_allclose(rotate_vector(IDENTITY, [1.0,2.0,3.0]),[1.0,2.0,3.0])

@pytest.mark.parametrize(
    "axis, vector, expected",
    [
        (Z, X, Y),
        (Z, Y, -X),
        (Y, Z, X),
        (X, Y, Z),
    ],
     ids=["about-z-x-to-y", "about-z-y-to-minus-x", "about-y-z-to-x", "about-x-y-to-z"],
)
def test_quarter_turns_follow_right_hand_rule(axis, vector, expected):
    q = quat_from_axis_angle(axis,np.pi/2)
    R = quat_to_rotation_matrix(q)
    np.testing.assert_allclose(R @ vector, expected, atol = 1e-12)

def test_rotation_matrix_is_a_proper_rotation():
    R = quat_to_rotation_matrix(Q_GENERAL)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)

def test_matrix_and_quaternion_product_agree():
    v=np.array([0.3, -1.2, 2.5])
    np.testing.assert_allclose(
        quat_to_rotation_matrix(Q_GENERAL) @ v, rotate_vector(Q_GENERAL, v), atol=1e-12
    )

def test_two_quarter_turns_make_a_half_turn():
    quarter = quat_from_axis_angle(Z, np.pi / 2)
    half = quat_multiply(quarter, quarter)
    np.testing.assert_allclose(rotate_vector(half, X), -X, atol=1e-12)

def test_normalize():
    np.testing.assert_allclose(quat_normalise([2.0, 0.0, 0.0, 0.0]), IDENTITY)
    with pytest.raises(ValueError):
        quat_normalise([0.0, 0.0, 0.0, 0.0])


def test_constant_spin_matches_exact_solution():
    """Spinning at a constant rate about z must give the exact rotation."""
    omega = np.array([0.0, 0.0, 1.0])  # 1 rad/s about the body z axis

    def f(t, q):
        return quat_derivative(q, omega)

    t, q = integrate_rk4(f, IDENTITY, 0.0, 2.0, 0.01)
    expected = quat_from_axis_angle(Z, 1.0 * t[-1])
    np.testing.assert_allclose(q[-1], expected, atol=1e-9)
    # RK4 does not keep the length at exactly 1, but the drift should be tiny.
    assert np.linalg.norm(q[-1]) == pytest.approx(1.0, abs=1e-9)