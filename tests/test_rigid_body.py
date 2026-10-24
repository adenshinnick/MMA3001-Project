import numpy as np
import pytest

from rocket6dof.integrators import integrate_rk4
from rocket6dof.quaternions import quat_derivative
from rocket6dof.rigid_body import (
    angular_acceleration,
    angular_momentum_earth,
    rotational_kinetic_energy,
    validate_inertia,
)

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])
NO_MOMENT = np.zeros(3)

# Rocket-like inertia: small about the long (body x) axis, large and equal
# about the two transverse axes [kg m^2].
J_ROCKET = np.diag([0.01, 0.5, 0.5])


def torque_free_rhs(J):
    """Build f(t, x) for a rigid body with no applied moment.

    The state is x = [q (4 numbers), omega (3 numbers)].
    """
    def f(t, x):
        q = x[0:4]
        omega = x[4:7]
        q_dot = quat_derivative(q, omega)
        omega_dot = angular_acceleration(J, omega, NO_MOMENT)
        return np.concatenate((q_dot, omega_dot))
    return f


def test_accepts_valid_inertia():
    np.testing.assert_allclose(validate_inertia(J_ROCKET), J_ROCKET)


@pytest.mark.parametrize(
    "J",
    [
        np.diag([1.0, 2.0]),
        np.array([[1.0, 0.1, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]),
        np.diag([1.0, -2.0, 3.0]),
        np.diag([1.0, 1.0, 3.0]),
    ],
    ids=["wrong-shape", "not-symmetric", "negative-moment", "triangle-inequality"],
)
def test_rejects_invalid_inertia(J):
    with pytest.raises(ValueError):
        validate_inertia(J)


def test_steady_spin_about_principal_axis():
    """With no moment, a spin about a principal axis does not change."""
    omega_dot = angular_acceleration(J_ROCKET, [5.0, 0.0, 0.0], NO_MOMENT)
    np.testing.assert_allclose(omega_dot, np.zeros(3), atol=1e-15)


def test_moment_from_rest():
    """From rest, omega_dot = J^-1 M."""
    J = np.diag([1.0, 2.0, 4.0])
    omega_dot = angular_acceleration(J, NO_MOMENT, [0.0, 0.0, 1.0])
    np.testing.assert_allclose(omega_dot, [0.0, 0.0, 0.25])


def test_gyroscopic_term_matches_eulers_equations():
    """Check the sign of omega x (J omega) against the component form.

    Euler's third equation: I3 * w3_dot = (I1 - I2) * w1 * w2 = (1 - 2) * 1 * 1 = -1,
    so w3_dot = -1/3. The other two components are zero because w3 = 0.
    """
    J = np.diag([1.0, 2.0, 3.0])
    omega_dot = angular_acceleration(J, [1.0, 1.0, 0.0], NO_MOMENT)
    np.testing.assert_allclose(omega_dot, [0.0, 0.0, -1.0 / 3.0], atol=1e-15)


def test_axisymmetric_coning_matches_exact_solution():
    """A spinning, wobbling rocket with no moment has an exact solution.

    The roll rate stays constant, and the transverse rates rotate at
    lam = (I_a - I_t) / I_t * roll. Estimated RK4 error here is about
    4e-9, so atol = 1e-7 leaves a margin.
    """
    I_a, I_t = 0.01, 0.5
    roll, wy0, wz0 = 2.0, 0.3, 0.0
    lam = (I_a - I_t) / I_t * roll
    J = np.diag([I_a, I_t, I_t])

    x0 = np.concatenate((IDENTITY, [roll, wy0, wz0]))
    t, x = integrate_rk4(torque_free_rhs(J), x0, 0.0, 5.0, 0.01)

    T = t[-1]
    expected_omega = [
        roll,
        wy0 * np.cos(lam * T) - wz0 * np.sin(lam * T),
        wy0 * np.sin(lam * T) + wz0 * np.cos(lam * T),
    ]
    np.testing.assert_allclose(x[-1, 4:7], expected_omega, atol=1e-7)


def test_torque_free_tumbling_conserves_energy_and_momentum():
    """With no moment, kinetic energy and the Earth-frame angular momentum
    vector must stay constant, even for an irregular tumble."""
    J = np.diag([1.0, 2.0, 3.0])
    x0 = np.concatenate((IDENTITY, [0.1, 1.0, 0.1]))  # spin near the middle axis
    t, x = integrate_rk4(torque_free_rhs(J), x0, 0.0, 20.0, 0.01)

    q_start, w_start = x[0, 0:4], x[0, 4:7]
    q_end, w_end = x[-1, 0:4], x[-1, 4:7]

    E_start = rotational_kinetic_energy(J, w_start)
    E_end = rotational_kinetic_energy(J, w_end)
    assert E_end == pytest.approx(E_start, rel=1e-6)

    L_start = angular_momentum_earth(q_start, J, w_start)
    L_end = angular_momentum_earth(q_end, J, w_end)
    np.testing.assert_allclose(L_end, L_start, rtol=1e-6, atol=1e-9)

    assert np.linalg.norm(q_end) == pytest.approx(1.0, abs=1e-6)