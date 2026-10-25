"""Equations of motion for the 13-state rocket.

State: position x[0:3] and velocity x[3:6] (Earth frame, z up), attitude
quaternion x[6:10] (body to Earth), body angular velocity x[10:13].
Under the parachute the rocket is treated as a point mass.
"""
import numpy as np

from rocket6dof.aerodynamics import aerodynamic_loads
from rocket6dof.atmosphere import G0, isa
from rocket6dof.quaternions import quat_derivative, quat_normalise, quat_to_rotation_matrix
from rocket6dof.rigid_body import angular_acceleration

# Named slices for the parts of the state vector.
POS = slice(0, 3)
VEL = slice(3, 6)
QUAT = slice(6, 10)
OMEGA = slice(10, 13)

GRAVITY = np.array([0.0, 0.0, -G0])  # flat Earth, constant gravity [m/s^2]


def make_state(position, velocity, q, omega):
    """Build a 13 number state vector, normalising the quaternion.

    Raises
    ------
    ValueError
        If the parts do not add up to 13 numbers.
    """
    x = np.concatenate((position, velocity, quat_normalise(q), omega)).astype(float)
    if x.size != 13:
        raise ValueError(f"state must have 13 numbers, got {x.size}.")
    return x


def make_rhs(rocket, gravity=GRAVITY, wind=None, launch_altitude=0.0):
    """Build the right-hand side f(t, x) for powered and coasting flight.

    Parameters
    ----------
    rocket : Rocket
        The rocket to simulate. Aerodynamic forces act only if it has a geometry.
    gravity : numpy.ndarray
        Gravitational acceleration, Earth frame [m/s^2].
    wind : callable, optional
        ``wind(z)`` returning the wind velocity, Earth frame [m/s]. None means still air.
    launch_altitude : float
        Height of the launch pad above sea level [m], for the atmosphere.

    Returns
    -------
    callable
        f(t, x), returning dx/dt for the 13 number state.
    """
    geom = rocket.geometry
    drag_model = rocket.drag_model

    def f(t, x):
        q = x[QUAT]
        omega = x[OMEGA]
        R = quat_to_rotation_matrix(q)

        # Thrust acts along the body x axis.
        force_body = np.array([rocket.motor.thrust(t), 0.0, 0.0])
        moment_body = np.zeros(3)

        if geom is not None:
            _, _, rho, sound_speed = isa(launch_altitude + x[2])
            v_air = x[VEL] if wind is None else x[VEL] - wind(x[2])
            v_air_body = R.T @ v_air          # R is orthogonal, so R^T turns Earth into body axes
            f_aero, m_aero, _ = aerodynamic_loads(
                geom, drag_model, v_air_body, rho, sound_speed, rocket.cg(t), omega)
            force_body = force_body + f_aero
            moment_body = moment_body + m_aero

        x_dot = np.empty(13)
        x_dot[POS] = x[VEL]
        x_dot[VEL] = R @ force_body / rocket.mass(t) + gravity
        x_dot[QUAT] = quat_derivative(q, omega)
        x_dot[OMEGA] = angular_acceleration(rocket.inertia, omega, moment_body)
        return x_dot

    return f


def make_descent_rhs(rocket, gravity=GRAVITY, wind=None, launch_altitude=0.0):
    """Build f(t, x) for the descent under the parachute.

    The rocket and parachute fall as one point mass. Drag acts opposite the
    air-relative velocity, D = 0.5 rho |v_air| v_air C_D A, so the rocket
    drifts with the wind. Attitude and body rates are frozen.

    Parameters
    ----------
    rocket : Rocket
        Must have a parachute.
    gravity, wind, launch_altitude
        As for ``make_rhs``.

    Returns
    -------
    callable
        f(t, x) for the 13 number state.

    Raises
    ------
    ValueError
        If the rocket has no parachute.
    """
    if rocket.parachute is None:
        raise ValueError("the rocket has no parachute")
    cd_area = rocket.parachute.drag_coefficient * rocket.parachute.area

    def f(t, x):
        rho = isa(launch_altitude + x[2])[2]
        v_air = x[VEL] if wind is None else x[VEL] - wind(x[2])
        speed = np.sqrt(v_air @ v_air)
        x_dot = np.zeros(13)
        x_dot[POS] = x[VEL]
        x_dot[VEL] = -0.5 * rho * speed * cd_area * v_air / rocket.mass(t) + gravity
        return x_dot

    return f
