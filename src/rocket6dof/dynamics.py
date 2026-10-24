"""Equations of motion for the 6-DOF rocket

State vecotr x (13 numbers):
x[0:3] position, Earth frame [m]
x[3:6] velocity, Earth frame [m/s]
x[6:10] attitude quaternion, body to earth [-]
x[10:13] angular velocitty, body frame [rad/s]
"""
import numpy as np

from rocket6dof.atmosphere import G0
from rocket6dof.quaternions import quat_derivative, quat_normalise, quat_to_rotation_matrix
from rocket6dof.rigid_body import angular_acceleration

#named slices for the poarts of the state vector

POS = slice(0,3)
VEL = slice(3,6)
QUAT = slice(6,10)
OMEGA = slice(10,13)

GRAVITY = np.array([0.0, 0.0, -G0]) #flat earth, constant gravity [m/s^2]

def make_state(position, velocity, q, omega):
    """Build a 13 number state vector, normalising the quaternion.
    
    Raises
    ------
    ValueError
        if the parts do not add up to 13 numbers.
        """
    x = np.concatenate((position, velocity, quat_normalise(q), omega)).astype(float)
    if x.size !=13:
        raise ValueError(f"state must have 13 numbers, got {x.size}.")
    return x

def make_rhs(rocket, gravity=GRAVITY):
    """Build the right hand side of f(t,x) for a rocket under gravity and thrust.
    Parameters
    ----------
    rocket: Rocket
        the rocket to simulate.
    gravity: numpy.ndarray
        Gravitational Acceleration, earth frame[m/s^2]
        
    Returns
    -------
    callable
        f(t,x), returning dx/dt for the 13 number state
    """
    J= rocket.inertia

    def f(t,x):
        q = x[QUAT]
        omega = x[OMEGA]
        R = quat_to_rotation_matrix(q)

        #thrust acts along the body x axis: turn it into earth axes
        thrust_body= np.array([rocket.motor.thrust(t), 0.0, 0.0])
        acceleration = R @ thrust_body/ rocket.mass(t) + gravity

        moment_body= np.zeros(3) #no aero moments yet

        x_dot = np.empty(13)
        x_dot[POS] = x[VEL]
        x_dot[VEL] = acceleration
        x_dot[QUAT] = quat_derivative(q, omega)
        x_dot[OMEGA] = angular_acceleration(J, omega, moment_body)
        return x_dot
    return f