""" rigid body rotational dynamics

Eulers equation in the body frame:

    J @ omega_dot = M - omega x (J @ omega)
    
    J is the inertia tensor about the centre of mass [kg m^2], omega the body angular
    velocity [rad/s] and M the applied moment about the centre of mass [N m], all in body axes.
    
    Assumption: J is treated as constant at each instant. the effects of inertia changing 
    during the burn (the J-dot term and jet damping) are neglected
    """

import numpy as np

from rocket6dof.quaternions import quat_to_rotation_matrix

def validate_inertia(J, tol=1e-9):
    """Check that J is a physically valid inertia tensor,
    
    Call this once hwen a rocket is set up, not inside f(t,x), so the check
    does not slow down every step.
    
    Parameters
    ----------
    J : array_like
        Inertia tensor [kg m^2], 3x3
    tol : float
        Relative tolerance for the symmetry and triangle inequality checks
        
    Returns
    -------
    numpy.ndarray
        J as a float array.
        
    Raises
    ------
    ValueError
        If J is not 3x3, not symmetric, not positive definite, or its
        principal moments breakt the triangle inequality
    """
    J = np.asarray(J, dtype=float)
    if J.shape != (3,3):
        raise ValueError(f"Inertia tensor mst be 3x3, got shape {J.shape}.")
    scale = np.max(np.abs(J))
    if not np.allclose(J, J.T, rtol = 0.0, atol= tol*scale):
        raise ValueError("Inertia tensor must be symmetric")
    principal = np.linalg.eigvalsh(J) #principal moments, smallest first
    if principal[0] <= 0.0:
        raise ValueError("inertia tensor must be positive definite.")
    if principal[0] + principal[1] < principal[2] * (1.0 - tol):
        raise ValueError("Principal moments break the triangle inequality")
    return J

def angular_acceleration(J, omega, moment):
    """Body angular acceleration from Euler's equation.
    
    Parameters
    ----------
    J : numpy.ndarray
        Inertia tensor [kg m^3], 3x3, already validated
    omega: array_like
        Body angular velocity [rad/s]
    moment : array_like
        Applied moment about the centre of mass, body axes [N m].
        
    Returns
    -------
    numpy.ndarray
        omega_dot [rad/s^2].
    """
    omega = np.asarray(omega, dtype = float)
    L_body = J @ omega              #angular momentum, body axes
    rhs = np.asarray(moment, dtype = float) - np.cross(omega, L_body)
    return np.linalg.solve(J, rhs)  # solves J @ omega_dot = rhs

def rotational_kinetic_energy(J,omega):
    """Rotational kinetic energy 0.5 * omega. J(omega) [J]."""
    omega= np.asarray(omega, dtype=float)
    return 0.5 * omega @ (J @ omega)

def angular_momentum_earth(q,J, omega):
    """angular mmentum expressed in the earth frame [kg m^2/s].
    
    Constant during torque-free motion, so it is a useful check"""
    return quat_to_rotation_matrix(q) @ (J @ np.asarray(omega, dtype=float))
