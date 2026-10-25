"""Euler angles: the baseline attitude representation, kept for comparison.

The simulator stores attitude as a quaternion. The textbook alternative is
three Euler angles: yaw psi, pitch theta and roll phi, applied in that order
(the aerospace 3-2-1 sequence). Body rates turn into angle rates through

    phi_dot   = p + (q sin(phi) + r cos(phi)) tan(theta)
    theta_dot = q cos(phi) - r sin(phi)
    psi_dot   = (q sin(phi) + r cos(phi)) / cos(theta)

with (p, q, r) the body angular velocity. At theta = +/-90 deg, cos(theta) = 0
and these rates are infinite: gimbal lock. A rocket launched vertically sits
exactly there, which is why the quaternion form is used instead.

Frames for this module
----------------------
Euler angles are defined between the body frame and a local
north-east-down (NED) frame: x north, y east, z down. The simulator's Earth
frame is east-north-up (ENU), so ``quat_to_euler`` converts first. Pitch is
the nose's angle above the horizon.
"""

from math import asin, atan2, cos, sin, tan

import numpy as np

from rocket6dof.quaternions import quat_to_rotation_matrix

# Rotation taking ENU coordinates (x east, y north, z up) to NED (x north, y east, z down).
ENU_TO_NED = np.array([[0.0, 1.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [0.0, 0.0, -1.0]])

# The simulator's body frame has x through the nose, but its y and z are not
# the aircraft-style right/down axes. Body angular rates are used directly,
# which is valid because any right-handed body frame works with 3-2-1 angles.


def euler_rates(angles, omega):
    """Rates of change of the 3-2-1 Euler angles [rad/s].

    Parameters
    ----------
    angles : array_like
        [phi, theta, psi]: roll, pitch, yaw [rad].
    omega : array_like
        Body angular velocity [p, q, r] [rad/s].

    Returns
    -------
    numpy.ndarray
        [phi_dot, theta_dot, psi_dot] [rad/s].

    Raises
    ------
    ZeroDivisionError
        At exactly theta = +/-90 deg (gimbal lock) in exact arithmetic;
        near it the rates become very large instead.
    """
    phi, theta, _ = angles
    p, q, r = omega
    sp, cp = sin(phi), cos(phi)
    ct = cos(theta)
    return np.array([
        p + (q * sp + r * cp) * tan(theta),
        q * cp - r * sp,
        (q * sp + r * cp) / ct,
    ])


def rotation_to_euler(R_body_to_ned):
    """3-2-1 Euler angles [phi, theta, psi] from a body-to-NED rotation matrix [rad]."""
    R = R_body_to_ned
    theta = asin(max(-1.0, min(1.0, -R[2, 0])))
    phi = atan2(R[2, 1], R[2, 2])
    psi = atan2(R[1, 0], R[0, 0])
    return np.array([phi, theta, psi])


def euler_to_rotation(angles):
    """Body-to-NED rotation matrix for 3-2-1 Euler angles [phi, theta, psi]."""
    phi, theta, psi = angles
    cph, sph = cos(phi), sin(phi)
    cth, sth = cos(theta), sin(theta)
    cps, sps = cos(psi), sin(psi)
    return np.array([
        [cth * cps, sph * sth * cps - cph * sps, cph * sth * cps + sph * sps],
        [cth * sps, sph * sth * sps + cph * cps, cph * sth * sps - sph * cps],
        [-sth, sph * cth, cph * cth],
    ])


def quat_to_euler(q):
    """3-2-1 Euler angles of a simulator attitude quaternion [rad].

    The quaternion maps body to ENU; this converts to body-to-NED first.
    """
    return rotation_to_euler(ENU_TO_NED @ quat_to_rotation_matrix(q))
