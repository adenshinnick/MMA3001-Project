"""Quaternion tools for describing the rocket's attitude.

A quaternion stores a rotation as four numbers. Unlike three Euler angles,
it has no singularity when the rocket points straight up, which is exactly
how it launches.

Conventions, used throughout the simulator:

- Quaternions are scalar-first: q = [w, x, y, z], with length 1.
- The attitude quaternion rotates vectors from the body frame into the
  earth frame: v_earth = R(q) @ v_body
- Earth frame: x east, y north, z up
- Body frame: x along the rocket's axis, out through the nose; y and z 
perpendicular to it.
- Angular velocity omega is expressed in the body frame [rad/s]
"""
import numpy as np


def cross3(a, b):
    """Cross product of two 3-vectors, a x b.

    Same result as ``numpy.cross(a, b)``, but written out for 3-vectors only.
    numpy.cross handles arrays of any shape, and that generality costs about
    10 microseconds per call. Profiling (cProfile) showed it was the
    largest single cost in the right-hand side, which calls it twice per
    evaluation. See the performance section of ``scripts/make_figures.py``.
    """
    return np.array([a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]])

def quat_multiply(p,q):
    """Hamilton product p ⊗ q, which combines two rotations.
    
    Parameters
    ----------
    p,q: array_like
        Quaternions [w, x, y, z]
    
    Returns
    -------
    numpy.ndarray
        The product p ⊗ q. Applying q, then p, is the same as applying p ⊗ q.
    """
    p = np.asarray(p, dtype = float)
    q = np.asarray(q, dtype=float)
    # Split each quaternion into its scalar part (w) and vector part (x, y, z).
    pw, pv = p[0], p[1:]
    qw, qv = q[0], q[1:]
    w = pw * qw - np.dot(pv, qv)               # scalar part of the product
    v = pw * qv + qw * pv + cross3(pv, qv)     # vector part of the product
    return np.concatenate(([w],v))

def quat_conjugate(q):
    """Conjugate [w, -x, -y, -z]; the reverse rotation for a unit quaternion"""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype = float)

def quat_normalise(q):
    """ Scale a quaternion to length 1.
    
    Raises
    ------
    ValueError
        If the quaternion has (almost) zero length.
    """
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if norm < 1e-12:                 # a zero quaternion has no direction to keep
        raise ValueError("Cannot normalise a quaternion with the zero length.")
    return q/norm

def quat_from_axis_angle(axis,angle):
    """Quaternion for a rotation by ''angle'' [rad] about ''axis''.
    the axis does not need to be unit length. Positive angles follow the right hand rule.
    
    Raises
    ------
    ValueError
        If the axis is the zero vector
    """
    axis = np.asarray(axis, dtype=float)
    axis_length = np.linalg.norm(axis)
    if axis_length < 1e-12:
        raise ValueError("Rotation axis must not be the zero vector.")
    n = axis/axis_length             # unit vector along the axis
    half=0.5*angle                   # quaternions use half the rotation angle
    return np.concatenate(([np.cos(half)], np.sin(half)*n))

def quat_to_rotation_matrix(q):
    """3x3 rotation matrix R with v_earth = R @ v_body."""
    # Standard formula for the rotation matrix of a unit quaternion. Its
    # columns are the rocket's body axes written in earth coordinates, so the
    # first column is the direction the nose points.
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])

def rotate_vector(q,v):
    """Rotate vector v from the body frame into the earth frame, using q ⊗ (0,v) ⊗ q*.
    
    Gives the same result as 'quat_to_rotation_matrix(q) @ v', 
    but by a different route, which makes it a useful independent check.
    """

    v_quat = np.concatenate(([0.0], v))     # write v as a quaternion with zero scalar part
    return quat_multiply(quat_multiply(q, v_quat), quat_conjugate(q))[1:]

def quat_derivative(q, omega):
    """Rate of change of the attitude quaternion, dq/dt = 0.5 q ⊗ (0, omega).
    
    Parameters
    ----------
    q : array_like
        Attitude quaternion [w, x, y, z]
    omega : array_like
        Angular velocity in the body frame [rad/s]
    
    Returns
    -------
    numpy.ndarray
        dq/dt [1/s]
    """
    # omega is in body axes, so it multiplies on the right of q.
    omega_quat = np.concatenate(([0.0], omega))
    return 0.5 * quat_multiply(q, omega_quat)
    