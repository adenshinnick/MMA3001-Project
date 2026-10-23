""" Quaternion tools for describing the rockets attitude.

Conventions, used throughout the simulator

-Quaternions are scalar-first : q = [w,x,y,z], with length 1.
- The attitude quaternion rotates vectros from the body frame into the 
earth frame: v_earth = R(q) @ v_body
- Earth frame: x east, y north, z up
- Body frame: x along the rocket's axid, out through the nose; y and z 
perpendicular to it.
- Angular velocity omega is expressed in the body frame [rad/s]
"""
import numpy as np

def quat_multiply(p,q):
    """Hamilton product p cross q, which combines two rotations.
    
    Parameters
    ----------
    p,q: array_like
        Quaternions [w, x, y, z]
    
    Returns
    -------
    numpy.ndarray
        The product p cross q. Applying q, then p, is the same as applying p cross q.    
    """
    p = np.asarray(p, dtype = float)
    q = np.asarray(q, dtype=float)
    pw, pv = p[0], p[1:]
    qw, qv = q[0], q[1:]
    w = pw * qw - np.dot(pv,qv)
    v = pw * qv + qw * pv + np.cross(pv,qv)
    return np.concatenate(([w],v))

def quat_conjugate(q):
    """Conjugate [w, -x, -y, -z]; the reverse rotation for a unit quaternion"""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype = float)

def quat_normalise(q):
    """ Scale a quaternion to length 1.
    
    Raises
    ------
    ValuesError
        If the quaternion has (almost) zero length.
    """
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
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
    n = axis/axis_length
    half=0.5*angle
    return np.concatenate(([np.cos(half)], np.sin(half)*n))

def quat_to_rotation_matrix(q):
    """" 3x3 rotation matrix R with v_earth = R @ v_body."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])

def rotate_vector(q,v):
    """Rotate vector v from the body frame into the earth frame, using q cross (0,v) cross q*.
    
    Gives the same result as 'quat_to_rotation_matrix(q) @ v', 
    but by a different reout, which makes it a useful independent check.
    """

    v_quat = np.concatenate(([0.0], v))
    return quat_multiply(quat_multiply(q, v_quat), quat_conjugate(q))[1:]

def quat_derivative(q, omega):
    """Rate of change of the attitude quaternion, dq/dt = 0.5 cross (0,omega)
    
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
    omega_quat = np.concatenate(([0.0], omega))
    return 0.5 * quat_multiply(q, omega_quat)
    