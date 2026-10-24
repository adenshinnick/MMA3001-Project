"""flight simulation from ignition to landing.

Phases:

1. on the launchrail: the rocket can only move along the rail
2. free flight: the full 6-dof equations, until it reaches launch height

after every step the attitude quaternion is renormalised. A step is shortened 
so that one ends exactly at motor burnout, whree the thrust curve has a kink.
Events are located within a step by assumign constant acceleration accros that 
step, which is exactly a vacuum coast.
"""
import numpy as np

from rocket6dof.dynamics import OMEGA, POS, QUAT, VEL, make_rhs, make_state
from rocket6dof.integrators import rk4_step
from rocket6dof.quaternions import quat_normalise, quat_to_rotation_matrix

def make_rail_rhs(rocket, rail_direction):
    """Build f(t,x) for the rocket while it is on the launch rail.
    
    Parameters
    ----------
    rocket: Rocket
        the rocket.
    rail_direction: numpy.ndarray
        Unit vector along the rail, Earth frame.
        
    Returns
    -------
    callable
        f(t,x) with motion restricted to therail
    """
    f_free = make_rhs(rocket)

    def f(t,x):
        x_dot = f_free(t,x)
        a_along = np.dot(x_dot[VEL], rail_direction)
        speed_along = np.dot(x[VEL], rail_direction)
        if speed_along <= 0.0 and a_along < 0.0:
            a_along=0.0
        x_dot[VEL] = a_along* rail_direction
        x_dot[QUAT] = 0.0
        x_dot[OMEGA] = 0.0
        return x_dot

    return f

def _point_in_step(x_old,a,tau):
    """Position and velocity a time tau into a step, assuming constant 
    acceleration a across the step."""

    position = x_old[POS] + x_old[VEL] * tau + 0.5 *a*tau**2
    velocity = x_old[VEL] + a *tau
    return position, velocity 

def simulate(rocket, q_launch, rail_length=1.0, h=0.01, t_max=300.0):
    """Simulate a flight from ignition on the pad to landing. 
    
    Parameters
    ----------
    rocket: Rocket
        The rocket to fly
    q_launch: array_like
        Attitude quaterniion on the rail; the nose sets the rail direction
    rail_length : float
        Distance the rocket travels along the rail before it is free [m].
    h : float
        Normal time step [s]
    t_max : float
        Stop here if the rocket has not landed [s].

    Returns
    -------
    dict
        ``"t"``: times [s], ``"x"``: states (one row per time),
        ``"events"``: dict of rail_exit, burnout, apogee and landing,
        ``"status"``: ``"landed"`` or ``"t_max reached"``.
    
    Raises
    ------
    ValueError
        if ''h'' or ''rail_length'' is not positive.
    """
    if not h > 0.0:
        raise ValueError(f"Step size h must be positive, got {h}.")
    if not rail_length > 0.0:
        raise ValueError(f"Rail length must be positive, got {rail_length}.")

    x = make_state([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], q_launch, [0.0, 0.0, 0.0])
    rail_direction = quat_to_rotation_matrix(x[QUAT]) @ np.array([1.0,0.0,0.0])
    f_rail = make_rail_rhs(rocket, rail_direction)
    f_free = make_rhs(rocket)
    burn_time = rocket.motor.burn_time

    t=0.0
    times=[t]
    states = [x.copy()]
    events = {}
    on_rail = True
    status = "t_max reached"

    while t < t_max:
        # shorten the step so that one ends exactly at burnout.
        if t < burn_time < t+h:
            step, t_new = burn_time - t, burn_time
        else:
            step, t_new = h, t+h
    
        x_new = rk4_step(f_rail if on_rail else f_free, t, x, step)
        x_new[QUAT] = quat_normalise(x_new[QUAT])
        a = (x_new[VEL] - x[VEL])/step #average acceleration over the step

        if on_rail:
            s_old=np.dot(x[POS], rail_direction) #distance along rail
            s_new =np.dot(x_new[POS], rail_direction)
            if s_new >= rail_length: 
                fraction = (rail_length - s_old)/(s_new - s_old)
                v_exit = x[VEL] + fraction * (x_new[VEL] - x[VEL])
                events["rail_exit"] = {
                    "t" : t + fraction *step,
                    "speed": np.dot(v_exit, rail_direction)
                }
                on_rail = False

        if "burnout" not in events and t_new >= burn_time:
            events["burnout"] = {"t":t_new, "state": x_new.copy()}

        vz_old, vz_new = x[VEL][2], x_new[VEL][2]
        if "apogee" not in events and not on_rail and vz_old>0.0>= vz_new:
            tau = -vz_old / a[2]
            position, _ = _point_in_step(x,a,tau)
            events["apogee"] = {"t":t+tau, "position": position}

        if not on_rail and vz_new < 0.0 and x_new[POS][2] <= 0.0:
            z_old = x[POS][2]
            #Root of z_old + vz_old*tau _0.5*a_z*tau^2 = 0 written so
            # that it never divides by a_z (which can be zero).
            tau = 2.0 *z_old/(-vz_old + np.sqrt(vz_old**2 - 2.0 * a[2]*z_old)) 
            position, velocity = _point_in_step(x,a,tau)
            x_land = x_new.copy()
            x_land[POS] = position
            x_land[VEL] = velocity
            events["landing"] = {"t": t+tau,"position": position }
            times.append(t+tau)
            states.append(x_land)
            status = "landed"
            break
        t,x = t_new, x_new
        times.append(t)
        states.append(x)

    return{ 
        "t": np.array(times),
        "x" : np.array(states),
        "events":events,
        "status": status,
    }