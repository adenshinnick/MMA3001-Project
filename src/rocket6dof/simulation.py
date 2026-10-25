"""Flight simulation from ignition to landing.

Phases:

1. On the launch rail: the rocket can only move along the rail, and the pad
   holds it until thrust exceeds weight.
2. Free flight: the full 6-DOF equations, until apogee.
3. Descent: under the parachute (if the rocket has one) or still in free
   flight, until it is back at launch height.

After every step the attitude quaternion is renormalised. RK4 is only
fourth-order accurate where the right-hand side is smooth, and the thrust
curve (linear between data points) has a kink at every data point, so a
step is shortened to end exactly on each one, and on liftoff. The
integration restarts at rail exit and at apogee, where the equations change.

Events (rail exit, apogee, landing) are located inside a step with cubic
Hermite interpolation: the state and its rate of change are known at both
ends of the step, which fixes a cubic in time. Its root is found with
Brent's method. This is exact for a vacuum coast (where height is a
quadratic in time) and accurate to fourth order otherwise.
"""
import numpy as np
from scipy.optimize import brentq

from rocket6dof.atmosphere import isa
from rocket6dof.dynamics import (GRAVITY, OMEGA, POS, QUAT, VEL, make_descent_rhs,
                                 make_rhs, make_state)
from rocket6dof.integrators import STEPPERS
from rocket6dof.quaternions import (quat_from_axis_angle, quat_multiply, quat_normalise,
                                    quat_to_rotation_matrix)


def launch_attitude(elevation_deg, azimuth_deg=90.0):
    """Attitude quaternion for a rail at a given elevation and azimuth.

    Parameters
    ----------
    elevation_deg : float
        Rail angle above the horizontal [deg], 0 to 90.
    azimuth_deg : float
        Compass direction the rail leans towards, clockwise from north [deg].
        Irrelevant at 90 deg elevation. Default 90 (east).

    Returns
    -------
    numpy.ndarray
        Quaternion [w, x, y, z], body to Earth.

    Raises
    ------
    ValueError
        If the elevation is outside 0 to 90 deg.
    """
    if not 0.0 <= elevation_deg <= 90.0:
        raise ValueError(f"rail elevation must be 0 to 90 deg, got {elevation_deg}")
    # First pitch the nose up from east (rotation about -y), then turn it about
    # the vertical so it leans towards the azimuth (east is 90 deg).
    pitch = quat_from_axis_angle([0.0, 1.0, 0.0], -np.radians(elevation_deg))
    heading = quat_from_axis_angle([0.0, 0.0, 1.0], np.radians(90.0 - azimuth_deg))
    return quat_multiply(heading, pitch)


def make_rail_rhs(rocket, rail_direction, **rhs_options):
    """Build f(t, x) for the rocket while it is on the launch rail.

    Parameters
    ----------
    rocket : Rocket
        The rocket.
    rail_direction : numpy.ndarray
        Unit vector along the rail, Earth frame.
    **rhs_options
        Passed to ``make_rhs`` (gravity, wind, launch_altitude).

    Returns
    -------
    callable
        f(t, x) with motion restricted to the rail.
    """
    f_free = make_rhs(rocket, **rhs_options)

    def f(t, x):
        x_dot = f_free(t, x)
        a_along = np.dot(x_dot[VEL], rail_direction)
        speed_along = np.dot(x[VEL], rail_direction)
        if speed_along <= 0.0 and a_along < 0.0:
            a_along = 0.0              # the pad holds the rocket up
        x_dot[VEL] = a_along * rail_direction
        x_dot[QUAT] = 0.0
        x_dot[OMEGA] = 0.0
        return x_dot

    return f


def _hermite(x0, x1, f0, f1, step, tau):
    """State a time ``tau`` into a step, by cubic Hermite interpolation.

    Parameters
    ----------
    x0, x1 : numpy.ndarray
        States at the start and end of the step.
    f0, f1 : numpy.ndarray
        Their rates of change, f(t, x0) and f(t + step, x1).
    step : float
        Step length [s].
    tau : float
        Time into the step, 0 to ``step`` [s].

    Returns
    -------
    numpy.ndarray
    """
    s = tau / step
    h00 = 2 * s**3 - 3 * s**2 + 1      # the four Hermite basis polynomials
    h10 = s**3 - 2 * s**2 + s
    h01 = -2 * s**3 + 3 * s**2
    h11 = s**3 - s**2
    return h00 * x0 + h10 * step * f0 + h01 * x1 + h11 * step * f1


def _locate(g, x0, x1, f0, f1, step):
    """Time into the step where ``g(state)`` crosses zero, and the state there."""
    tau = brentq(lambda tau: g(_hermite(x0, x1, f0, f1, step, tau)), 0.0, step, xtol=1e-14)
    return tau, _hermite(x0, x1, f0, f1, step, tau)


def liftoff_time(rocket, g_along):
    """Time when thrust first exceeds the weight component along the rail [s].

    Until then the pad holds the rocket, so the acceleration jumps from being
    held at zero to growing with thrust: a kink in the right-hand side.
    Returns None if the thrust never exceeds the weight.

    Parameters
    ----------
    rocket : Rocket
    g_along : float
        Component of gravity pulling the rocket back along the rail [m/s^2].
    """
    def excess(t):
        return rocket.motor.thrust(t) - rocket.mass(t) * g_along

    times = rocket.motor.times
    for t0, t1 in zip(times[:-1], times[1:]):
        if excess(t0) <= 0.0 < excess(t1):
            return brentq(excess, t0, t1, xtol=1e-15)
        if excess(t0) > 0.0:
            return None if t0 == 0.0 else t0
    return None


def simulate(rocket, q_launch, rail_length=1.0, h=0.01, t_max=600.0, *,
             gravity=GRAVITY, wind=None, launch_altitude=0.0, h_descent=None,
             method="rk4", align="thrust_points", renormalise=True, stop_at_apogee=False):
    """Simulate a flight from ignition on the pad to landing.

    Parameters
    ----------
    rocket : Rocket
        The rocket to fly.
    q_launch : array_like
        Attitude quaternion on the rail; the nose sets the rail direction.
    rail_length : float
        Distance the rocket travels along the rail before it is free [m].
    h : float
        Time step for the ascent [s].
    t_max : float
        Stop here if the rocket has not landed [s].
    gravity : numpy.ndarray
        Gravitational acceleration, Earth frame [m/s^2].
    wind : callable, optional
        ``wind(z)`` returning the wind velocity, Earth frame [m/s].
    launch_altitude : float
        Pad height above sea level [m].
    h_descent : float, optional
        Time step after apogee [s]. Defaults to ``h``.
    method : str
        ``"rk4"`` (default) or ``"euler"``.
    align : str
        Which kinks in the right-hand side a step is shortened to end on:
        ``"thrust_points"`` (default: liftoff and every data point of the
        thrust curve, where linear interpolation leaves a kink),
        ``"burnout"`` (liftoff and burnout only) or ``"none"``.
    renormalise : bool
        Rescale the quaternion to length 1 after every step (default True).
    stop_at_apogee : bool
        End the run at apogee (faster when only the ascent is needed).

    Returns
    -------
    dict
        ``"t"``: times [s]; ``"x"``: states, one row per time;
        ``"events"``: dict of rail_exit, burnout, apogee, deployment, landing;
        ``"status"``: ``"landed"``, ``"apogee"`` or ``"t_max reached"``;
        ``"n_steps"`` and ``"n_rhs"``: steps taken and right-hand-side calls.

    Raises
    ------
    ValueError
        If ``h``, ``h_descent`` or ``rail_length`` is not positive, the method
        is unknown, or the flight leaves the range of the atmosphere or
        aerodynamic models.
    """
    if not h > 0.0:
        raise ValueError(f"Step size h must be positive, got {h}.")
    h_descent = h if h_descent is None else h_descent
    if not h_descent > 0.0:
        raise ValueError(f"Step size h_descent must be positive, got {h_descent}.")
    if not rail_length > 0.0:
        raise ValueError(f"Rail length must be positive, got {rail_length}.")
    if method not in STEPPERS:
        raise ValueError(f"method must be one of {list(STEPPERS)}, got '{method}'.")
    x = make_state([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], q_launch, [0.0, 0.0, 0.0])
    rail_direction = quat_to_rotation_matrix(x[QUAT]) @ np.array([1.0, 0.0, 0.0])

    # Times where the right-hand side has a kink, so a step can end on them.
    t_lift = liftoff_time(rocket, -np.dot(gravity, rail_direction))
    lift = [] if t_lift is None else [t_lift]
    breakpoints = {"burnout": sorted(lift + [rocket.motor.burn_time]),
                   "thrust_points": sorted(set(lift + list(rocket.motor.times[1:]))),
                   "none": []}
    if align not in breakpoints:
        raise ValueError(f"align must be one of {list(breakpoints)}, got '{align}'.")
    kinks = breakpoints[align]
    stepper = STEPPERS[method]
    options = {"gravity": gravity, "wind": wind, "launch_altitude": launch_altitude}

    n_rhs = 0

    def counted(f):
        """Wrap f so every call is counted (the main measure of cost)."""
        def wrapper(t, state):
            nonlocal n_rhs
            n_rhs += 1
            return f(t, state)
        return wrapper

    f_rail = counted(make_rail_rhs(rocket, rail_direction, **options))
    f_free = counted(make_rhs(rocket, **options))
    f_descent = (counted(make_descent_rhs(rocket, **options))
                 if rocket.parachute is not None else f_free)
    burn_time = rocket.motor.burn_time

    t = 0.0
    times = [t]
    states = [x.copy()]
    events = {}
    phase = "rail"            # "rail", then "free", then "descent"
    status = "t_max reached"
    n_steps = 0

    while t < t_max:
        f = {"rail": f_rail, "free": f_free, "descent": f_descent}[phase]
        step = h_descent if phase == "descent" else h
        for t_kink in kinks:
            if t < t_kink < t + step:
                step = t_kink - t          # end this step exactly on the kink
                break
        t_new = t + step

        x_new = stepper(f, t, x, step)
        if renormalise:
            x_new[QUAT] = quat_normalise(x_new[QUAT])
        n_steps += 1

        if phase == "rail":
            s_new = np.dot(x_new[POS], rail_direction)    # distance along the rail
            if s_new >= rail_length:
                f0, f1 = f(t, x), f(t_new, x_new)
                tau, x_exit = _locate(lambda s: np.dot(s[POS], rail_direction) - rail_length,
                                      x, x_new, f0, f1, step)
                events["rail_exit"] = {"t": t + tau, "speed": np.dot(x_exit[VEL], rail_direction)}
                # Restart at the exit point: the equations change from here on.
                t_new, x_new = t + tau, x_exit
                phase = "free"

        if "burnout" not in events and t_new >= burn_time:
            events["burnout"] = {"t": t_new, "state": x_new.copy()}

        if phase == "free" and "apogee" not in events and x[VEL][2] > 0.0 >= x_new[VEL][2]:
            f0, f1 = f(t, x), f(t_new, x_new)
            tau, x_apo = _locate(lambda s: s[VEL][2], x, x_new, f0, f1, step)
            events["apogee"] = {"t": t + tau, "position": x_apo[POS].copy(), "state": x_apo}
            if stop_at_apogee:
                times.append(t + tau)
                states.append(x_apo)
                status = "apogee"
                break
            if rocket.parachute is not None:
                # Restart the integration at apogee with the parachute open.
                events["deployment"] = {"t": t + tau}
                t_new, x_new = t + tau, x_apo.copy()
                x_new[OMEGA] = 0.0
                phase = "descent"

        if phase != "rail" and x_new[VEL][2] < 0.0 and x_new[POS][2] <= 0.0:
            f0, f1 = f(t, x), f(t_new, x_new)
            tau, x_land = _locate(lambda s: s[POS][2], x, x_new, f0, f1, step)
            x_land[POS][2] = 0.0           # remove round-off from the root finder
            events["landing"] = {"t": t + tau, "position": x_land[POS].copy(),
                                 "speed": float(np.linalg.norm(x_land[VEL]))}
            times.append(t + tau)
            states.append(x_land)
            status = "landed"
            break

        t, x = t_new, x_new
        times.append(t)
        states.append(x)

    return {
        "t": np.array(times),
        "x": np.array(states),
        "events": events,
        "status": status,
        "n_steps": n_steps,
        "n_rhs": n_rhs,
        "settings": {"wind": wind, "launch_altitude": launch_altitude, "gravity": gravity},
    }


def air_speed_and_mach(result):
    """Air-relative speed [m/s] and Mach number [-] at every stored time.

    Parameters
    ----------
    result : dict
        Output of ``simulate``.

    Returns
    -------
    speed, mach : numpy.ndarray
    """
    wind = result["settings"]["wind"]
    alt0 = result["settings"]["launch_altitude"]
    x = result["x"]
    speed = np.empty(len(x))
    mach = np.empty(len(x))
    for i, row in enumerate(x):
        v_air = row[VEL] if wind is None else row[VEL] - wind(row[2])
        speed[i] = np.linalg.norm(v_air)
        mach[i] = speed[i] / isa(alt0 + row[2])[3]
    return speed, mach


def summarise(result):
    """Headline numbers from a flight, as a flat dict.

    Returns
    -------
    dict
        apogee [m], t_apogee [s], max_speed [m/s], max_mach [-],
        rail_exit_speed [m/s], burnout_t [s], landing_t [s],
        landing_distance [m], landing_speed [m/s], n_steps, n_rhs.
        Missing events give NaN.
    """
    ev = result["events"]
    speed, mach = air_speed_and_mach(result)
    nan = float("nan")
    return {
        "apogee": float(ev["apogee"]["position"][2]) if "apogee" in ev else nan,
        "t_apogee": float(ev["apogee"]["t"]) if "apogee" in ev else nan,
        "max_speed": float(speed.max()),
        "max_mach": float(mach.max()),
        "rail_exit_speed": float(ev["rail_exit"]["speed"]) if "rail_exit" in ev else nan,
        "burnout_t": float(ev["burnout"]["t"]) if "burnout" in ev else nan,
        "landing_t": float(ev["landing"]["t"]) if "landing" in ev else nan,
        "landing_distance": (float(np.linalg.norm(ev["landing"]["position"][:2]))
                             if "landing" in ev else nan),
        "landing_speed": float(ev["landing"]["speed"]) if "landing" in ev else nan,
        "n_steps": result["n_steps"],
        "n_rhs": result["n_rhs"],
    }
