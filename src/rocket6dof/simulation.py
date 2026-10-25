"""Flight simulation from ignition to landing.

Phases:

1. On the launch rail: the rocket can only move along the rail, and the pad
   holds it until thrust exceeds weight.
2. Free flight: the full 6-DOF equations, until apogee.
3. Descent: under the parachute (if the rocket has one) until it is back at
   launch height.

After every step the attitude quaternion is renormalised. RK4 is only
fourth-order accurate where the right-hand side is smooth; the thrust curve
(linear between data points) has a kink at every data point, and liftoff is
another, so a step is shortened to end exactly on each one. Events are
located within a step by assuming constant acceleration across that step,
which is exact for a vacuum coast.
"""
import numpy as np

from rocket6dof.atmosphere import isa
from rocket6dof.dynamics import (GRAVITY, OMEGA, POS, QUAT, VEL, make_descent_rhs,
                                 make_rhs, make_state)
from rocket6dof.integrators import rk4_step
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
        Default 90 (east).

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
    # Pitch the nose up from east (rotation about -y), then turn it about the vertical.
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


def _point_in_step(x_old, a, tau):
    """Position and velocity a time tau into a step, assuming constant
    acceleration a across the step."""
    position = x_old[POS] + x_old[VEL] * tau + 0.5 * a * tau**2
    velocity = x_old[VEL] + a * tau
    return position, velocity


def liftoff_time(rocket, g_along):
    """Time when thrust first exceeds the weight component along the rail [s].

    Found by bisection inside the first thrust-curve segment where it happens.
    Returns None if the thrust never exceeds the weight.
    """
    def excess(t):
        return rocket.motor.thrust(t) - rocket.mass(t) * g_along

    times = rocket.motor.times
    for t0, t1 in zip(times[:-1], times[1:]):
        if excess(t0) > 0.0:
            return None if t0 == 0.0 else t0
        if excess(t1) > 0.0:
            for _ in range(60):        # halve the interval 60 times
                mid = 0.5 * (t0 + t1)
                t0, t1 = (t0, mid) if excess(mid) > 0.0 else (mid, t1)
            return t1
    return None


def simulate(rocket, q_launch, rail_length=1.0, h=0.01, t_max=600.0, *,
             gravity=GRAVITY, wind=None, launch_altitude=0.0, h_descent=None,
             align="thrust_points", stop_at_apogee=False):
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
    align : str
        Kinks a step is shortened to end on: ``"thrust_points"`` (default:
        liftoff and every thrust-curve point), ``"burnout"`` (liftoff and
        burnout) or ``"none"``.
    stop_at_apogee : bool
        End the run at apogee.

    Returns
    -------
    dict
        ``"t"``: times [s]; ``"x"``: states, one row per time;
        ``"events"``: rail_exit, burnout, apogee, landing;
        ``"status"``: ``"landed"``, ``"apogee"`` or ``"t_max reached"``;
        ``"n_steps"``, ``"n_rhs"``: steps and right-hand-side calls.

    Raises
    ------
    ValueError
        If a setting is invalid, or the flight leaves the range of the
        atmosphere or aerodynamic models.
    """
    if not h > 0.0:
        raise ValueError(f"Step size h must be positive, got {h}.")
    h_descent = h if h_descent is None else h_descent
    if not h_descent > 0.0:
        raise ValueError(f"Step size h_descent must be positive, got {h_descent}.")
    if not rail_length > 0.0:
        raise ValueError(f"Rail length must be positive, got {rail_length}.")
    options = {"gravity": gravity, "wind": wind, "launch_altitude": launch_altitude}

    x = make_state([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], q_launch, [0.0, 0.0, 0.0])
    rail_direction = quat_to_rotation_matrix(x[QUAT]) @ np.array([1.0, 0.0, 0.0])

    # Times where the right-hand side has a kink, so a step can end on them.
    t_lift = liftoff_time(rocket, -np.dot(gravity, rail_direction))
    lift = [] if t_lift is None else [t_lift]
    kink_options = {"thrust_points": sorted(set(lift + list(rocket.motor.times[1:]))),
                    "burnout": sorted(lift + [rocket.motor.burn_time]),
                    "none": []}
    if align not in kink_options:
        raise ValueError(f"align must be one of {list(kink_options)}, got '{align}'.")
    kinks = kink_options[align]

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
    f_descent = counted(make_descent_rhs(rocket, **options)) if rocket.parachute else f_free
    burn_time = rocket.motor.burn_time

    t = 0.0
    times, states = [t], [x.copy()]
    events = {}
    phase = "rail"                     # then "free", then "descent"
    status = "t_max reached"
    n_steps = 0

    while t < t_max:
        f = {"rail": f_rail, "free": f_free, "descent": f_descent}[phase]
        step = h_descent if phase == "descent" else h
        for t_kink in kinks:
            if t < t_kink < t + step:
                step = t_kink - t      # end this step exactly on the kink
                break
        t_new = t + step

        x_new = rk4_step(f, t, x, step)
        x_new[QUAT] = quat_normalise(x_new[QUAT])
        n_steps += 1
        a = (x_new[VEL] - x[VEL]) / step   # average acceleration over the step

        if phase == "rail":
            s_old = np.dot(x[POS], rail_direction)       # distance along the rail
            s_new = np.dot(x_new[POS], rail_direction)
            if s_new >= rail_length:
                # Shorten this step so it ends at the end of the rail, as at
                # burnout: guess the length by linear interpolation, redo the
                # step, and repeat a few times until it lands on the rail end.
                tau_low, s_low = 0.0, s_old
                tau_high, s_high = step, s_new
                for _ in range(4):
                    tau = tau_low + (rail_length - s_low) / (s_high - s_low) * (tau_high - tau_low)
                    x_exit = rk4_step(f, t, x, tau)
                    s_exit = np.dot(x_exit[POS], rail_direction)
                    if s_exit < rail_length:
                        tau_low, s_low = tau, s_exit
                    else:
                        tau_high, s_high = tau, s_exit
                t_new, x_new = t + tau, x_exit
                events["rail_exit"] = {"t": t_new, "speed": np.dot(x_new[VEL], rail_direction)}
                phase = "free"           # free flight starts exactly at the rail exit

        if "burnout" not in events and t_new >= burn_time:
            events["burnout"] = {"t": t_new, "state": x_new.copy()}

        vz_old, vz_new = x[VEL][2], x_new[VEL][2]
        if phase == "free" and "apogee" not in events and vz_old > 0.0 >= vz_new:
            tau = -vz_old / a[2]
            position, velocity = _point_in_step(x, a, tau)
            events["apogee"] = {"t": t + tau, "position": position}
            if stop_at_apogee or rocket.parachute is not None:
                x_apo = x_new.copy()
                x_apo[POS], x_apo[VEL] = position, velocity
                t_new, x_new = t + tau, x_apo
                if stop_at_apogee:
                    times.append(t_new)
                    states.append(x_new)
                    status = "apogee"
                    break
                x_new[OMEGA] = 0.0         # the parachute opens: restart from apogee
                phase = "descent"

        if phase != "rail" and x_new[VEL][2] < 0.0 and x_new[POS][2] <= 0.0:
            z_old, vz_old = x[POS][2], x[VEL][2]
            # Root of z_old + vz_old*tau + 0.5*a_z*tau^2 = 0, written so that it
            # never divides by a_z (which can be zero).
            tau = 2.0 * z_old / (-vz_old + np.sqrt(vz_old**2 - 2.0 * a[2] * z_old))
            position, velocity = _point_in_step(x, a, tau)
            x_land = x_new.copy()
            x_land[POS], x_land[VEL] = position, velocity
            events["landing"] = {"t": t + tau, "position": position,
                                 "speed": float(np.linalg.norm(velocity))}
            times.append(t + tau)
            states.append(x_land)
            status = "landed"
            break

        t, x = t_new, x_new
        times.append(t)
        states.append(x)

    return {"t": np.array(times), "x": np.array(states), "events": events, "status": status,
            "n_steps": n_steps, "n_rhs": n_rhs,
            "settings": {"wind": wind, "launch_altitude": launch_altitude}}


def air_speed_and_mach(result):
    """Air-relative speed [m/s] and Mach number [-] at every stored time."""
    wind = result["settings"]["wind"]
    alt0 = result["settings"]["launch_altitude"]
    speed, mach = [], []
    for row in result["x"]:
        v_air = row[VEL] if wind is None else row[VEL] - wind(row[2])
        speed.append(np.linalg.norm(v_air))
        mach.append(speed[-1] / isa(alt0 + row[2])[3])
    return np.array(speed), np.array(mach)


def attitude_history(result):
    """Tilt from vertical and angle of attack [deg] at every stored time.

    The nose direction in the Earth frame is the body x axis rotated by the
    attitude quaternion. The tilt is its angle from straight up. The angle of
    attack is the angle between the nose and the air-relative velocity
    (velocity minus wind): zero when the rocket points straight into the
    airflow it feels. Where there is no airflow the angle of attack is
    undefined and is returned as NaN.
    """
    wind = result["settings"]["wind"]
    tilt, aoa = [], []
    for row in result["x"]:
        nose = quat_to_rotation_matrix(row[QUAT]) @ np.array([1.0, 0.0, 0.0])
        tilt.append(np.degrees(np.arccos(np.clip(nose[2], -1.0, 1.0))))
        v_air = row[VEL] if wind is None else row[VEL] - wind(row[2])
        speed = np.linalg.norm(v_air)
        if speed < 1e-6:                  # no airflow (on the pad, or exactly at apogee)
            aoa.append(np.nan)
        else:
            aoa.append(np.degrees(np.arccos(np.clip(np.dot(nose, v_air) / speed, -1.0, 1.0))))
    return np.array(tilt), np.array(aoa)


def summarise(result):
    """Headline numbers from a flight, as a dict (missing events give NaN)."""
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
