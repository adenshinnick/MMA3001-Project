"""Helpers for the studies in scripts/make_figures.py: convergence,
sensitivity runs and timing."""

import time

import numpy as np
from rocket6dof.dynamics import GRAVITY, make_rhs, make_state
from rocket6dof.motor import Motor
from rocket6dof.quaternions import cross3
from rocket6dof.rocket import Rocket
from rocket6dof.simulation import launch_attitude, simulate, summarise
from rocket6dof.wind import Wind


class ScaledDrag:
    """A drag model multiplied by a constant factor (for the sensitivity study)."""

    def __init__(self, model, factor):
        self.model = model
        self.factor = factor

    def __call__(self, mach):
        return self.factor * self.model(mach)


def variant(rocket, cd_factor=1.0, mass_factor=1.0, thrust_factor=1.0, drag_model=None):
    """A copy of ``rocket`` with scaled drag, dry mass and thrust.

    Parameters
    ----------
    rocket : Rocket
        The nominal rocket (must have geometry).
    cd_factor, mass_factor, thrust_factor : float
        Multiply the drag coefficient, the dry mass (and dry inertia), and the
        thrust curve (so the total impulse).
    drag_model : callable, optional
        Replace the drag model before scaling.

    Returns
    -------
    Rocket
    """
    m = rocket.motor
    motor = Motor(m.name, m.times, m.thrusts * thrust_factor, m.propellant_mass,
                  m.total_mass, m.diameter, m.length)
    base_drag = rocket.drag_model if drag_model is None else drag_model
    drag = base_drag if cd_factor == 1.0 else ScaledDrag(base_drag, cd_factor)
    return Rocket(rocket.dry_mass * mass_factor, rocket.inertia * mass_factor, motor,
                  geometry=rocket.geometry, drag_model=drag, dry_cg=rocket.dry_cg,
                  motor_station=rocket.motor_station, parachute=rocket.parachute,
                  name=rocket.name)


def apogee_fixed_step(rocket, h, align="thrust_points", elevation_deg=90.0,
                      rail_length=1.0, gravity=GRAVITY):
    """Apogee [m], right-hand-side calls and run time [s] for a fixed-step ascent."""
    t0 = time.perf_counter()
    result = simulate(rocket, launch_attitude(elevation_deg), rail_length=rail_length, h=h,
                      align=align, gravity=gravity, stop_at_apogee=True)
    runtime = time.perf_counter() - t0
    return float(result["events"]["apogee"]["position"][2]), result["n_rhs"], runtime


def observed_order(h, error):
    """Observed order of accuracy between successive step sizes: log(e1/e2) / log(h1/h2)."""
    h, error = np.asarray(h), np.asarray(error)
    return np.log(error[:-1] / error[1:]) / np.log(h[:-1] / h[1:])


NOMINAL = {"cd_factor": 1.0, "mass_factor": 1.0, "thrust_factor": 1.0,
           "wind_speed": 0.0, "wind_direction": 270.0, "elevation": 90.0,
           "azimuth": 90.0, "launch_altitude": 0.0}


def fly(rocket, params, h=0.02, h_descent=0.1, full=True):
    """Fly one variant of the rocket and return its summary.

    Parameters
    ----------
    rocket : Rocket
        Nominal rocket.
    params : dict
        Any keys of ``NOMINAL``; missing keys take the nominal value.
    h, h_descent : float
        Ascent and descent step sizes [s].
    full : bool
        Fly to landing (True) or stop at apogee (False).

    Returns
    -------
    dict
        ``summarise`` output.
    """
    p = {**NOMINAL, **params}
    r = variant(rocket, p["cd_factor"], p["mass_factor"], p["thrust_factor"])
    wind = Wind(p["wind_speed"], p["wind_direction"]) if p["wind_speed"] > 0 else None
    result = simulate(r, launch_attitude(p["elevation"], p["azimuth"]), rail_length=1.0,
                      h=h, h_descent=h_descent, wind=wind, launch_altitude=p["launch_altitude"],
                      stop_at_apogee=not full)
    return summarise(result)


def time_rhs(rocket, n=2000):
    """Average time of one right-hand-side call at a typical coasting state [s]."""
    f = make_rhs(rocket, wind=Wind(5.0, 270.0))
    x = make_state([0.0, 0.0, 300.0], [3.0, 0.0, 150.0], launch_attitude(88.0),
                   [0.0, 0.1, -0.05])
    f(2.0, x)                                  # first call outside the timing
    t0 = time.perf_counter()
    for _ in range(n):
        f(2.0, x)
    return (time.perf_counter() - t0) / n


def time_cross_products(n=20000):
    """Time per call of ``numpy.cross`` and of ``cross3`` on 3-vectors [s].

    Profiling showed ``numpy.cross`` was the biggest single cost in the
    right-hand side; this measures what replacing it with ``cross3`` saves.
    """
    a, b = np.array([0.3, -1.2, 2.5]), np.array([-0.7, 0.4, 1.1])
    times = []
    for func in (np.cross, cross3):
        func(a, b)
        t0 = time.perf_counter()
        for _ in range(n):
            func(a, b)
        times.append((time.perf_counter() - t0) / n)
    return times[0], times[1]


# Hand count of floating-point operations in one right-hand-side call
# (sqrt, log, exp and powers counted as one each). Approximate, to compare
# with the measured time per call.
FLOP_ESTIMATE = {
    "quat_to_rotation_matrix": 33,
    "R^T v_air, R F / m + g": 36,
    "isa (troposphere) + viscosity": 20,
    "aerodynamic_loads (drag, normal force, CP, damping)": 140,
    "quat_derivative (quaternion product)": 32,
    "inertia_at (motor + parallel axis)": 35,
    "angular_acceleration (J w, cross, 3x3 solve)": 60,
    "thrust, mass, CG (interpolation)": 25,
}
