"""Studies built on the simulator: convergence, integrators, sensitivity, Monte Carlo.

These functions return numbers; ``scripts/make_figures.py`` turns them into
the figures and tables of the report. Keeping the studies here (and not in
the script) means they can be tested.
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager

import numpy as np
from scipy.integrate import solve_ivp

from rocket6dof.dynamics import GRAVITY, POS, QUAT, VEL, make_rhs, make_state
from rocket6dof.motor import Motor
from rocket6dof.quaternions import quat_to_rotation_matrix
from rocket6dof.rocket import Rocket
from rocket6dof.simulation import (launch_attitude, liftoff_time, make_rail_rhs, simulate,
                                   summarise)
from rocket6dof.wind import Wind


# ---------------------------------------------------------------------------
# Building variants of a rocket (for sensitivity and Monte Carlo runs)
# ---------------------------------------------------------------------------

class ScaledDrag:
    """A drag model multiplied by a constant factor (for uncertainty studies)."""

    def __init__(self, model, factor):
        self.model = model
        self.factor = factor

    def __call__(self, mach, reynolds):
        return self.factor * self.model(mach, reynolds)


def variant(rocket, cd_factor=1.0, mass_factor=1.0, thrust_factor=1.0, drag_model=None):
    """A copy of ``rocket`` with scaled drag, dry mass and thrust.

    Parameters
    ----------
    rocket : Rocket
        The nominal rocket (must have geometry).
    cd_factor : float
        Multiplies the drag coefficient.
    mass_factor : float
        Multiplies the dry mass and dry inertia (same mass distribution).
    thrust_factor : float
        Multiplies the thrust curve, and so the total impulse.
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


# ---------------------------------------------------------------------------
# Convergence and integrator comparison
# ---------------------------------------------------------------------------

def apogee_fixed_step(rocket, h, method="rk4", align="thrust_points", elevation_deg=90.0,
                      rail_length=1.0, gravity=GRAVITY):
    """Apogee [m], RHS calls and run time [s] for a fixed-step ascent."""
    t0 = time.perf_counter()
    result = simulate(rocket, launch_attitude(elevation_deg), rail_length=rail_length, h=h,
                      method=method, align=align, gravity=gravity, stop_at_apogee=True)
    runtime = time.perf_counter() - t0
    return float(result["events"]["apogee"]["position"][2]), result["n_rhs"], runtime


def apogee_rk45(rocket, rtol, atol=1e-9, elevation_deg=90.0, rail_length=1.0, gravity=GRAVITY):
    """Apogee with SciPy's adaptive RK45 (Dormand-Prince), for comparison.

    The flight is split at the same kinks the fixed-step runs align to
    (liftoff and every thrust-curve point) and at rail exit, so all methods
    face the same smooth pieces. The quaternion is not renormalised (SciPy
    offers no hook between steps), which is acceptable over a 13 s ascent.

    Parameters
    ----------
    rocket : Rocket
    rtol, atol : float
        Relative and absolute error tolerances for each step.
    elevation_deg, rail_length, gravity
        Launch settings.

    Returns
    -------
    apogee : float
        [m]
    n_rhs : int
        Right-hand-side calls.
    runtime : float
        [s]
    """
    t_start = time.perf_counter()
    q0 = launch_attitude(elevation_deg)
    x = make_state([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], q0, [0.0, 0.0, 0.0])
    rail = quat_to_rotation_matrix(q0) @ np.array([1.0, 0.0, 0.0])
    f_rail = make_rail_rhs(rocket, rail, gravity=gravity)
    f_free = make_rhs(rocket, gravity=gravity)

    t_lift = liftoff_time(rocket, -np.dot(gravity, rail))
    cuts = sorted(set([t_lift] + list(rocket.motor.times[1:])))
    cuts = [c for c in cuts if c > 0.0] + [600.0]

    def rail_exit(t, s):
        return np.dot(s[POS], rail) - rail_length
    rail_exit.terminal, rail_exit.direction = True, 1

    def apogee(t, s):
        return s[VEL][2]
    apogee.terminal, apogee.direction = True, -1

    t, n_rhs, on_rail = 0.0, 0, True
    for t_cut in cuts:
        while t < t_cut:
            f, event = (f_rail, rail_exit) if on_rail else (f_free, apogee)
            sol = solve_ivp(f, (t, t_cut), x, method="RK45", rtol=rtol, atol=atol, events=event)
            n_rhs += sol.nfev
            if sol.t_events[0].size:               # the event ended this piece
                t, x = sol.t_events[0][0], sol.y_events[0][0]
                if on_rail:
                    on_rail = False
                    continue
                return float(x[2]), n_rhs, time.perf_counter() - t_start
            t, x = t_cut, sol.y[:, -1]
    raise RuntimeError("no apogee found")


def observed_order(h, error):
    """Observed order of accuracy between successive step sizes: log2(e(h)/e(h/2))."""
    h, error = np.asarray(h), np.asarray(error)
    return np.log(error[:-1] / error[1:]) / np.log(h[:-1] / h[1:])


# ---------------------------------------------------------------------------
# Sensitivity and Monte Carlo
# ---------------------------------------------------------------------------

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
        Any keys of ``NOMINAL`` (missing keys take the nominal value).
    h, h_descent : float
        Ascent and descent step sizes [s].
    full : bool
        Fly to landing (True) or stop at apogee (False).

    Returns
    -------
    dict
        ``summarise`` output plus ``landing_x`` and ``landing_y`` [m].
    """
    p = {**NOMINAL, **params}
    r = variant(rocket, p["cd_factor"], p["mass_factor"], p["thrust_factor"])
    wind = Wind(p["wind_speed"], p["wind_direction"], exponent=1.0 / 7.0) if p["wind_speed"] > 0 else None
    result = simulate(r, launch_attitude(p["elevation"], p["azimuth"]), rail_length=1.0,
                      h=h, h_descent=h_descent, wind=wind, launch_altitude=p["launch_altitude"],
                      stop_at_apogee=not full)
    out = summarise(result)
    land = result["events"].get("landing")
    out["landing_x"] = float(land["position"][0]) if land else float("nan")
    out["landing_y"] = float(land["position"][1]) if land else float("nan")
    out["apogee_x"] = float(result["events"]["apogee"]["position"][0])
    out["apogee_y"] = float(result["events"]["apogee"]["position"][1])
    return out


def sample_inputs(n, seed=2026):
    """Random input sets for the Monte Carlo study (fixed seed, so repeatable).

    Distributions (assumptions, stated in the report):

    - drag factor: normal, mean 1, standard deviation 5 %
    - dry mass factor: normal, mean 1, sd 2 %
    - thrust factor: normal, mean 1, sd 2 % (motor-to-motor impulse scatter)
    - wind speed at 10 m: uniform 0 to 6 m/s, from a uniformly random direction,
      1/7 power-law profile
    - rail elevation: normal, mean 87 deg, sd 1 deg (capped at 90)
    - rail azimuth: uniform, 0 to 360 deg

    Returns
    -------
    list of dict
    """
    rng = np.random.default_rng(seed)
    return [{
        "cd_factor": rng.normal(1.0, 0.05),
        "mass_factor": rng.normal(1.0, 0.02),
        "thrust_factor": rng.normal(1.0, 0.02),
        "wind_speed": rng.uniform(0.0, 6.0),
        "wind_direction": rng.uniform(0.0, 360.0),
        "elevation": min(90.0, rng.normal(87.0, 1.0)),
        "azimuth": rng.uniform(0.0, 360.0),
    } for _ in range(n)]


def _fly_star(args):
    rocket, params = args
    return fly(rocket, params)


def monte_carlo(rocket, samples, workers=1):
    """Fly every sample; optionally spread the runs over several processes.

    The runs are independent, so they can run in parallel on separate CPU
    cores (``workers > 1``). Results come back in the same order as the
    samples either way.

    Parameters
    ----------
    rocket : Rocket
    samples : list of dict
        From ``sample_inputs``.
    workers : int
        Number of processes. 1 runs everything here, one after another.

    Returns
    -------
    list of dict
        One ``fly`` summary per sample.
    """
    jobs = [(rocket, s) for s in samples]
    if workers == 1:
        return [_fly_star(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_fly_star, jobs, chunksize=max(1, len(jobs) // (4 * workers))))


def available_workers():
    """Number of CPU cores Python can use (at least 1)."""
    return max(1, os.cpu_count() or 1)


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------

@contextmanager
def numpy_cross_baseline():
    """Temporarily put NumPy's general ``np.cross`` back in place of ``cross3``.

    The original code used ``np.cross``; profiling showed it took about 45 %
    of the run time, so it was replaced by the 3-vector ``cross3``. This
    context manager recreates the original code path, so the before/after
    timing can be repeated on any machine:

        with numpy_cross_baseline():
            ...  # runs exactly as the original code did
    """
    import rocket6dof.aerodynamics as aero
    import rocket6dof.quaternions as quats
    import rocket6dof.rigid_body as rigid

    saved = (quats.cross3, rigid.cross3, aero.cross3)
    quats.cross3 = rigid.cross3 = aero.cross3 = np.cross
    try:
        yield
    finally:
        quats.cross3, rigid.cross3, aero.cross3 = saved


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


# Hand count of floating-point operations in one right-hand-side call, by
# function (additions, multiplications, divisions; sqrt, log, exp and powers
# counted as one each). Approximate, to compare with the measured run time.
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
