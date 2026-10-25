"""Regenerate every figure and results table in the report.

Run from the project folder, with the virtual environment active:

    python scripts/make_figures.py            # everything (about 3 to 6 minutes)
    python scripts/make_figures.py --quick    # fewer Monte Carlo runs, for a fast check

Outputs
-------
reports/figures/*.png   the figures
reports/results.json    every number quoted in the report, machine-readable
reports/results.md      the same numbers as Markdown tables
"""

import argparse
import cProfile
import io
import json
import pstats
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                      # draw to files, no window
import matplotlib.pyplot as plt
import numpy as np

from rocket6dof.aerodynamics import ConstantDrag, DragTable, drag_breakdown, normal_force_slope_and_cp, stability_margin
from rocket6dof.analysis import (FLOP_ESTIMATE, apogee_fixed_step, apogee_rk45, available_workers,
                                 fly, monte_carlo, numpy_cross_baseline, observed_order,
                                 sample_inputs, time_rhs, variant)
from rocket6dof.atmosphere import isa
from rocket6dof.benchmark import (compare, infer_drag_coefficient, load_openrocket_csv,
                                  openrocket_gravity)
from rocket6dof.config import load_rocket
from rocket6dof.dynamics import OMEGA, QUAT
from rocket6dof.euler_angles import euler_rates, euler_to_rotation, ENU_TO_NED, quat_to_euler
from rocket6dof.integrators import rk4_step
from rocket6dof.quaternions import quat_derivative, quat_normalise, quat_to_rotation_matrix
from rocket6dof.simulation import air_speed_and_mach, launch_attitude, simulate, summarise
from rocket6dof.wind import Wind

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "reports" / "figures"
ROCKET_FILE = ROOT / "data" / "rocket" / "l1_rocket.toml"
BENCHMARK_FILE = ROOT / "data" / "benchmark" / "L1_benchmark.csv"

LATITUDE = 28.61          # launch latitude set in the OpenRocket simulation [deg]
G_OR = np.array([0.0, 0.0, -openrocket_gravity(LATITUDE, 0.0)])   # OpenRocket's gravity
H_SELECTED = 0.02         # ascent step chosen from the convergence study [s]
H_DESCENT = 0.1           # descent step [s]
VERTICAL = launch_attitude(90.0)

# Colours: a colour-blind-checked categorical palette, used in a fixed order.
SIM, OPENROCKET, C3, C4, C5, C6, C7, C8 = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                           "#e87ba4", "#008300", "#6250d6", "#e34948")
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9, "axes.titlesize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e4e3df", "grid.linewidth": 0.6, "lines.linewidth": 1.5,
    "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "legend.frameon": False,
})

results = {}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / name)
    plt.close(fig)
    print(f"  wrote reports/figures/{name}")


def pct(a, b):
    return 100.0 * (a - b) / b


# ---------------------------------------------------------------------------
# 1. Validation against OpenRocket
# ---------------------------------------------------------------------------

def openrocket_drag_table(data):
    """OpenRocket's C_D(M), inferred from its coast data and smoothed.

    A least-squares fit C_D = c0 + c2 M^2 + c4 M^4 (drag terms grow with M^2)
    is tabulated from M = 0 to 0.7, just past the flight's peak (0.665).
    """
    mach, cd = infer_drag_coefficient(data, np.pi * 0.03**2, LATITUDE, 1.236, 11.5)
    A = np.column_stack([np.ones_like(mach), mach**2, mach**4])
    coef, *_ = np.linalg.lstsq(A, cd, rcond=None)
    rms = float(np.sqrt(np.mean((A @ coef - cd) ** 2)))
    grid = np.linspace(0.0, 0.7, 29)
    table = DragTable(grid, coef[0] + coef[1] * grid**2 + coef[2] * grid**4)
    return table, mach, cd, coef, rms


def validation(rocket, data, events):
    print("1. Validation against OpenRocket")
    table, mach_or, cd_or, coef, rms = openrocket_drag_table(data)
    i_apo = np.argmax(data["altitude"])
    ref = {"apogee": float(data["altitude"][i_apo]), "t_apogee": float(data["t"][i_apo]),
           "max_speed": float(np.nanmax(data["speed"])), "rail_exit_speed": None,
           "burnout_t": events["BURNOUT"], "landing_t": events["GROUND_HIT"],
           "landing_speed": float(abs(data["vz"][-1]))}
    i_rod = np.argmin(abs(data["t"] - events["LAUNCHROD"]))
    ref["rail_exit_speed"] = float(data["speed"][i_rod])
    ref["rail_exit_height"] = float(data["altitude"][i_rod])

    runs = {
        "Build-up C_D (selected)": variant(rocket),
        "OpenRocket C_D table": variant(rocket, drag_model=table),
    }
    flights, rows = {}, {}
    for label, r in runs.items():
        res = simulate(r, VERTICAL, rail_length=1.0, h=H_SELECTED, h_descent=H_DESCENT, gravity=G_OR)
        s = summarise(res)
        ascent = res["t"] <= s["t_apogee"]
        or_ascent = data["t"] <= ref["t_apogee"]
        s["rmse_ascent"] = compare(res["t"][ascent], res["x"][ascent, 2],
                                   data["t"][or_ascent], data["altitude"][or_ascent])
        s["rmse_flight"] = compare(res["t"], res["x"][:, 2], data["t"], data["altitude"])
        flights[label], rows[label] = res, s

    # Same flight with standard gravity instead of OpenRocket's latitude-dependent value.
    std = summarise(simulate(variant(rocket), VERTICAL, h=H_SELECTED, h_descent=H_DESCENT,
                             stop_at_apogee=True))
    # Rail exit measured at OpenRocket's own clearance height (its rod event).
    longer = simulate(variant(rocket), VERTICAL, rail_length=ref["rail_exit_height"],
                      h=H_SELECTED, gravity=G_OR, stop_at_apogee=True)

    sel = rows["Build-up C_D (selected)"]
    results["validation"] = {
        "openrocket": ref,
        "runs": rows,
        "apogee_error_pct": pct(sel["apogee"], ref["apogee"]),
        "apogee_error_pct_or_drag": pct(rows["OpenRocket C_D table"]["apogee"], ref["apogee"]),
        "apogee_standard_gravity": std["apogee"],
        "rail_exit_speed_at_or_height": float(longer["events"]["rail_exit"]["speed"]),
        "or_cd_fit": {"c0": coef[0], "c2": coef[1], "c4": coef[2], "rms": rms},
        "acceptance_tolerance_pct": 5.0,
    }

    # Figure: altitude and vertical velocity, ascent; plus altitude difference.
    res = flights["Build-up C_D (selected)"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    up = data["t"] <= 20.0
    axes[0].plot(data["t"][up], data["altitude"][up], color=OPENROCKET, label="OpenRocket")
    axes[0].plot(res["t"][res["t"] <= 20], res["x"][res["t"] <= 20, 2], "--", color=SIM,
                 label="This simulator")
    axes[0].set(xlabel="Time [s]", ylabel="Altitude [m]", title="Altitude, ascent")
    axes[0].legend()
    axes[1].plot(data["t"][up], data["vz"][up], color=OPENROCKET, label="OpenRocket")
    axes[1].plot(res["t"][res["t"] <= 20], res["x"][res["t"] <= 20, 5], "--", color=SIM,
                 label="This simulator")
    axes[1].set(xlabel="Time [s]", ylabel="Vertical velocity [m/s]", title="Vertical velocity")
    for label, colour in [("Build-up C_D (selected)", SIM), ("OpenRocket C_D table", C3)]:
        r = flights[label]
        t_common = data["t"][(data["t"] <= r["t"][-1]) & (data["t"] <= 20.0)]
        diff = np.interp(t_common, r["t"], r["x"][:, 2]) - np.interp(t_common, data["t"], data["altitude"])
        axes[2].plot(t_common, diff, color=colour, label=label)
    axes[2].axhline(0.0, color="#52514e", linewidth=0.8)
    axes[2].set(xlabel="Time [s]", ylabel="Simulator - OpenRocket [m]", title="Altitude difference")
    axes[2].legend()
    save(fig, "validation_flight.png")

    # Figure: CP, CG and stability margin over the burn and coast.
    t = res["t"][res["t"] <= ref["t_apogee"]]
    _, mach = air_speed_and_mach(res)
    mach = mach[: len(t)]
    cp = np.array([normal_force_slope_and_cp(rocket.geometry, m)[1] for m in mach])
    cg = np.array([rocket.cg(ti) for ti in t])
    ok = np.isfinite(data["cp"]) & (data["t"] <= ref["t_apogee"])
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    axes[0].plot(data["t"][ok], 100 * data["cp"][ok], color=OPENROCKET, label="CP, OpenRocket")
    axes[0].plot(t[1:], 100 * cp[1:], "--", color=SIM, label="CP, this simulator")
    axes[0].plot(data["t"][ok], 100 * data["cg"][ok], color=C4, label="CG, OpenRocket")
    axes[0].plot(t, 100 * cg, "--", color=C7, label="CG, this simulator")
    axes[0].set(xlabel="Time [s]", ylabel="Station from nose tip [cm]", title="Centre of pressure and gravity")
    axes[0].legend(fontsize=7)
    axes[1].plot(data["t"][ok], data["stability"][ok], color=OPENROCKET, label="OpenRocket")
    axes[1].plot(t[1:], stability_margin(cp[1:], cg[1:], 0.06), "--", color=SIM, label="This simulator")
    axes[1].set(xlabel="Time [s]", ylabel="Stability margin [calibres]", title="Static stability margin")
    axes[1].legend()
    save(fig, "cp_cg_stability.png")
    # Compare CPs between rail exit and 12 s. Near apogee the speed falls to
    # zero, the angle of attack grows, and OpenRocket adds body lift at angle
    # of attack (moving its CP forward), which this linear model leaves out.
    window = (t > res["events"]["rail_exit"]["t"]) & (t < 12.0)
    cp_or = np.interp(t[window], data["t"][ok], data["cp"][ok])
    results["validation"]["cp_max_difference_mm"] = float(1000 * np.max(abs(cp[window] - cp_or)))
    results["validation"]["stability_at_rail_exit"] = float(stability_margin(
        np.interp(res["events"]["rail_exit"]["t"], t[1:], cp[1:]),
        rocket.cg(res["events"]["rail_exit"]["t"]), 0.06))

    # Figure: drag coefficient against Mach number.
    grid = np.linspace(0.02, 0.7, 120)
    parts = [drag_breakdown(rocket.geometry, m, 1e7) for m in grid]
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    base = np.array([p["base"] for p in parts])
    fins = base + np.array([p["fin_pressure"] for p in parts])
    total = np.array([p["total"] for p in parts])
    ax.fill_between(grid, 0, base, color=C4, alpha=0.35, linewidth=0, label="Base drag")
    ax.fill_between(grid, base, fins, color=C3, alpha=0.35, linewidth=0, label="Fin edge pressure drag")
    ax.fill_between(grid, fins, total, color=SIM, alpha=0.35, linewidth=0, label="Skin friction")
    ax.plot(grid, total, color=SIM, label="Build-up total (this simulator)")
    ax.plot(mach_or, cd_or, ".", color=OPENROCKET, markersize=3, label="OpenRocket (inferred from its output)")
    ax.set(xlabel="Mach number", ylabel="Drag coefficient C_D", ylim=(0, 0.6), title="Drag coefficient")
    ax.legend(fontsize=7, loc="lower right")
    save(fig, "drag_coefficient.png")
    cd_build = np.array([drag_breakdown(rocket.geometry, m, 1e7)["total"] for m in mach_or])
    results["validation"]["cd_difference_pct_max"] = float(np.max(abs(pct(cd_build, cd_or))))
    results["validation"]["cd_difference_pct_mean"] = float(np.mean(pct(cd_build, cd_or)))
    return table


# ---------------------------------------------------------------------------
# 2. Convergence and step size (numerical verification and stability)
# ---------------------------------------------------------------------------

def convergence(rocket):
    print("2. Convergence and step size")
    ref, _, _ = apogee_fixed_step(rocket, 0.0005)
    hs = np.array([0.32, 0.16, 0.08, 0.04, 0.02, 0.01, 0.005])
    out = {"reference_apogee": ref, "h": hs.tolist()}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for align, colour, label in [("none", C8, "RK4, no alignment"),
                                 ("burnout", C4, "RK4, steps end at liftoff + burnout"),
                                 ("thrust_points", SIM, "RK4, steps end on every thrust point")]:
        err = np.array([abs(apogee_fixed_step(rocket, h, align=align)[0] - ref) for h in hs])
        out[align] = err.tolist()
        axes[0].loglog(hs, np.maximum(err, 1e-9), "o-", color=colour, markersize=4, label=label)
    hs_e = np.array([0.02, 0.01, 0.005, 0.0025])
    err_e = np.array([abs(apogee_fixed_step(rocket, h, method="euler")[0] - ref) for h in hs_e])
    out["euler_h"], out["euler"] = hs_e.tolist(), err_e.tolist()
    out["euler_order"] = observed_order(hs_e, err_e).tolist()
    axes[0].loglog(hs_e, err_e, "s-", color=C7, markersize=4, label="Explicit Euler")
    # Orders from successive halvings (steps above ~0.3 s are cut short by the
    # thrust points, so the coarse end is not a clean test).
    out["rk4_order"] = observed_order(hs, np.array(out["thrust_points"])).tolist()
    tp = np.array(out["thrust_points"])
    axes[0].loglog(hs[2:5], tp[3] * (hs[2:5] / hs[3]) ** 4, ":", color="#52514e", label="slope 4 (guide)")
    axes[0].set(xlabel="Step size h [s]", ylabel="|Apogee error| [m]", ylim=(1e-7, 1e3),
                title="Vertical flight: apogee error vs step")
    axes[0].legend(fontsize=6, loc="upper left")

    # Wind case: the pitch oscillation limits the step (RK4 stability).
    wind = Wind(5.0, 270.0, exponent=1.0 / 7.0)
    def wind_apogee(h):
        res = simulate(rocket, VERTICAL, h=h, wind=wind, stop_at_apogee=True)
        return res["events"]["apogee"]["position"]
    ref_w = wind_apogee(0.001)
    hs_w = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005]
    err_w, unstable = [], []
    for h in hs_w:
        try:
            err_w.append(float(np.linalg.norm(wind_apogee(h) - ref_w)))
        except ValueError:                    # left the model's Mach range: the run blew up
            err_w.append(np.nan)
            unstable.append(h)
    # Highest pitch natural frequency in the flight (at peak dynamic pressure).
    rho, a = isa(150.0)[2], isa(150.0)[3]
    V = 225.0
    cn, xcp = normal_force_slope_and_cp(rocket.geometry, V / a)
    k = 0.5 * rho * V**2 * rocket.geometry.reference_area * cn * (xcp - rocket.cg(1.0))
    omega_n = float(np.sqrt(k / rocket.inertia_at(1.0)[1, 1]))
    out.update({"wind_h": hs_w, "wind_error": err_w, "wind_unstable_h": unstable,
                "pitch_omega_n": omega_n, "rk4_stability_h": 2.83 / omega_n})
    good = np.isfinite(err_w)
    axes[1].loglog(np.array(hs_w)[good], np.maximum(np.array(err_w)[good], 1e-9), "o-", color=SIM,
                   markersize=4, label="RK4, 5 m/s crosswind")
    for h in unstable:
        axes[1].axvline(h, color=C8, linestyle="--", linewidth=1.0)
    axes[1].axvline(2.83 / omega_n, color="#52514e", linestyle=":", linewidth=1.0,
                    label=f"RK4 stability limit 2.83/omega_n = {2.83 / omega_n:.3f} s")
    axes[1].set(xlabel="Step size h [s]", ylabel="|Apogee position error| [m]",
                title="Crosswind flight: error vs step (dashed: run blew up)")
    axes[1].legend(fontsize=7)
    save(fig, "convergence.png")
    results["convergence"] = out


# ---------------------------------------------------------------------------
# 3. Integrator comparison (accuracy against cost)
# ---------------------------------------------------------------------------

def integrators(rocket):
    print("3. Integrator comparison")
    ref, _, _ = apogee_fixed_step(rocket, 0.0005)
    rows = []
    for h in [0.32, 0.16, 0.08, 0.04, 0.02, 0.01]:
        a, n, t = apogee_fixed_step(rocket, h)
        rows.append(("RK4", h, abs(a - ref), n, t))
    for h in [0.02, 0.01, 0.005, 0.0025, 0.00125]:
        a, n, t = apogee_fixed_step(rocket, h, method="euler")
        rows.append(("Euler", h, abs(a - ref), n, t))
    for rtol in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:
        a, n, t = apogee_rk45(rocket, rtol)
        rows.append(("RK45", rtol, abs(a - ref), n, t))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for method, colour, marker in [("Euler", C7, "s"), ("RK4", SIM, "o"), ("RK45", OPENROCKET, "^")]:
        sub = [r for r in rows if r[0] == method]
        err = np.maximum([r[2] for r in sub], 1e-9)
        axes[0].loglog([r[3] for r in sub], err, marker + "-", color=colour, markersize=4, label=method)
        axes[1].loglog([r[4] for r in sub], err, marker + "-", color=colour, markersize=4, label=method)
    for ax, xlabel in zip(axes, ["Right-hand-side evaluations", "Run time [s]"]):
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.axhline(0.1, color="#52514e", linestyle=":", linewidth=1.0, label="0.1 m tolerance")
        ax.set(xlabel=xlabel, ylabel="|Apogee error| [m]")
        ax.legend(fontsize=7)
    axes[0].set_title("Accuracy against cost")
    axes[1].set_title("Accuracy against run time")
    save(fig, "integrators.png")
    results["integrators"] = [dict(zip(["method", "setting", "error_m", "n_rhs", "runtime_s"], r))
                              for r in rows]


# ---------------------------------------------------------------------------
# 4. Drag model comparison
# ---------------------------------------------------------------------------

def drag_models(rocket, or_table, data):
    print("4. Drag model comparison")
    cd_low = drag_breakdown(rocket.geometry, 0.05, 1e7)["total"]
    models = {f"Constant C_D = {cd_low:.3f} (low-speed value)": ConstantDrag(cd_low),
              "Build-up C_D(M, Re) (selected)": rocket.drag_model,
              "OpenRocket C_D(M) table": or_table}
    ref_apogee = float(np.max(data["altitude"]))
    out = {}
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for (label, model), colour in zip(models.items(), [C8, SIM, C3]):
        res = simulate(variant(rocket, drag_model=model), VERTICAL, h=H_SELECTED, gravity=G_OR,
                       stop_at_apogee=True)
        s = summarise(res)
        out[label] = {"apogee": s["apogee"], "error_pct": pct(s["apogee"], ref_apogee),
                      "max_speed": s["max_speed"], "t_apogee": s["t_apogee"]}
        t = data["t"][data["t"] <= s["t_apogee"]]
        diff = np.interp(t, res["t"], res["x"][:, 2]) - np.interp(t, data["t"], data["altitude"])
        ax.plot(t, diff, color=colour, label=label)
    ax.axhline(0.0, color="#52514e", linewidth=0.8)
    ax.set(xlabel="Time [s]", ylabel="Altitude - OpenRocket [m]", title="Effect of the drag model")
    ax.legend(fontsize=7)
    save(fig, "drag_models.png")
    results["drag_models"] = out


# ---------------------------------------------------------------------------
# 5. Attitude representation: Euler angles against quaternions
# ---------------------------------------------------------------------------

def attitude(rocket):
    """Replay realistic body rates through Euler angles and through a quaternion.

    The body rates come from a 6-DOF flight in a crosswind, plus a steady
    2 rad/s roll (real rockets roll slowly from small fin misalignments).
    Both attitude forms are integrated with RK4 at the flight step (0.02 s)
    and compared with a quaternion integrated at 0.0005 s (the reference).
    """
    print("5. Euler angles against quaternions")
    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    wind = Wind(5.0, 270.0, exponent=1.0 / 7.0)
    roll = np.array([2.0, 0.0, 0.0])
    for elevation, colour in [(80.0, C3), (88.0, C4), (89.9, C8), (90.0, C7)]:
        res = simulate(rocket, launch_attitude(elevation, 90.0), h=H_SELECTED, wind=wind,
                       stop_at_apogee=True)
        t_f, w_f = res["t"], res["x"][:, OMEGA]

        def omega_at(tt):
            return np.array([np.interp(tt, t_f, w_f[:, k]) for k in range(3)]) + roll

        def rhs_q(tt, qq):
            return quat_derivative(qq, omega_at(tt))

        def rhs_e(tt, ee):
            return euler_rates(ee, omega_at(tt))

        q0 = res["x"][0, QUAT]
        t_end = min(6.0, t_f[-1])
        h_ref, h = 0.0005, H_SELECTED
        q_ref = q0.copy()
        R_ref = {}
        for i in range(int(round(t_end / h_ref))):
            q_ref = quat_normalise(rk4_step(rhs_q, i * h_ref, q_ref, h_ref))
            if (i + 1) % int(round(h / h_ref)) == 0:
                R_ref[(i + 1) // int(round(h / h_ref))] = ENU_TO_NED @ quat_to_rotation_matrix(q_ref)

        def angle_between(R1, R2):
            return np.degrees(np.arccos(np.clip((np.trace(R1.T @ R2) - 1.0) / 2.0, -1.0, 1.0)))

        q, e, ok = q0.copy(), quat_to_euler(q0), True
        times, err_q, err_e = [], [], []
        for j in range(1, int(round(t_end / h)) + 1):
            tj = (j - 1) * h
            q = quat_normalise(rk4_step(rhs_q, tj, q, h))
            if ok:
                with np.errstate(all="ignore"):
                    e = rk4_step(rhs_e, tj, e, h)
                ok = bool(np.all(np.isfinite(e)))
            times.append(j * h)
            err_q.append(angle_between(ENU_TO_NED @ quat_to_rotation_matrix(q), R_ref[j]))
            err_e.append(angle_between(euler_to_rotation(e), R_ref[j]) if ok else np.nan)
        label = f"rail {elevation:g} deg"
        axes[0].semilogy(times, np.maximum(err_e, 1e-10), color=colour, label=f"Euler angles, {label}")
        axes[0].semilogy(times, np.maximum(err_q, 1e-10), ":", color=colour)
        finite = np.isfinite(err_e)
        out[label] = {"euler_max_error_deg": float(np.max(np.array(err_e)[finite])) if finite.any() else None,
                      "euler_failed": not ok,
                      "quaternion_max_error_deg": float(np.max(err_q))}
    axes[0].set(xlabel="Time [s]", ylabel="Attitude error [deg]", ylim=(1e-4, 1.0),
                title="Same body rates, h = 0.02 s (dotted: quaternion)")
    axes[0].legend(fontsize=6, loc="center right")
    theta = np.linspace(0.0, 89.99, 400)
    axes[1].semilogy(theta, 1.0 / np.cos(np.radians(theta)), color=SIM)
    axes[1].set(xlabel="Pitch angle theta [deg]", ylabel="1 / cos(theta)",
                title="Euler-angle rate amplification")
    save(fig, "attitude.png")
    results["attitude"] = out


# ---------------------------------------------------------------------------
# 6. Wind: trajectories and weathercocking
# ---------------------------------------------------------------------------

def wind_flights(rocket):
    print("6. Wind flights")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    out = {}
    for speed, colour in [(0.0, SIM), (3.0, C3), (6.0, OPENROCKET)]:
        wind = Wind(speed, 270.0, exponent=1.0 / 7.0) if speed else None
        res = simulate(rocket, VERTICAL, h=H_SELECTED, h_descent=H_DESCENT, wind=wind)
        t, x = res["t"], res["x"]
        axes[0].plot(x[:, 0], x[:, 2], color=colour, label=f"{speed:g} m/s from the west")
        aoa = []
        for row in x:
            v = quat_to_rotation_matrix(row[QUAT]).T @ (row[3:6] - (wind(row[2]) if wind else 0.0))
            aoa.append(np.degrees(np.arctan2(np.hypot(v[1], v[2]), v[0])))
        tr = (t >= res["events"]["rail_exit"]["t"]) & (t <= 4.0)
        axes[1].plot(t[tr], np.array(aoa)[tr], color=colour, label=f"{speed:g} m/s")
        s = summarise(res)
        out[f"{speed:g} m/s"] = {"apogee": s["apogee"],
                                 "apogee_x": float(res["events"]["apogee"]["position"][0]),
                                 "landing_x": float(res["events"]["landing"]["position"][0]),
                                 "max_aoa_after_rail_deg": float(np.max(np.array(aoa)[tr]))}
    axes[0].set(xlabel="Distance east [m]", ylabel="Altitude [m]", title="Flight paths (rocket turns into the wind)")
    axes[0].legend(fontsize=7)
    axes[1].set(xlabel="Time [s]", ylabel="Angle of attack [deg]", title="Weathercocking after rail exit")
    axes[1].legend(fontsize=7)
    save(fig, "wind_flights.png")
    results["wind"] = out


# ---------------------------------------------------------------------------
# 7. Sensitivity (one factor at a time)
# ---------------------------------------------------------------------------

def sensitivity(rocket):
    print("7. Sensitivity")
    nominal = fly(rocket, {})
    cases = [("Drag coefficient", "cd_factor", 0.9, 1.1, "-10 % / +10 %"),
             ("Dry mass", "mass_factor", 0.95, 1.05, "-5 % / +5 %"),
             ("Thrust (impulse)", "thrust_factor", 0.97, 1.03, "-3 % / +3 %"),
             ("Wind speed at 10 m", "wind_speed", 0.0, 6.0, "0 / 6 m/s"),
             ("Rail elevation", "elevation", 85.0, 90.0, "85 / 90 deg"),
             ("Launch site altitude", "launch_altitude", 0.0, 1000.0, "0 / 1000 m")]
    rows = []
    for name, key, low, high, text in cases:
        lo, hi = fly(rocket, {key: low}), fly(rocket, {key: high})
        rows.append({"input": name, "range": text,
                     "apogee_low": lo["apogee"] - nominal["apogee"],
                     "apogee_high": hi["apogee"] - nominal["apogee"],
                     "landing_low": lo["landing_distance"], "landing_high": hi["landing_distance"]})
    g_std = summarise(simulate(rocket, VERTICAL, h=H_SELECTED, stop_at_apogee=True))["apogee"]
    g_or = summarise(simulate(rocket, VERTICAL, h=H_SELECTED, gravity=G_OR, stop_at_apogee=True))["apogee"]
    rows.append({"input": "Gravity model", "range": "9.80665 / 9.7922 m/s^2",
                 "apogee_low": 0.0, "apogee_high": g_or - g_std,
                 "landing_low": 0.0, "landing_high": 0.0})
    rows.sort(key=lambda r: max(abs(r["apogee_low"]), abs(r["apogee_high"])))
    fig, ax = plt.subplots(figsize=(6, 3.4))
    y = np.arange(len(rows))
    ax.barh(y, [r["apogee_low"] for r in rows], color=C3, height=0.6, label="Low value")
    ax.barh(y, [r["apogee_high"] for r in rows], color=SIM, height=0.6, label="High value")
    ax.set_yticks(y, [f"{r['input']}\n({r['range']})" for r in rows], fontsize=7)
    ax.axvline(0.0, color="#52514e", linewidth=0.8)
    ax.set(xlabel="Change in apogee [m]", title=f"Sensitivity of apogee (nominal {nominal['apogee']:.0f} m)")
    ax.legend(fontsize=7)
    save(fig, "sensitivity.png")
    results["sensitivity"] = {"nominal_apogee": nominal["apogee"], "rows": rows[::-1]}


# ---------------------------------------------------------------------------
# 8. Monte Carlo dispersion
# ---------------------------------------------------------------------------

def monte_carlo_study(rocket, n, workers):
    print(f"8. Monte Carlo ({n} flights, {workers} processes)")
    samples = sample_inputs(n)
    t0 = time.perf_counter()
    runs = monte_carlo(rocket, samples, workers=workers)
    elapsed = time.perf_counter() - t0
    apogee = np.array([r["apogee"] for r in runs])
    lx = np.array([r["landing_x"] for r in runs])
    ly = np.array([r["landing_y"] for r in runs])
    dist = np.hypot(lx, ly)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].hist(apogee, bins=25, color=SIM, edgecolor="white", linewidth=0.5)
    for q in np.percentile(apogee, [5, 95]):
        axes[0].axvline(q, color="#52514e", linestyle=":", linewidth=1.0)
    axes[0].set(xlabel="Apogee [m]", ylabel="Flights", title="Apogee (dotted: 5th and 95th percentiles)")
    axes[1].plot(lx, ly, "o", color=SIM, markersize=3, alpha=0.7, label="Landing points")
    axes[1].plot(0, 0, "^", color=OPENROCKET, markersize=8, label="Launch pad")
    theta = np.linspace(0, 2 * np.pi, 200)
    r95 = np.percentile(dist, 95)
    axes[1].plot(r95 * np.cos(theta), r95 * np.sin(theta), "--", color="#52514e", linewidth=1.0,
                 label=f"95 % of landings within {r95:.0f} m")
    axes[1].set(xlabel="East [m]", ylabel="North [m]", title="Landing dispersion")
    axes[1].set_aspect("equal")
    axes[1].legend(fontsize=7, loc="upper right")
    save(fig, "monte_carlo.png")
    # Which input explains the spread? Correlation of each input with the outputs.
    keys = ["cd_factor", "mass_factor", "thrust_factor", "wind_speed", "elevation"]
    corr = {k: {"apogee": float(np.corrcoef([s[k] for s in samples], apogee)[0, 1]),
                "landing_distance": float(np.corrcoef([s[k] for s in samples], dist)[0, 1])}
            for k in keys}
    results["monte_carlo"] = {
        "n": n, "workers": workers, "seconds": elapsed,
        "apogee_mean": float(apogee.mean()), "apogee_std": float(apogee.std(ddof=1)),
        "apogee_p5": float(np.percentile(apogee, 5)), "apogee_p95": float(np.percentile(apogee, 95)),
        "landing_distance_median": float(np.median(dist)), "landing_distance_p95": float(r95),
        "correlations": corr,
    }


# ---------------------------------------------------------------------------
# 9. Performance: profiling, FLOPs, before/after
# ---------------------------------------------------------------------------

def profile_top(rocket, n=8):
    pr = cProfile.Profile()
    pr.enable()
    simulate(rocket, VERTICAL, h=0.01, stop_at_apogee=True)
    pr.disable()
    stats = pstats.Stats(pr)
    total = stats.total_tt
    rows = sorted(((v[2], f"{Path(k[0]).name}:{k[2]}") for k, v in stats.stats.items()), reverse=True)[:n]
    # Total time inside numpy.cross, including the helper functions it calls.
    cross = sum(v[3] for k, v in stats.stats.items() if k[2] == "cross" and "numeric" in k[0])
    top = [{"function": name, "own_time_s": t, "share_pct": 100 * t / total} for t, name in rows]
    return top, total, 100 * cross / total


def performance(rocket, workers):
    print("9. Performance")
    out = {}
    with numpy_cross_baseline():
        out["profile_before"], out["profile_total_before"], out["np_cross_share_pct"] = profile_top(rocket)
        out["rhs_us_before"] = 1e6 * time_rhs(rocket)
        t0 = time.perf_counter()
        base = simulate(rocket, VERTICAL, h=0.01)            # original settings: h = 0.01 throughout
        out["flight_s_before"] = time.perf_counter() - t0
        out["flight_rhs_before"] = base["n_rhs"]
    out["profile_after"], out["profile_total_after"], _ = profile_top(rocket)
    out["rhs_us_after"] = 1e6 * time_rhs(rocket)
    t0 = time.perf_counter()
    sel = simulate(rocket, VERTICAL, h=H_SELECTED, h_descent=H_DESCENT)
    out["flight_s_after"] = time.perf_counter() - t0
    out["flight_rhs_after"] = sel["n_rhs"]
    out["landing_difference_s"] = abs(summarise(sel)["landing_t"] - summarise(base)["landing_t"])
    out["apogee_difference_m"] = abs(summarise(sel)["apogee"] - summarise(base)["apogee"])
    flops = sum(FLOP_ESTIMATE.values())
    out["flops_per_rhs"] = flops
    out["flop_breakdown"] = FLOP_ESTIMATE
    out["mflops_achieved"] = flops / out["rhs_us_after"]          # FLOP per microsecond = MFLOP/s
    out["state_bytes"] = 13 * 8
    out["history_bytes_full_flight"] = int(sel["x"].nbytes + sel["t"].nbytes)
    # Batch: the same 2*workers flights, one process against all cores.
    batch = sample_inputs(max(8, 2 * workers), seed=7)
    t0 = time.perf_counter()
    monte_carlo(rocket, batch, workers=1)
    out["batch_serial_s"] = time.perf_counter() - t0
    if workers > 1:
        t0 = time.perf_counter()
        monte_carlo(rocket, batch, workers=workers)
        out["batch_parallel_s"] = time.perf_counter() - t0
    out["batch_size"], out["workers"] = len(batch), workers

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    names = [r["function"] for r in out["profile_before"]][::-1]
    shares = [r["share_pct"] for r in out["profile_before"]][::-1]
    ax.barh(np.arange(len(names)), shares, color=C8, height=0.6)
    ax.set_yticks(np.arange(len(names)), names, fontsize=7)
    ax.set(xlabel="Share of run time (own time) [%]",
           title=f"Original code: {out['profile_total_before']:.2f} s per ascent; "
                 f"numpy.cross and its helpers {out['np_cross_share_pct']:.0f} %")
    save(fig, "profile.png")
    results["performance"] = out


# ---------------------------------------------------------------------------
# Results as Markdown tables (for the report)
# ---------------------------------------------------------------------------

def write_markdown():
    v = results["validation"]
    ref, sel = v["openrocket"], v["runs"]["Build-up C_D (selected)"]
    tab = v["runs"]["OpenRocket C_D table"]
    lines = ["# Results (generated by scripts/make_figures.py)", "",
             "## Validation against OpenRocket", "",
             "| Quantity | OpenRocket | This simulator | Difference | With OpenRocket's C_D |",
             "| --- | --- | --- | --- | --- |"]
    for key, name, unit in [("apogee", "Apogee", "m"), ("t_apogee", "Time to apogee", "s"),
                            ("max_speed", "Maximum speed", "m/s"), ("burnout_t", "Burnout", "s"),
                            ("landing_t", "Landing time", "s"), ("landing_speed", "Landing speed", "m/s"),
                            ("rail_exit_speed", "Rail exit speed", "m/s")]:
        lines.append(f"| {name} [{unit}] | {ref[key]:.3f} | {sel[key]:.3f} | "
                     f"{pct(sel[key], ref[key]):+.2f} % | {tab[key]:.3f} ({pct(tab[key], ref[key]):+.2f} %) |")
    lines += ["", f"- Altitude RMSE, ascent: {sel['rmse_ascent']:.2f} m (OpenRocket C_D: {tab['rmse_ascent']:.2f} m)",
              f"- Altitude RMSE, whole flight: {sel['rmse_flight']:.2f} m",
              f"- Rail exit speed at OpenRocket's clearance height ({ref['rail_exit_height']:.3f} m): "
              f"{v['rail_exit_speed_at_or_height']:.2f} m/s",
              f"- Apogee with standard gravity 9.80665 m/s^2: {v['apogee_standard_gravity']:.2f} m",
              f"- Largest CP difference from OpenRocket: {v['cp_max_difference_mm']:.2f} mm",
              f"- Stability margin at rail exit: {v['stability_at_rail_exit']:.2f} calibres",
              f"- Build-up C_D vs OpenRocket: mean {v['cd_difference_pct_mean']:+.2f} %, "
              f"worst {v['cd_difference_pct_max']:.2f} %", ""]
    c = results["convergence"]
    lines += ["## Convergence", "", "| h [s] | no alignment [m] | liftoff+burnout [m] | all thrust points [m] |",
              "| --- | --- | --- | --- |"]
    for i, h in enumerate(c["h"]):
        lines.append(f"| {h} | {c['none'][i]:.2e} | {c['burnout'][i]:.2e} | {c['thrust_points'][i]:.2e} |")
    lines += ["", f"- RK4 observed order between successive h (aligned): {np.round(c['rk4_order'], 2).tolist()}",
              f"- Euler observed order: {np.round(c['euler_order'], 2).tolist()}",
              f"- Pitch natural frequency at max dynamic pressure: {c['pitch_omega_n']:.1f} rad/s; "
              f"RK4 stability limit 2.83/omega_n = {c['rk4_stability_h']:.3f} s",
              f"- Crosswind runs that blew up: h = {c['wind_unstable_h']}", ""]
    lines += ["## Integrators", "", "| Method | h or rtol | Error [m] | RHS calls | Run time [s] |",
              "| --- | --- | --- | --- | --- |"]
    for r in results["integrators"]:
        lines.append(f"| {r['method']} | {r['setting']:g} | {r['error_m']:.2e} | {r['n_rhs']} | {r['runtime_s']:.3f} |")
    lines += ["", "## Drag models", "", "| Model | Apogee [m] | vs OpenRocket |", "| --- | --- | --- |"]
    for k, r in results["drag_models"].items():
        lines.append(f"| {k} | {r['apogee']:.1f} | {r['error_pct']:+.2f} % |")
    lines += ["", "## Attitude representation", "", "| Rail | Euler-angle max error [deg] | Quaternion max error [deg] |",
              "| --- | --- | --- |"]
    for k, r in results["attitude"].items():
        e = "failed (NaN)" if r["euler_failed"] else f"{r['euler_max_error_deg']:.2e}"
        lines.append(f"| {k} | {e} | {r['quaternion_max_error_deg']:.2e} |")
    lines += ["", "## Wind", "", "| Wind | Apogee [m] | Apogee east [m] | Landing east [m] | Max AoA after rail [deg] |",
              "| --- | --- | --- | --- | --- |"]
    for k, r in results["wind"].items():
        lines.append(f"| {k} | {r['apogee']:.1f} | {r['apogee_x']:.1f} | {r['landing_x']:.0f} | {r['max_aoa_after_rail_deg']:.1f} |")
    s = results["sensitivity"]
    lines += ["", f"## Sensitivity (nominal apogee {s['nominal_apogee']:.1f} m)", "",
              "| Input | Range | Apogee change [m] | Landing distance [m] |", "| --- | --- | --- | --- |"]
    for r in s["rows"]:
        lines.append(f"| {r['input']} | {r['range']} | {r['apogee_low']:+.1f} / {r['apogee_high']:+.1f} | "
                     f"{r['landing_low']:.0f} / {r['landing_high']:.0f} |")
    m = results["monte_carlo"]
    lines += ["", f"## Monte Carlo ({m['n']} flights, {m['seconds']:.0f} s on {m['workers']} processes)", "",
              f"- Apogee: mean {m['apogee_mean']:.1f} m, sd {m['apogee_std']:.1f} m, "
              f"5-95 %: {m['apogee_p5']:.1f} to {m['apogee_p95']:.1f} m",
              f"- Landing distance: median {m['landing_distance_median']:.0f} m, 95 % within {m['landing_distance_p95']:.0f} m",
              "", "| Input | Correlation with apogee | with landing distance |", "| --- | --- | --- |"]
    for k, r in m["correlations"].items():
        lines.append(f"| {k} | {r['apogee']:+.2f} | {r['landing_distance']:+.2f} |")
    p = results["performance"]
    lines += ["", "## Performance", "",
              "| Measure | Original | Selected |", "| --- | --- | --- |",
              f"| One right-hand-side call [us] | {p['rhs_us_before']:.1f} | {p['rhs_us_after']:.1f} |",
              f"| Full flight, RHS calls | {p['flight_rhs_before']} | {p['flight_rhs_after']} |",
              f"| Full flight, run time [s] | {p['flight_s_before']:.2f} | {p['flight_s_after']:.2f} |",
              f"| {p['batch_size']} flights, run time [s] | {p['batch_serial_s']:.1f} (1 process) | "
              f"{p.get('batch_parallel_s', float('nan')):.1f} ({p['workers']} processes) |",
              "", f"- Apogee difference original vs selected: {p['apogee_difference_m']:.4f} m; "
              f"landing time difference {p['landing_difference_s']:.4f} s",
              f"- Estimated {p['flops_per_rhs']} FLOP per RHS call -> {p['mflops_achieved']:.1f} MFLOP/s achieved",
              f"- numpy.cross and its helpers took {p['np_cross_share_pct']:.0f} % of the original run time (cProfile, cumulative)",
              f"- State vector {p['state_bytes']} bytes; full-flight history {p['history_bytes_full_flight'] / 1024:.0f} kB",
              "", "Profile of the original code (own time):", "", "| Function | Share |", "| --- | --- |"]
    for r in p["profile_before"]:
        lines.append(f"| {r['function']} | {r['share_pct']:.1f} % |")
    lines += ["", "Profile after the change:", "", "| Function | Share |", "| --- | --- |"]
    for r in p["profile_after"]:
        lines.append(f"| {r['function']} | {r['share_pct']:.1f} % |")
    (ROOT / "reports" / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  wrote reports/results.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="40 Monte Carlo flights instead of 300")
    parser.add_argument("--workers", type=int, default=available_workers(),
                        help="processes for the Monte Carlo runs (default: all cores)")
    args = parser.parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    rocket = load_rocket(ROCKET_FILE)
    data, events = load_openrocket_csv(BENCHMARK_FILE)
    or_table = validation(rocket, data, events)
    convergence(rocket)
    integrators(rocket)
    drag_models(rocket, or_table, data)
    attitude(rocket)
    wind_flights(rocket)
    sensitivity(rocket)
    monte_carlo_study(rocket, 40 if args.quick else 300, args.workers)
    performance(rocket, args.workers)

    (ROOT / "reports" / "results.json").write_text(
        json.dumps(results, indent=2, default=float), encoding="utf-8")
    print("  wrote reports/results.json")
    write_markdown()
    print(f"Done in {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
