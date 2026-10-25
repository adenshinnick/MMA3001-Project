"""Regenerate every figure and results table in the report.

Run from the project folder, with the virtual environment active:

    python scripts/make_figures.py            # about 1 to 2 minutes

Outputs
-------
reports/figures/*.png   the figures
reports/results.md      every number quoted in the report, as Markdown tables
"""

import cProfile
import pstats
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                      # draw to files, no window
import matplotlib.pyplot as plt
import numpy as np

from rocket6dof.aerodynamics import ConstantDrag, normal_force_slope_and_cp, stability_margin
from rocket6dof.analysis import (FLOP_ESTIMATE, apogee_fixed_step, fly,
                                 observed_order, time_cross_products, time_rhs, variant)
from rocket6dof.atmosphere import isa
from rocket6dof.benchmark import (G_OPENROCKET, infer_drag_coefficient, load_openrocket_csv,
                                  rms_difference)
from rocket6dof.config import load_rocket
from rocket6dof.simulation import air_speed_and_mach, launch_attitude, simulate, summarise
from rocket6dof.wind import Wind

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "reports" / "figures"
ROCKET_FILE = ROOT / "data" / "rocket" / "l1_rocket.toml"
BENCHMARK_FILE = ROOT / "data" / "benchmark" / "L1_benchmark.csv"

G_OR = np.array([0.0, 0.0, -G_OPENROCKET])   # gravity used in the OpenRocket run
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

def validation(rocket, data, events):
    print("1. Validation against OpenRocket")
    i_apo = np.argmax(data["altitude"])
    i_rod = np.argmin(abs(data["t"] - events["LAUNCHROD"]))
    ref = {"apogee": float(data["altitude"][i_apo]), "t_apogee": float(data["t"][i_apo]),
           "max_speed": float(np.nanmax(data["speed"])), "burnout_t": events["BURNOUT"],
           "landing_t": events["GROUND_HIT"], "landing_speed": float(abs(data["vz"][-1])),
           "rail_exit_speed": float(data["speed"][i_rod]),
           "rail_exit_height": float(data["altitude"][i_rod])}

    res = simulate(rocket, VERTICAL, rail_length=1.0, h=H_SELECTED, h_descent=H_DESCENT, gravity=G_OR)
    sel = summarise(res)
    ascent, or_ascent = res["t"] <= sel["t_apogee"], data["t"] <= ref["t_apogee"]
    sel["rmse_ascent"] = rms_difference(res["t"][ascent], res["x"][ascent, 2],
                                        data["t"][or_ascent], data["altitude"][or_ascent])
    sel["rmse_flight"] = rms_difference(res["t"], res["x"][:, 2], data["t"], data["altitude"])
    # Rail exit measured at OpenRocket's own rod-clearance height.
    longer = simulate(rocket, VERTICAL, rail_length=ref["rail_exit_height"], h=H_SELECTED,
                      gravity=G_OR, stop_at_apogee=True)
    results["validation"] = {"openrocket": ref, "simulator": sel,
                             "rail_exit_speed_at_or_height": float(longer["events"]["rail_exit"]["speed"])}

    # Figure: altitude and vertical velocity in the ascent, and their difference.
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    up, up_sim = data["t"] <= 20.0, res["t"] <= 20.0
    axes[0].plot(data["t"][up], data["altitude"][up], color=OPENROCKET, label="OpenRocket")
    axes[0].plot(res["t"][up_sim], res["x"][up_sim, 2], "--", color=SIM, label="This simulator")
    axes[0].set(xlabel="Time [s]", ylabel="Altitude [m]", title="Altitude, ascent")
    axes[0].legend()
    axes[1].plot(data["t"][up], data["vz"][up], color=OPENROCKET, label="OpenRocket")
    axes[1].plot(res["t"][up_sim], res["x"][up_sim, 5], "--", color=SIM, label="This simulator")
    axes[1].set(xlabel="Time [s]", ylabel="Vertical velocity [m/s]", title="Vertical velocity")
    t_common = data["t"][up]
    diff = np.interp(t_common, res["t"], res["x"][:, 2]) - np.interp(t_common, data["t"], data["altitude"])
    axes[2].plot(t_common, diff, color=SIM)
    axes[2].axhline(0.0, color="#52514e", linewidth=0.8)
    axes[2].set(xlabel="Time [s]", ylabel="Simulator - OpenRocket [m]", title="Altitude difference")
    save(fig, "validation_flight.png")

    # Figure: CP, CG and stability margin over the burn and coast.
    t = res["t"][res["t"] <= ref["t_apogee"]]
    mach = air_speed_and_mach(res)[1][: len(t)]
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
    # Compare CPs from rail exit to 12 s. Near apogee the speed falls to zero and
    # OpenRocket adds body lift at high angle of attack, which this model leaves out.
    window = (t > res["events"]["rail_exit"]["t"]) & (t < 12.0)
    cp_or = np.interp(t[window], data["t"][ok], data["cp"][ok])
    results["validation"]["cp_max_difference_mm"] = float(1000 * np.max(abs(cp[window] - cp_or)))
    t_exit = res["events"]["rail_exit"]["t"]
    results["validation"]["stability_at_rail_exit"] = float(stability_margin(
        np.interp(t_exit, t[1:], cp[1:]), rocket.cg(t_exit), 0.06))

    # Figure: drag coefficient against Mach number: OpenRocket's (inferred) and the fit.
    mach_or, cd_or = infer_drag_coefficient(data, rocket.geometry.reference_area,
                                            events["BURNOUT"] + 0.001, 11.5)
    grid = np.linspace(0.0, 0.7, 100)
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot(mach_or, cd_or, ".", color=OPENROCKET, markersize=3, label="OpenRocket (inferred from its coast data)")
    ax.plot(grid, [rocket.drag_model(m) for m in grid], color=SIM,
            label=f"Fit C_D = {rocket.drag_model.cd0:.4f} + {rocket.drag_model.k:.4f} M^2")
    ax.axhline(rocket.drag_model.cd0, color=C8, linestyle=":", label="Constant C_D (baseline)")
    ax.set(xlabel="Mach number", ylabel="Drag coefficient C_D", ylim=(0.45, 0.56), title="Drag coefficient")
    ax.legend(fontsize=7, loc="upper left")
    save(fig, "drag_coefficient.png")
    fit_error = cd_or - np.array([rocket.drag_model(m) for m in mach_or])
    results["validation"]["cd_fit_rms"] = float(np.sqrt(np.mean(fit_error**2)))


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
    # Orders from successive halvings (steps above ~0.3 s are cut short by the
    # thrust points, so the coarse end is not a clean test).
    out["rk4_order"] = observed_order(hs, np.array(out["thrust_points"])).tolist()
    tp = np.array(out["thrust_points"])
    axes[0].loglog(hs[2:5], tp[3] * (hs[2:5] / hs[3]) ** 4, ":", color="#52514e", label="slope 4 (guide)")
    axes[0].set(xlabel="Step size h [s]", ylabel="|Apogee error| [m]", ylim=(1e-7, 1e3),
                title="Vertical flight: apogee error vs step")
    axes[0].legend(fontsize=6, loc="upper left")

    # Wind case: the pitch oscillation limits the step (RK4 stability).
    wind = Wind(5.0, 270.0)
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
    omega_n = float(np.sqrt(k / rocket.inertia[1, 1]))
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
# 3. Drag model comparison
# ---------------------------------------------------------------------------

def drag_models(rocket, data):
    print("3. Drag model comparison")
    ref_apogee = float(np.max(data["altitude"]))
    cd0 = rocket.drag_model.cd0
    models = {f"Constant C_D = {cd0:.4f} (baseline)": ConstantDrag(cd0),
              "C_D = cd0 + k M^2 (selected)": rocket.drag_model}
    out = {}
    for label, model in models.items():
        s = summarise(simulate(variant(rocket, drag_model=model), VERTICAL, h=H_SELECTED,
                               gravity=G_OR, stop_at_apogee=True))
        out[label] = {"apogee": s["apogee"], "error_pct": pct(s["apogee"], ref_apogee)}
    results["drag_models"] = out


# ---------------------------------------------------------------------------
# 4. Sensitivity (one factor at a time)
# ---------------------------------------------------------------------------

def sensitivity(rocket):
    print("4. Sensitivity")
    nominal = fly(rocket, {})
    cases = [("Drag coefficient", "cd_factor", 0.9, 1.1, "-10 % / +10 %"),
             ("Dry mass", "mass_factor", 0.95, 1.05, "-5 % / +5 %"),
             ("Thrust (impulse)", "thrust_factor", 0.97, 1.03, "-3 % / +3 %"),
             ("Wind speed", "wind_speed", 0.0, 6.0, "0 / 6 m/s"),
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
# 5. Performance: profiling, FLOPs, before/after
# ---------------------------------------------------------------------------

def profile_top(rocket, n=8):
    """The functions with the most own time in one ascent (cProfile)."""
    pr = cProfile.Profile()
    pr.enable()
    simulate(rocket, VERTICAL, h=0.01, stop_at_apogee=True)
    pr.disable()
    stats = pstats.Stats(pr)
    total = stats.total_tt
    rows = sorted(((v[2], f"{Path(k[0]).name}:{k[2]}") for k, v in stats.stats.items()), reverse=True)[:n]
    return [{"function": name, "share_pct": 100 * t / total} for t, name in rows], total


def performance(rocket):
    print("5. Performance")
    out = {}
    out["profile"], out["profile_total_s"] = profile_top(rocket)
    out["rhs_us"] = 1e6 * time_rhs(rocket)
    np_cross, own_cross = time_cross_products()
    out["np_cross_us"], out["cross3_us"] = 1e6 * np_cross, 1e6 * own_cross

    # Before/after on the step-size settings: the original code used h = 0.01 s
    # for the whole flight; the selected settings come from the convergence study.
    t0 = time.perf_counter()
    base = simulate(rocket, VERTICAL, h=0.01)
    out["flight_s_before"] = time.perf_counter() - t0
    out["flight_rhs_before"] = base["n_rhs"]
    t0 = time.perf_counter()
    sel = simulate(rocket, VERTICAL, h=H_SELECTED, h_descent=H_DESCENT)
    out["flight_s_after"] = time.perf_counter() - t0
    out["flight_rhs_after"] = sel["n_rhs"]
    out["apogee_difference_m"] = abs(summarise(sel)["apogee"] - summarise(base)["apogee"])
    out["landing_difference_s"] = abs(summarise(sel)["landing_t"] - summarise(base)["landing_t"])

    out["flops_per_rhs"] = sum(FLOP_ESTIMATE.values())
    out["mflops_achieved"] = out["flops_per_rhs"] / out["rhs_us"]     # FLOP per microsecond = MFLOP/s
    out["history_kb_full_flight"] = (sel["x"].nbytes + sel["t"].nbytes) / 1024
    results["performance"] = out


# ---------------------------------------------------------------------------
# Results as Markdown tables (for the report)
# ---------------------------------------------------------------------------

def write_markdown():
    v = results["validation"]
    ref, sel = v["openrocket"], v["simulator"]
    lines = ["# Results (generated by scripts/make_figures.py)", "",
             "## Validation against OpenRocket", "",
             "| Quantity | OpenRocket | This simulator | Difference |", "| --- | --- | --- | --- |"]
    for key, name, unit in [("apogee", "Apogee", "m"), ("t_apogee", "Time to apogee", "s"),
                            ("max_speed", "Maximum speed", "m/s"), ("burnout_t", "Burnout", "s"),
                            ("landing_t", "Landing time", "s"), ("landing_speed", "Landing speed", "m/s"),
                            ("rail_exit_speed", "Rail exit speed", "m/s")]:
        lines.append(f"| {name} [{unit}] | {ref[key]:.3f} | {sel[key]:.3f} | {pct(sel[key], ref[key]):+.2f} % |")
    lines += ["", f"- Altitude RMSE: ascent {sel['rmse_ascent']:.2f} m, whole flight {sel['rmse_flight']:.2f} m",
              f"- Rail exit speed at OpenRocket's clearance height ({ref['rail_exit_height']:.3f} m): "
              f"{v['rail_exit_speed_at_or_height']:.2f} m/s",
              f"- Largest CP difference from OpenRocket (rail exit to 12 s): {v['cp_max_difference_mm']:.2f} mm",
              f"- Stability margin at rail exit: {v['stability_at_rail_exit']:.2f} calibres",
              f"- Drag fit C_D = cd0 + k M^2 against OpenRocket's inferred C_D: rms error {v['cd_fit_rms']:.4f}", ""]
    c = results["convergence"]
    lines += ["## Convergence", "", "| h [s] | no alignment [m] | liftoff+burnout [m] | all thrust points [m] |",
              "| --- | --- | --- | --- |"]
    for i, h in enumerate(c["h"]):
        lines.append(f"| {h} | {c['none'][i]:.2e} | {c['burnout'][i]:.2e} | {c['thrust_points'][i]:.2e} |")
    lines += ["", f"- RK4 observed order between successive h (aligned): {np.round(c['rk4_order'], 2).tolist()}",
              f"- Pitch natural frequency at max dynamic pressure: {c['pitch_omega_n']:.1f} rad/s; "
              f"RK4 stability limit 2.83/omega_n = {c['rk4_stability_h']:.3f} s",
              f"- Crosswind runs that blew up: h = {c['wind_unstable_h']}", ""]
    lines += ["", "## Drag models", "", "| Model | Apogee [m] | vs OpenRocket |", "| --- | --- | --- |"]
    for k, r in results["drag_models"].items():
        lines.append(f"| {k} | {r['apogee']:.1f} | {r['error_pct']:+.2f} % |")
    s = results["sensitivity"]
    lines += ["", f"## Sensitivity (nominal apogee {s['nominal_apogee']:.1f} m)", "",
              "| Input | Range | Apogee change [m] | Landing distance [m] |", "| --- | --- | --- | --- |"]
    for r in s["rows"]:
        lines.append(f"| {r['input']} | {r['range']} | {r['apogee_low']:+.1f} / {r['apogee_high']:+.1f} | "
                     f"{r['landing_low']:.0f} / {r['landing_high']:.0f} |")
    p = results["performance"]
    lines += ["", "## Performance", "",
              "| Measure | Original settings (h = 0.01 s throughout) | Selected (h = 0.02 s, descent 0.1 s) |",
              "| --- | --- | --- |",
              f"| Full flight, RHS calls | {p['flight_rhs_before']} | {p['flight_rhs_after']} |",
              f"| Full flight, run time [s] | {p['flight_s_before']:.2f} | {p['flight_s_after']:.2f} |",
              "", f"- Apogee difference: {p['apogee_difference_m']:.4f} m; landing time difference "
              f"{p['landing_difference_s']:.4f} s",
              f"- One right-hand-side call: {p['rhs_us']:.1f} us; numpy.cross {p['np_cross_us']:.1f} us "
              f"per call against cross3 {p['cross3_us']:.1f} us",
              f"- Estimated {p['flops_per_rhs']} FLOP per RHS call -> {p['mflops_achieved']:.1f} MFLOP/s achieved",
              f"- Full-flight history {p['history_kb_full_flight']:.0f} kB",
              "", f"Profile of one ascent ({p['profile_total_s']:.2f} s, own time):", "",
              "| Function | Share |", "| --- | --- |"]
    for r in p["profile"]:
        lines.append(f"| {r['function']} | {r['share_pct']:.1f} % |")
    (ROOT / "reports" / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  wrote reports/results.md")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    rocket = load_rocket(ROCKET_FILE)
    data, events = load_openrocket_csv(BENCHMARK_FILE)
    validation(rocket, data, events)
    convergence(rocket)
    drag_models(rocket, data)
    sensitivity(rocket)
    performance(rocket)
    write_markdown()
    print(f"Done in {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
