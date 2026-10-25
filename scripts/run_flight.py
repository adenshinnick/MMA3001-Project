"""Simulate one flight of a rocket and print the headline results.

Examples (run from the project folder with the virtual environment active):

    python scripts/run_flight.py
    python scripts/run_flight.py --elevation 85 --wind 4 --wind-from 270
    python scripts/run_flight.py --rocket data/rocket/l1_rocket.toml --h 0.01

The flight path is saved to reports/figures/flight.png.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rocket6dof.config import load_rocket
from rocket6dof.simulation import launch_attitude, simulate, summarise
from rocket6dof.wind import Wind

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Simulate one rocket flight.")
    parser.add_argument("--rocket", default=str(ROOT / "data" / "rocket" / "l1_rocket.toml"),
                        help="rocket description file (.toml)")
    parser.add_argument("--elevation", type=float, default=90.0, help="rail elevation [deg]")
    parser.add_argument("--azimuth", type=float, default=90.0,
                        help="direction the rail leans towards, clockwise from north [deg]")
    parser.add_argument("--rail", type=float, default=1.0, help="rail length [m]")
    parser.add_argument("--wind", type=float, default=0.0, help="wind speed [m/s]")
    parser.add_argument("--wind-from", type=float, default=270.0,
                        help="direction the wind blows from, clockwise from north [deg]")
    parser.add_argument("--site-altitude", type=float, default=0.0, help="launch site altitude [m]")
    parser.add_argument("--h", type=float, default=0.02, help="ascent time step [s]")
    parser.add_argument("--h-descent", type=float, default=0.1, help="descent time step [s]")
    args = parser.parse_args()

    rocket = load_rocket(args.rocket)
    wind = Wind(args.wind, args.wind_from) if args.wind > 0 else None
    result = simulate(rocket, launch_attitude(args.elevation, args.azimuth), rail_length=args.rail,
                      h=args.h, h_descent=args.h_descent, wind=wind,
                      launch_altitude=args.site_altitude)
    s = summarise(result)
    land = result["events"].get("landing", {}).get("position")

    print(f"Rocket:          {rocket.name} with {rocket.motor.name}")
    print(f"Status:          {result['status']}")
    print(f"Rail exit speed: {s['rail_exit_speed']:.1f} m/s")
    print(f"Burnout:         {s['burnout_t']:.3f} s")
    print(f"Max speed:       {s['max_speed']:.1f} m/s (Mach {s['max_mach']:.3f})")
    print(f"Apogee:          {s['apogee']:.1f} m at {s['t_apogee']:.2f} s")
    if land is not None:
        print(f"Landing:         {s['landing_t']:.1f} s, {s['landing_speed']:.2f} m/s, "
              f"{land[0]:.0f} m east, {land[1]:.0f} m north of the pad")
    print(f"Cost:            {s['n_steps']} steps, {s['n_rhs']} right-hand-side calls")

    t, x = result["t"], result["x"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
    ax1.plot(t, x[:, 2], color="#2a78d6")
    ax1.set(xlabel="Time [s]", ylabel="Altitude [m]", title="Altitude")
    ax2.plot(x[:, 0], x[:, 1], color="#2a78d6")
    ax2.plot(0, 0, "^", color="#eb6834", label="Launch pad")
    ax2.set(xlabel="East [m]", ylabel="North [m]", title="Ground track")
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.legend()
    fig.tight_layout()
    out = ROOT / "reports" / "figures" / "flight.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
