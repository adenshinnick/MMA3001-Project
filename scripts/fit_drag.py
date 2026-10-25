"""Fit the drag law C_D = cd0 + k M^2 to OpenRocket's benchmark data.

Prints the two numbers that go in the [drag] section of
data/rocket/l1_rocket.toml.

Run:  python scripts/fit_drag.py
"""

from pathlib import Path

import numpy as np

from rocket6dof.benchmark import fit_mach_drag, infer_drag_coefficient, load_openrocket_csv

ROOT = Path(__file__).resolve().parent.parent
data, events = load_openrocket_csv(ROOT / "data" / "benchmark" / "L1_benchmark.csv")

# Coast window: just after burnout, until the speed is small enough that
# rounding in the export dominates.
mach, cd = infer_drag_coefficient(data, np.pi * 0.03**2, events["BURNOUT"] + 0.001, 11.5)
cd0, k = fit_mach_drag(mach, cd)
residual = cd - (cd0 + k * mach**2)

print(f"{len(cd)} points, Mach {mach.min():.3f} to {mach.max():.3f}")
print(f"cd0 = {cd0:.4f}, k = {k:.4f}")
print(f"fit error: rms {np.sqrt(np.mean(residual**2)):.4f}, worst {np.abs(residual).max():.4f}")
