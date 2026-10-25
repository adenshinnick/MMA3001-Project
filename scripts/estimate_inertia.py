"""Estimate the dry L1 ROCKET's mass, CG and inertia from its parts.

The OpenRocket CSV export gives the total mass and CG but not the moments of
inertia, which the 6-DOF rotation equations need. This script rebuilds the
dry rocket from the parts listed in ``data/rocket/L1 ROCKET.ork`` (sizes and
materials copied from that file), checks the total mass and CG against
OpenRocket, and prints the inertia to put in ``data/rocket/l1_rocket.toml``.

Run:  python scripts/estimate_inertia.py
"""

from math import pi

from rocket6dof.mass_properties import Part, combine, ellipsoid_nose, trapezoidal_fins, tube

# OpenRocket values for the dry rocket (from the benchmark CSV at t = 0, with
# the 207 g motor removed; see data/rocket/l1_rocket.toml).
OR_DRY_MASS = 0.675724      # [kg]
OR_DRY_CG = 0.31464         # [m]

BODY_TOP = 0.17             # body tube starts where the nose ends [m]
BASE = 0.77                 # station of the base [m]

parts = [
    ellipsoid_nose("Nose cone (ABS)", 1050.0, 0.17, 0.03, 0.0175),
    tube("Nose shoulder", 1050.0, 0.06, 0.02775, 0.02575, BODY_TOP),
    tube("Bulkhead (birch)", 670.0, 0.003, 0.02775, 0.0, BODY_TOP + 0.06 - 0.003),
    Part("Eye bolt", 0.030, 0.231 + 0.010),
    Part("Nut", 0.003, 0.216 + 0.005),
    tube("Body tube (cardboard)", 680.0, 0.60, 0.03, 0.0284, BODY_TOP),
    trapezoidal_fins("Fins x4 (birch ply)", 630.0, 4, 0.117, 0.030, 0.045, 0.0871,
                     0.003, BASE - 0.117, 0.03),
    # Fin tabs (87 mm x 13.5 mm) inside the body, treated as a point mass.
    Part("Fin tabs", 630.0 * 0.003 * 0.087 * 0.0135 * 4, BASE - 0.117 + 0.005 + 0.0585),
    tube("Motor mount tube (phenolic)", 950.0, 0.203, 0.0165, 0.0145, BASE - 0.203),
    tube("Centring ring, aft", 680.0, 0.009, 0.0284, 0.0165, BASE - 0.009),
    tube("Centring ring, fwd", 680.0, 0.009, 0.0284, 0.0165, BASE - 0.098 - 0.009),
    Part("Parachute + lines", pi * 0.6**2 * 0.067 + 8 * 0.9 * 0.0018, BODY_TOP + 0.2 + 0.0325),
    Part("Launch lugs x2", 0.020, BODY_TOP + 0.30 + 0.155),
]


def main():
    """Print the parts table, the totals, and the comparison with OpenRocket."""
    print(f"{'Part':30s} {'mass [g]':>9s} {'x_cg [m]':>9s}")
    for p in parts:
        print(f"{p.name:30s} {1000 * p.mass:9.1f} {p.x_cg:9.4f}")
    mass, x_cg, I_axial, I_transverse = combine(parts)
    print()
    print(f"Total mass  {1000 * mass:7.1f} g   (OpenRocket {1000 * OR_DRY_MASS:.1f} g, "
          f"difference {100 * (mass / OR_DRY_MASS - 1):+.2f} %)")
    print(f"CG station  {x_cg:7.4f} m   (OpenRocket {OR_DRY_CG:.4f} m, "
          f"difference {1000 * (x_cg - OR_DRY_CG):+.1f} mm)")
    # Move the transverse inertia to OpenRocket's CG (parallel-axis theorem),
    # because the simulation uses OpenRocket's CG.
    I_t_at_or_cg = I_transverse + mass * (x_cg - OR_DRY_CG) ** 2
    print(f"I_axial       = {I_axial:.3e} kg m^2")
    print(f"I_transverse  = {I_t_at_or_cg:.3e} kg m^2 (about the OpenRocket CG)")


if __name__ == "__main__":
    main()
