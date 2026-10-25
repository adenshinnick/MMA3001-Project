"""Read an OpenRocket CSV export and compare a flight against it.

Columns are converted to SI units. A unit this loader does not know stops
the load, so a changed export setting (feet, grams) cannot slip through.
"""

import re

import numpy as np

from rocket6dof.atmosphere import isa

# Multipliers that convert the units in the export to SI.
UNIT_TO_SI = {"s": 1.0, "m": 1.0, "ft": 0.3048, "cm": 0.01, "m/s": 1.0, "ft/s": 0.3048,
              "m/s²": 1.0, "kg": 1.0, "g": 0.001, "": 1.0, "​": 1.0}

# Short names for the columns used here.
COLUMN_NAMES = {"Time": "t", "Altitude": "altitude", "Vertical velocity": "vz",
                "Total velocity": "speed", "Total acceleration": "acceleration",
                "Mass": "mass", "CP location": "cp", "CG location": "cg",
                "Stability margin calibers": "stability"}

# Gravity in the OpenRocket run: its latitude-dependent value at the
# benchmark's launch latitude (28.61 deg), instead of the standard 9.80665.
G_OPENROCKET = 9.792  # [m/s^2]


def load_openrocket_csv(path):
    """Load an OpenRocket CSV export, converted to SI units.

    Parameters
    ----------
    path : str or pathlib.Path
        The exported ``.csv`` file.

    Returns
    -------
    data : dict of numpy.ndarray
        One array per column, keyed by the short names in ``COLUMN_NAMES``.
    events : dict of float
        Event name to time [s], for example ``events["APOGEE"]``.

    Raises
    ------
    ValueError
        If there is no header line or a unit is not recognised.
    """
    header, rows, events = None, [], {}
    with open(path, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line.startswith("# Event"):
                name, time = re.search(r"Event (\w+) occurred at t=([\d.]+)", line).groups()
                events.setdefault(name, float(time))
            elif line.startswith("#") and "," in line and header is None:
                header = [c.strip() for c in line.lstrip("#").split(",")]
            elif line and not line.startswith("#"):
                rows.append([float(v) for v in line.split(",")])
    if header is None or not rows:
        raise ValueError(f"{path}: no column header or no data rows found")

    table = np.array(rows)
    data = {}
    for k, column in enumerate(header):
        name, unit = re.match(r"(.*?)\s*\((.*)\)", column).groups()
        if unit not in UNIT_TO_SI:
            raise ValueError(f"{path}: column '{column}' has a unit ('{unit}') this loader "
                             "does not know; re-export in SI units")
        data[COLUMN_NAMES.get(name, name)] = table[:, k] * UNIT_TO_SI[unit]
    return data, events


def infer_drag_coefficient(data, reference_area, t_start, t_end):
    """OpenRocket's drag coefficient, backed out of a vertical coast.

    In a vertical coast only gravity and drag act, both downwards, so the
    exported total acceleration is |a| = g + D/m. Hence D = m (|a| - g) and
    C_D = D / (0.5 rho V^2 A), with rho from the ISA.

    Parameters
    ----------
    data : dict
        From ``load_openrocket_csv``.
    reference_area : float
        Reference area [m^2].
    t_start, t_end : float
        Coast window [s], after burnout and before the speed gets small.

    Returns
    -------
    mach, cd : numpy.ndarray
    """
    keep = (data["t"] > t_start) & (data["t"] < t_end)
    h, V = data["altitude"][keep], data["speed"][keep]
    drag = data["mass"][keep] * (data["acceleration"][keep] - G_OPENROCKET)
    rho = np.array([isa(z)[2] for z in h])
    sound = np.array([isa(z)[3] for z in h])
    return V / sound, drag / (0.5 * rho * V**2 * reference_area)


def fit_mach_drag(mach, cd):
    """Least-squares fit of C_D = cd0 + k M^2. Returns (cd0, k)."""
    k, cd0 = np.polyfit(mach**2, cd, 1)
    return cd0, k


def rms_difference(sim_t, sim_z, ref_t, ref_z):
    """Root-mean-square altitude difference [m], over the times both runs cover."""
    inside = (ref_t >= sim_t[0]) & (ref_t <= sim_t[-1])
    z_sim = np.interp(ref_t[inside], sim_t, sim_z)
    return float(np.sqrt(np.mean((z_sim - ref_z[inside]) ** 2)))
