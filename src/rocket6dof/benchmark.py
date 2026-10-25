"""Read an OpenRocket simulation export and compare a flight against it.

OpenRocket's CSV export has comment lines starting with ``#``. One of them
lists the column names with their units, for example
``# Time (s),Altitude (ft),Vertical velocity (m/s),...``, and others record
events such as ``# Event APOGEE occurred at t=12.714 seconds``.

``load_openrocket_csv`` converts every recognised column to SI units and
refuses columns whose units it does not know, so a changed export setting
(feet instead of metres, grams instead of kilograms) cannot slip through.
"""

import re

import numpy as np

from rocket6dof.atmosphere import isa

# Multipliers that convert each unit OpenRocket can write into SI.
UNIT_TO_SI = {
    "s": 1.0, "m": 1.0, "ft": 0.3048, "cm": 0.01, "mm": 0.001, "km": 1000.0,
    "m/s": 1.0, "ft/s": 0.3048, "km/h": 1.0 / 3.6, "mph": 0.44704,
    "m/s²": 1.0, "m/s^2": 1.0, "ft/s²": 0.3048, "G": 9.80665,
    "kg": 1.0, "g": 0.001, "lb": 0.45359237, "oz": 0.028349523125,
    "N": 1.0, "°": np.pi / 180.0, "rad": 1.0, "kg·m²": 1.0, "kg/m³": 1.0,
    "": 1.0, "​": 1.0,     # dimensionless columns (OpenRocket writes a zero-width space)
}

# Short names for the columns the comparison uses.
COLUMN_NAMES = {
    "Time": "t",
    "Altitude": "altitude",
    "Vertical velocity": "vz",
    "Total velocity": "speed",
    "Total acceleration": "acceleration",
    "Mass": "mass",
    "CP location": "cp",
    "CG location": "cg",
    "Stability margin calibers": "stability",
    "Mach number": "mach",
    "Drag coefficient": "cd",
}

_HEADER = re.compile(r"^(.*?)\s*\((.*)\)\s*$")    # "Altitude (ft)" -> ("Altitude", "ft")
_EVENT = re.compile(r"Event (\w+) occurred at t=([\d.eE+-]+) seconds")


def load_openrocket_csv(path):
    """Load an OpenRocket CSV export, converted to SI units.

    Parameters
    ----------
    path : str or pathlib.Path
        The exported ``.csv`` file.

    Returns
    -------
    data : dict of numpy.ndarray
        One array per column, keyed by the short names in ``COLUMN_NAMES``
        (other columns keep their OpenRocket name). SI units: s, m, m/s,
        m/s^2, kg.
    events : dict of float
        Event name to time [s], for example ``events["APOGEE"]``.

    Raises
    ------
    ValueError
        If there is no header line, a unit is not recognised, a row has the
        wrong number of values, or the time column is missing.
    """
    header = None
    rows = []
    events = {}
    with open(path, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                event = _EVENT.search(line)
                if event:
                    events.setdefault(event.group(1), float(event.group(2)))
                elif "," in line and "(" in line and header is None:
                    header = [c.strip() for c in line.lstrip("#").split(",")]
                continue
            values = line.split(",")
            if header is None:
                raise ValueError(f"{path}: data found before the column header line")
            if len(values) != len(header):
                raise ValueError(f"{path}: row has {len(values)} values, header has {len(header)}")
            rows.append([float(v) for v in values])   # "NaN" is read as nan

    if header is None or not rows:
        raise ValueError(f"{path}: no column header or no data rows found")
    table = np.array(rows)

    data = {}
    for k, column in enumerate(header):
        match = _HEADER.match(column)
        name, unit = (match.group(1), match.group(2).strip()) if match else (column, "")
        if unit not in UNIT_TO_SI:
            raise ValueError(f"{path}: column '{column}' has a unit ('{unit}') this loader "
                             "does not know; re-export in SI units")
        data[COLUMN_NAMES.get(name, name)] = table[:, k] * UNIT_TO_SI[unit]
    if "t" not in data:
        raise ValueError(f"{path}: no 'Time' column")
    return data, events


def openrocket_gravity(latitude_deg, altitude):
    """Gravity used by OpenRocket at a latitude and altitude [m/s^2].

    OpenRocket uses the WGS84 ellipsoidal gravity formula (Somigliana) with an
    inverse-square fall-off with height. It differs from the standard
    9.80665 m/s^2 by about 0.15 % at the benchmark's latitude of 28.6 deg.
    """
    s2 = np.sin(np.radians(latitude_deg)) ** 2
    g_surface = 9.7803267714 * (1.0 + 0.00193185138639 * s2) / np.sqrt(1.0 - 0.00669437999013 * s2)
    return g_surface * (6371000.0 / (6371000.0 + altitude)) ** 2


def infer_drag_coefficient(data, reference_area, latitude_deg, t_start, t_end):
    """Back out OpenRocket's drag coefficient from a vertical coast.

    During a vertical coast only gravity and drag act, both downwards, so the
    exported total acceleration is |a| = g + D/m. Hence

        D  = m (|a| - g),     C_D = D / (0.5 rho V^2 A)

    with rho from the ISA at the exported altitude. This needs no thrust
    data, so the inferred C_D is independent of the motor model.

    Parameters
    ----------
    data : dict
        From ``load_openrocket_csv``; needs t, altitude, speed, acceleration, mass.
    reference_area : float
        Reference area used for C_D [m^2].
    latitude_deg : float
        Launch latitude set in OpenRocket [deg] (for its gravity model).
    t_start, t_end : float
        Coast window [s]: after burnout, ending before speed gets so small
        that the rounding in the export dominates.

    Returns
    -------
    mach, cd : numpy.ndarray
        Mach number and inferred drag coefficient, sorted by Mach.
    """
    keep = (data["t"] > t_start) & (data["t"] < t_end)
    h = data["altitude"][keep]
    V = data["speed"][keep]
    D = data["mass"][keep] * (data["acceleration"][keep] - openrocket_gravity(latitude_deg, h))
    rho = np.array([isa(z)[2] for z in h])
    a = np.array([isa(z)[3] for z in h])
    cd = D / (0.5 * rho * V**2 * reference_area)
    order = np.argsort(V / a)
    return (V / a)[order], cd[order]


def compare(sim_t, sim_z, ref_t, ref_z):
    """Root-mean-square difference of two altitude histories [m].

    The simulation is linearly interpolated onto the reference times that lie
    inside both runs.

    Parameters
    ----------
    sim_t, sim_z : numpy.ndarray
        Simulated times [s] and altitudes [m].
    ref_t, ref_z : numpy.ndarray
        Reference (OpenRocket) times [s] and altitudes [m].

    Returns
    -------
    float
    """
    inside = (ref_t >= sim_t[0]) & (ref_t <= sim_t[-1])
    z_sim = np.interp(ref_t[inside], sim_t, sim_z)
    return float(np.sqrt(np.mean((z_sim - ref_z[inside]) ** 2)))
