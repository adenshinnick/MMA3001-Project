"""Load a rocket description from a TOML file.

TOML is a plain-text settings format: ``key = value`` lines grouped under
``[section]`` headings. Python 3.11+ reads it with the built-in ``tomllib``.

Every value is checked as it is read. A missing or invalid value stops the
load with a ``ValueError`` that names the section and field, so a typo in
the file cannot silently change the simulation.

See ``data/rocket/l1_rocket.toml`` for a complete example.
"""

import tomllib
from pathlib import Path

import numpy as np

from rocket6dof.aerodynamics import BuildupDrag, ConstantDrag, Geometry
from rocket6dof.motor import load_eng
from rocket6dof.rocket import Parachute, Rocket


def _get(table, section, key, kind=float):
    """Read ``table[section][key]`` and convert it, or raise a clear error."""
    if section not in table:
        raise ValueError(f"rocket file is missing the [{section}] section")
    if key not in table[section]:
        raise ValueError(f"rocket file is missing '{key}' in [{section}]")
    value = table[section][key]
    try:
        if kind is float and isinstance(value, bool):
            raise TypeError
        return kind(value)
    except (TypeError, ValueError):
        raise ValueError(f"[{section}] {key} = {value!r} is not a valid {kind.__name__}") from None


def load_rocket(path, drag="buildup"):
    """Build a ``Rocket`` from a TOML description file.

    Parameters
    ----------
    path : str or pathlib.Path
        The ``.toml`` file. The motor file named inside it is found relative
        to this file's folder.
    drag : str or float
        ``"buildup"`` for the Mach-dependent component build-up (default),
        or a number for a constant drag coefficient.

    Returns
    -------
    Rocket

    Raises
    ------
    ValueError
        If a section or field is missing or invalid (the message names it).
    FileNotFoundError
        If the TOML file or the motor file does not exist.
    """
    path = Path(path)
    with open(path, "rb") as file:            # tomllib needs the file opened in binary
        table = tomllib.load(file)

    geometry = Geometry(
        diameter=_get(table, "body", "diameter"),
        length=_get(table, "body", "length"),
        nose_shape=_get(table, "nose", "shape", str),
        nose_length=_get(table, "nose", "length"),
        fin_count=_get(table, "fins", "count", int),
        root_chord=_get(table, "fins", "root_chord"),
        tip_chord=_get(table, "fins", "tip_chord"),
        span=_get(table, "fins", "span"),
        sweep=_get(table, "fins", "sweep_length"),
        fin_thickness=_get(table, "fins", "thickness"),
        fin_station=_get(table, "fins", "station"),
        roughness=_get(table, "body", "surface_roughness"),
        lug_count=_get(table, "launch_lugs", "count", int) if "launch_lugs" in table else 0,
        lug_radius=_get(table, "launch_lugs", "outer_radius") if "launch_lugs" in table else 0.0,
        lug_length=_get(table, "launch_lugs", "length") if "launch_lugs" in table else 0.0,
    )

    motor = load_eng(path.parent / _get(table, "motor", "file", str))
    motor_station = _get(table, "motor", "station")
    if not 0.0 <= motor_station <= geometry.length - motor.length + 1e-9:
        raise ValueError("[motor] station puts the motor outside the body")

    dry_mass = _get(table, "mass", "dry_mass")
    dry_cg = _get(table, "mass", "dry_cg")
    if not 0.0 < dry_cg < geometry.length:
        raise ValueError(f"[mass] dry_cg = {dry_cg} must lie inside the body (0 to {geometry.length} m)")
    I_axial = _get(table, "mass", "inertia_axial")
    I_transverse = _get(table, "mass", "inertia_transverse")
    inertia = np.diag([I_axial, I_transverse, I_transverse])

    parachute = None
    if "parachute" in table:
        parachute = Parachute(_get(table, "parachute", "diameter"),
                              _get(table, "parachute", "drag_coefficient"))

    if drag == "buildup":
        drag_model = BuildupDrag(geometry)
    else:
        drag_model = ConstantDrag(float(drag))

    return Rocket(dry_mass, inertia, motor, geometry=geometry, drag_model=drag_model,
                  dry_cg=dry_cg, motor_station=motor_station, parachute=parachute,
                  name=table.get("name", path.stem))
