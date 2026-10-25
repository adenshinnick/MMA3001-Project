"""Load a rocket description from a TOML file (see ``data/rocket/l1_rocket.toml``).

TOML is a plain-text settings format: ``key = value`` lines under
``[section]`` headings, read by Python's built-in ``tomllib``. Every value is
checked as it is read; a missing or invalid value raises a ``ValueError``
naming its section and field.
"""

import tomllib
from pathlib import Path

import numpy as np

from rocket6dof.aerodynamics import ConstantDrag, Geometry, MachDrag
from rocket6dof.motor import load_eng
from rocket6dof.rocket import Parachute, Rocket


def _get(table, section, key, kind=float):
    """Read ``table[section][key]`` and convert it, or raise a clear error."""
    if key not in table.get(section, {}):
        raise ValueError(f"rocket file is missing '{key}' in [{section}]")
    value = table[section][key]
    try:
        if isinstance(value, bool):
            raise TypeError
        return kind(value)
    except (TypeError, ValueError):
        raise ValueError(f"[{section}] {key} = {value!r} is not a valid {kind.__name__}") from None


def load_rocket(path, constant_cd=None):
    """Build a ``Rocket`` from a TOML description file.

    Parameters
    ----------
    path : str or pathlib.Path
        The ``.toml`` file. The motor file named inside it is found relative
        to this file's folder.
    constant_cd : float, optional
        Use this constant drag coefficient instead of the file's C_D(M) law.

    Returns
    -------
    Rocket

    Raises
    ------
    ValueError
        If a field is missing or invalid (the message names it).
    FileNotFoundError
        If the TOML file or the motor file does not exist.
    """
    path = Path(path)
    # Read the whole file into a nested dictionary: table["fins"]["span"], etc.
    with open(path, "rb") as file:            # tomllib needs the file opened in binary
        table = tomllib.load(file)

    # Each _get call checks that the field exists and has the right type;
    # Geometry, Rocket and Parachute then check the values themselves.
    geometry = Geometry(
        diameter=_get(table, "body", "diameter"),
        length=_get(table, "body", "length"),
        nose_length=_get(table, "nose", "length"),
        fin_count=_get(table, "fins", "count", int),
        root_chord=_get(table, "fins", "root_chord"),
        tip_chord=_get(table, "fins", "tip_chord"),
        span=_get(table, "fins", "span"),
        sweep=_get(table, "fins", "sweep_length"),
        fin_station=_get(table, "fins", "station"),
    )

    # The motor file path is relative to the rocket file's folder.
    motor = load_eng(path.parent / _get(table, "motor", "file", str))
    motor_station = _get(table, "motor", "station")
    if not 0.0 <= motor_station <= geometry.length - motor.length + 1e-9:
        raise ValueError("[motor] station puts the motor outside the body")

    dry_cg = _get(table, "mass", "dry_cg")
    if not 0.0 < dry_cg < geometry.length:
        raise ValueError(f"[mass] dry_cg = {dry_cg} must lie inside the body (0 to {geometry.length} m)")
    # The rocket is symmetric about its axis, so pitch and yaw inertia are equal.
    I_t = _get(table, "mass", "inertia_transverse")
    inertia = np.diag([_get(table, "mass", "inertia_axial"), I_t, I_t])

    if constant_cd is None:
        drag_model = MachDrag(_get(table, "drag", "cd0"), _get(table, "drag", "k"))
    else:
        drag_model = ConstantDrag(constant_cd)

    parachute = None                         # the parachute section is optional
    if "parachute" in table:
        parachute = Parachute(_get(table, "parachute", "diameter"),
                              _get(table, "parachute", "drag_coefficient"))

    return Rocket(_get(table, "mass", "dry_mass"), inertia, motor, geometry=geometry,
                  drag_model=drag_model, dry_cg=dry_cg, motor_station=motor_station,
                  parachute=parachute, name=table.get("name", path.stem))
