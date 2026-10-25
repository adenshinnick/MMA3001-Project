from pathlib import Path

import numpy as np
import pytest

from rocket6dof.aerodynamics import ConstantDrag
from rocket6dof.config import load_rocket

ROCKET_FILE = Path(__file__).parent.parent / "data" / "rocket" / "l1_rocket.toml"


def test_loads_l1_rocket_and_matches_openrocket_mass_properties():
    """OpenRocket: 882.724 g with CG at 39.762 cm at ignition, 767.724 g with
    CG at 35.703 cm after burnout (benchmark CSV)."""
    rocket = load_rocket(ROCKET_FILE)
    assert rocket.mass(0.0) == pytest.approx(0.882724, abs=1e-6)
    assert rocket.mass(5.0) == pytest.approx(0.767724, abs=1e-6)
    assert rocket.cg(0.0) == pytest.approx(0.39762, abs=5e-4)
    assert rocket.cg(5.0) == pytest.approx(0.35703, abs=5e-4)
    assert rocket.motor.name == "H182"
    assert rocket.parachute.diameter == 1.2


def test_constant_drag_option():
    rocket = load_rocket(ROCKET_FILE, drag=0.5)
    assert isinstance(rocket.drag_model, ConstantDrag)


def write_variant(tmp_path, old, new):
    """Copy the rocket file with one line changed, next to a copy of the motor."""
    text = ROCKET_FILE.read_text(encoding="utf-8")
    assert old in text
    (tmp_path / "rocket").mkdir()
    (tmp_path / "motors").mkdir()
    motor = ROCKET_FILE.parent.parent / "motors" / "AeroTech_H182R.eng"
    (tmp_path / "motors" / motor.name).write_text(motor.read_text(encoding="utf-8"))
    path = tmp_path / "rocket" / "variant.toml"
    path.write_text(text.replace(old, new), encoding="utf-8")
    return path


def test_missing_field_is_named(tmp_path):
    path = write_variant(tmp_path, "span = 0.045", "")
    with pytest.raises(ValueError, match="span"):
        load_rocket(path)


def test_invalid_value_is_rejected(tmp_path):
    path = write_variant(tmp_path, "span = 0.045", "span = -0.045")
    with pytest.raises(ValueError, match="span"):
        load_rocket(path)


def test_wrong_type_is_rejected(tmp_path):
    path = write_variant(tmp_path, "dry_mass = 0.675724", 'dry_mass = "heavy"')
    with pytest.raises(ValueError, match="dry_mass"):
        load_rocket(path)


def test_cg_outside_body_is_rejected(tmp_path):
    path = write_variant(tmp_path, "dry_cg = 0.31464", "dry_cg = 1.5")
    with pytest.raises(ValueError, match="dry_cg"):
        load_rocket(path)


def test_missing_motor_file(tmp_path):
    path = write_variant(tmp_path, "AeroTech_H182R.eng", "missing.eng")
    with pytest.raises(FileNotFoundError):
        load_rocket(path)
