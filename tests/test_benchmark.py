import numpy as np
import pytest

from rocket6dof.atmosphere import isa
from rocket6dof.benchmark import (compare, infer_drag_coefficient, load_openrocket_csv,
                                  openrocket_gravity)

CSV = """# Test run
# Time (s),Altitude (ft),Total velocity (m/s),Mass (g)
# Event LAUNCH occurred at t=0 seconds
0,0,0,1000
# Event APOGEE occurred at t=1.5 seconds
1,100,10,900
2,200,NaN,800
"""


def test_units_are_converted_to_si(tmp_path):
    path = tmp_path / "run.csv"
    path.write_text(CSV, encoding="utf-8")
    data, events = load_openrocket_csv(path)
    np.testing.assert_allclose(data["altitude"], [0.0, 30.48, 60.96])
    np.testing.assert_allclose(data["mass"], [1.0, 0.9, 0.8])
    assert np.isnan(data["speed"][2])
    assert events == {"LAUNCH": 0.0, "APOGEE": 1.5}


def test_unknown_unit_is_refused(tmp_path):
    path = tmp_path / "run.csv"
    path.write_text(CSV.replace("Mass (g)", "Mass (stone)"), encoding="utf-8")
    with pytest.raises(ValueError, match="stone"):
        load_openrocket_csv(path)


def test_ragged_row_is_refused(tmp_path):
    path = tmp_path / "run.csv"
    path.write_text(CSV + "3,300\n", encoding="utf-8")
    with pytest.raises(ValueError, match="values"):
        load_openrocket_csv(path)


def test_missing_header_is_refused(tmp_path):
    path = tmp_path / "run.csv"
    path.write_text("0,0,0,1000\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_openrocket_csv(path)


def test_openrocket_gravity_at_equator_and_pole():
    assert openrocket_gravity(0.0, 0.0) == pytest.approx(9.7803267714)
    assert openrocket_gravity(90.0, 0.0) == pytest.approx(9.8321849, rel=1e-6)  # WGS84 polar value


def test_drag_inference_recovers_a_known_cd():
    """Build fake coast data with C_D = 0.5 exactly and check it comes back."""
    A, lat, m = 0.0028, 28.61, 0.77
    h = np.linspace(100.0, 900.0, 50)
    V = np.linspace(200.0, 20.0, 50)
    rho = np.array([isa(z)[2] for z in h])
    accel = openrocket_gravity(lat, h) + 0.5 * rho * V**2 * 0.5 * A / m
    data = {"t": np.linspace(2.0, 10.0, 50), "altitude": h, "speed": V,
            "acceleration": accel, "mass": np.full(50, m)}
    mach, cd = infer_drag_coefficient(data, A, lat, 1.0, 11.0)
    np.testing.assert_allclose(cd, 0.5, rtol=1e-12)
    assert np.all(np.diff(mach) > 0.0)


def test_compare_gives_rms_difference():
    t = np.array([0.0, 1.0, 2.0])
    assert compare(t, np.array([0.0, 1.0, 2.0]), t, np.array([0.0, 2.0, 2.0])) == pytest.approx(np.sqrt(1 / 3))
