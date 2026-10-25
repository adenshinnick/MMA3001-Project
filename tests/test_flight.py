"""Whole-flight tests with the real rocket (aerodynamics, parachute, wind)."""
from pathlib import Path

import numpy as np
import pytest

from rocket6dof.atmosphere import G0, isa
from rocket6dof.config import load_rocket
from rocket6dof.dynamics import OMEGA, make_rhs, make_state
from rocket6dof.motor import Motor
from rocket6dof.rocket import Rocket
from rocket6dof.simulation import launch_attitude, liftoff_time, simulate, summarise
from rocket6dof.wind import Wind

ROCKET = load_rocket(Path(__file__).parent.parent / "data" / "rocket" / "l1_rocket.toml")
VERTICAL = launch_attitude(90.0)

# OpenRocket benchmark (data/benchmark/L1_benchmark.csv): same rocket, motor,
# 1.0 m vertical rod, no wind, ISA atmosphere.
OR_APOGEE = 1013.57      # [m]
OR_T_APOGEE = 12.664     # [s]
APOGEE_TOLERANCE = 0.05  # 5 %, the acceptance criterion set in the project proposal


@pytest.fixture(scope="module")
def vertical_flight():
    return simulate(ROCKET, VERTICAL, rail_length=1.0, h=0.02, h_descent=0.1)


def test_meets_benchmark_acceptance_criterion(vertical_flight):
    s = summarise(vertical_flight)
    assert vertical_flight["status"] == "landed"
    assert s["apogee"] == pytest.approx(OR_APOGEE, rel=APOGEE_TOLERANCE)
    assert s["t_apogee"] == pytest.approx(OR_T_APOGEE, rel=APOGEE_TOLERANCE)


def test_vertical_flight_in_still_air_stays_vertical(vertical_flight):
    """Symmetry: no wind and a vertical rail give no sideways force, so the
    rocket must not rotate or drift."""
    x = vertical_flight["x"]
    np.testing.assert_allclose(x[:, 0:2], 0.0, atol=1e-9)
    np.testing.assert_allclose(x[:, OMEGA], 0.0, atol=1e-12)


def test_descent_reaches_terminal_speed(vertical_flight):
    """Under the parachute, drag balances weight: v_t = sqrt(2 m g / (rho C_D A)).
    Landing speed must match the sea-level value within 0.1 %."""
    rho0 = isa(0.0)[2]
    chute = ROCKET.parachute
    v_t = np.sqrt(2.0 * ROCKET.mass(100.0) * G0 / (rho0 * chute.drag_coefficient * chute.area))
    assert vertical_flight["events"]["landing"]["speed"] == pytest.approx(v_t, rel=1e-3)


def test_drag_is_wired_into_the_equations():
    """At one coasting state, dv_z/dt must be -g - D/m with D worked out here."""
    f = make_rhs(ROCKET)
    z, vz, t = 500.0, 150.0, 5.0
    x = make_state([0.0, 0.0, z], [0.0, 0.0, vz], VERTICAL, [0.0, 0.0, 0.0])
    _, _, rho, a = isa(z)
    cd = ROCKET.drag_model(vz / a)
    drag = 0.5 * rho * vz**2 * cd * ROCKET.geometry.reference_area
    assert f(t, x)[5] == pytest.approx(-G0 - drag / ROCKET.mass(t), rel=1e-12)


def test_rocket_weathercocks_into_the_wind():
    """A stable rocket turns its nose into the relative wind, so with a wind
    from the west it flies west (x < 0) on the way up, then drifts east
    under the parachute."""
    result = simulate(ROCKET, VERTICAL, h=0.02, h_descent=0.1, wind=Wind(5.0, 270.0))
    assert result["events"]["apogee"]["position"][0] < -10.0
    assert result["events"]["landing"]["position"][0] > 100.0


def test_stops_when_leaving_the_aerodynamic_model():
    """A far too powerful motor drives the rocket past Mach 0.8: the run must
    stop with an error instead of extrapolating the aerodynamic data."""
    m = ROCKET.motor
    big = Motor("BIG", m.times, m.thrusts * 5.0, m.propellant_mass, m.total_mass, m.diameter, m.length)
    rocket = Rocket(ROCKET.dry_mass, ROCKET.inertia, big, geometry=ROCKET.geometry,
                    drag_model=ROCKET.drag_model, dry_cg=ROCKET.dry_cg,
                    motor_station=ROCKET.motor_station)
    with pytest.raises(ValueError, match="Mach"):
        simulate(rocket, VERTICAL, h=0.02)


def test_liftoff_time_by_hand():
    """Thrust rises linearly from 0 to 82.404 N at 0.007 s; liftoff is where it
    equals the weight m g."""
    t = liftoff_time(ROCKET, G0)
    assert 82.404 * t / 0.007 == pytest.approx(ROCKET.mass(t) * G0, rel=1e-9)


@pytest.mark.parametrize("bad", [-1.0, 91.0])
def test_rejects_bad_rail_elevation(bad):
    with pytest.raises(ValueError):
        launch_attitude(bad)


def test_rejects_unknown_options():
    with pytest.raises(ValueError):
        simulate(ROCKET, VERTICAL, align="sometimes")
