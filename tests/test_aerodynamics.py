import numpy as np
import pytest

from rocket6dof.aerodynamics import (
    MACH_MAX,
    BuildupDrag,
    ConstantDrag,
    DragTable,
    Geometry,
    aerodynamic_loads,
    drag_breakdown,
    fin_normal_force_slope,
    normal_force_slope_and_cp,
    skin_friction_coefficient,
    stability_margin,
)
from rocket6dof.integrators import integrate_rk4
from rocket6dof.quaternions import quat_derivative, quat_from_axis_angle, quat_to_rotation_matrix
from rocket6dof.rigid_body import angular_acceleration

# The L1 ROCKET, typed in again here so the tests do not depend on the TOML loader.
L1 = dict(diameter=0.06, length=0.77, nose_shape="ellipsoid", nose_length=0.17,
          fin_count=4, root_chord=0.117, tip_chord=0.030, span=0.045, sweep=0.08713,
          fin_thickness=0.003, fin_station=0.653, roughness=60e-6,
          lug_count=2, lug_radius=0.005, lug_length=0.03)
GEOM = Geometry(**L1)


def test_fin_slope_at_mach_zero_is_barrowmans_formula():
    """At M = 0 the fin formula must equal Barrowman's original, worked by hand:

    C_N = K * 4N (s/d)^2 / (1 + sqrt(1 + (2 l_f / (a + b))^2)), K = 1 + r/(s + r),
    with l_f the mid-chord line length.
    """
    a, b, s, m, d, N = 0.117, 0.030, 0.045, 0.08713, 0.06, 4
    l_f = np.sqrt(s**2 + (m + b / 2 - a / 2) ** 2)
    K = 1 + 0.03 / (s + 0.03)
    expected = K * 4 * N * (s / d) ** 2 / (1 + np.sqrt(1 + (2 * l_f / (a + b)) ** 2))
    assert fin_normal_force_slope(GEOM, 0.0) == pytest.approx(expected, rel=1e-12)


def test_cp_matches_hand_calculation():
    """Barrowman CP at M = 0: nose (C_N = 2) at L/3, fins at their CP."""
    a, b, m = 0.117, 0.030, 0.08713
    x_fin = 0.653 + m * (a + 2 * b) / (3 * (a + b)) + (a + b - a * b / (a + b)) / 6
    cn_fin = fin_normal_force_slope(GEOM, 0.0)
    expected = (2 * 0.17 / 3 + cn_fin * x_fin) / (2 + cn_fin)
    cn, x_cp = normal_force_slope_and_cp(GEOM, 0.0)
    assert cn == pytest.approx(2 + cn_fin)
    assert x_cp == pytest.approx(expected, rel=1e-12)


def test_cp_matches_openrocket():
    """OpenRocket (same rocket) puts the CP at 53.323 cm at Mach 0.056 and
    53.858 cm at Mach 0.570 (benchmark CSV). Agreement within 0.5 mm."""
    assert normal_force_slope_and_cp(GEOM, 0.056)[1] == pytest.approx(0.53323, abs=5e-4)
    assert normal_force_slope_and_cp(GEOM, 0.570)[1] == pytest.approx(0.53858, abs=5e-4)


def test_stability_margin_matches_openrocket():
    """OpenRocket: 2.31 calibres at rail exit (CG 39.463 cm, Mach 0.056)."""
    x_cp = normal_force_slope_and_cp(GEOM, 0.056)[1]
    assert stability_margin(x_cp, 0.39463, 0.06) == pytest.approx(2.31, abs=0.01)


@pytest.mark.parametrize("shape, fraction", [("ellipsoid", 1 / 3), ("conical", 2 / 3), ("ogive", 0.466)])
def test_nose_cp_positions(shape, fraction):
    geom = Geometry(**{**L1, "nose_shape": shape})
    assert geom.nose_cp == pytest.approx(fraction * 0.17)


def test_friction_is_roughness_limited_at_flight_speeds():
    """With a 60 um finish the roughness-limited value 0.032 (k/L)^0.2 wins
    at Re = 1e7 (about 200 m/s)."""
    expected = 0.032 * (60e-6 / 0.77) ** 0.2
    assert skin_friction_coefficient(1e7, 0.0, 60e-6, 0.77) == pytest.approx(expected)
    assert skin_friction_coefficient(1e7, 0.5, 60e-6, 0.77) == pytest.approx(expected * 0.975)


def test_drag_breakdown_adds_up_and_is_plausible():
    parts = drag_breakdown(GEOM, 0.3, 5e6)
    total = parts["friction"] + parts["fin_pressure"] + parts["nose_pressure"] + parts["base"]
    assert parts["total"] == pytest.approx(total)
    assert parts["nose_pressure"] == 0.0                  # smooth ellipsoid
    assert parts["base"] == pytest.approx(0.12 + 0.13 * 0.09)
    # OpenRocket's C_D for this rocket, inferred from the benchmark, is 0.485 at M 0.30.
    assert parts["total"] == pytest.approx(0.485, rel=0.02)


def test_drag_rises_with_mach():
    cds = [drag_breakdown(GEOM, m, 5e6)["total"] for m in [0.1, 0.3, 0.5, 0.7]]
    assert np.all(np.diff(cds) > 0.0)


def test_rejects_mach_outside_model():
    with pytest.raises(ValueError, match="Mach"):
        drag_breakdown(GEOM, MACH_MAX + 0.01, 5e6)
    with pytest.raises(ValueError, match="Mach"):
        fin_normal_force_slope(GEOM, 0.95)


@pytest.mark.parametrize(
    "field, value",
    [("diameter", 0.0), ("span", -0.01), ("fin_count", 2), ("fin_count", 3.5),
     ("nose_shape", "parabolic"), ("fin_station", 0.70), ("nose_length", 1.0)],
)
def test_geometry_rejects_invalid_fields(field, value):
    with pytest.raises(ValueError):
        Geometry(**{**L1, field: value})


def test_drag_table_interpolates_and_refuses_to_extrapolate():
    table = DragTable([0.0, 0.3, 0.6], [0.45, 0.48, 0.53])
    assert table(0.3) == pytest.approx(0.48)                  # passes through the data
    assert 0.48 < table(0.45) < 0.53                           # monotone: no overshoot
    with pytest.raises(ValueError, match="outside"):
        table(0.61)


@pytest.mark.parametrize("mach, cd", [([0.0], [0.5]), ([0.0, 0.0], [0.5, 0.5]),
                                      ([0.0, 0.5], [0.5, -0.1])])
def test_drag_table_rejects_bad_tables(mach, cd):
    with pytest.raises(ValueError):
        DragTable(mach, cd)


def test_constant_drag():
    assert ConstantDrag(0.5)(0.7, 1e6) == 0.5
    with pytest.raises(ValueError):
        ConstantDrag(0.0)


def loads(v_air, omega=(0.0, 0.0, 0.0), x_cg=0.40):
    return aerodynamic_loads(GEOM, BuildupDrag(GEOM), np.array(v_air, dtype=float),
                             1.225, 340.0, 1.8e-5, x_cg, np.array(omega, dtype=float))


def test_axial_flow_gives_pure_drag():
    force, moment, mach = loads([100.0, 0.0, 0.0])
    cd = BuildupDrag(GEOM)(100.0 / 340.0, 1.225 * 100.0 * 0.77 / 1.8e-5)
    assert force[0] == pytest.approx(-0.5 * 1.225 * 100.0**2 * GEOM.reference_area * cd)
    np.testing.assert_allclose(force[1:], 0.0)
    np.testing.assert_allclose(moment, 0.0)
    assert mach == pytest.approx(100.0 / 340.0)


def test_cross_flow_gives_restoring_moment():
    """Air-relative velocity with a +z part means the nose points 'below' the
    flight path. A stable rocket (CP behind CG) must turn its nose towards +z,
    which is a negative moment about body y."""
    force, moment, _ = loads([100.0, 0.0, 5.0])
    assert force[2] < 0.0          # normal force pushes against the cross-flow
    assert moment[1] < 0.0         # nose turns towards +z: restoring
    assert moment[2] == pytest.approx(0.0, abs=1e-12)


def test_unstable_rocket_gets_overturning_moment():
    """With the CG behind the CP the same cross-flow must turn the nose away."""
    _, moment, _ = loads([100.0, 0.0, 5.0], x_cg=0.60)
    assert moment[1] > 0.0


def test_damping_opposes_rotation():
    _, moment, _ = loads([100.0, 0.0, 0.0], omega=[0.0, 2.0, -1.0])
    assert moment[1] < 0.0 and moment[2] > 0.0


def test_no_air_speed_gives_no_load():
    force, moment, mach = loads([0.0, 0.0, 0.0])
    np.testing.assert_array_equal(force, 0.0)
    np.testing.assert_array_equal(moment, 0.0)
    assert mach == 0.0


def test_small_pitch_oscillation_frequency():
    """A stable rocket held at constant air speed oscillates in pitch like a
    damped spring. Small-angle theory gives

        omega_n^2 = q A C_N_alpha (x_cp - x_cg) / I,   zeta = c / (2 sqrt(k I)),

    with c the damping coefficient. The simulated period of the full
    nonlinear rotational equations (quaternions, Euler's equation, aero
    moment) must match omega_n sqrt(1 - zeta^2) within 1 %.
    """
    V, rho, a, mu, x_cg = 100.0, 1.225, 340.0, 1.8e-5, 0.40
    J = np.diag([3e-4, 0.05, 0.05])
    drag = BuildupDrag(GEOM)

    def f(t, s):
        q, w = s[0:4], s[4:7]
        R = quat_to_rotation_matrix(q)
        v_body = R.T @ np.array([V, 0.0, 0.0])      # air flows along Earth x
        _, moment, _ = aerodynamic_loads(GEOM, drag, v_body, rho, a, mu, x_cg, w)
        return np.concatenate((quat_derivative(q, w), angular_acceleration(J, w, moment)))

    q0 = quat_from_axis_angle([0.0, 1.0, 0.0], np.radians(1.0))   # 1 deg pitch offset
    t, s = integrate_rk4(f, np.concatenate((q0, [0.0, 0.0, 0.0])), 0.0, 1.0, 1e-4)

    w_y = s[:, 5]
    ups = np.where((w_y[:-1] < 0.0) & (w_y[1:] >= 0.0))[0]          # upward zero crossings
    crossings = t[ups] - w_y[ups] * (t[ups + 1] - t[ups]) / (w_y[ups + 1] - w_y[ups])
    period = np.mean(np.diff(crossings))

    mach = V / a
    cn_alpha, x_cp = normal_force_slope_and_cp(GEOM, mach)
    qA = 0.5 * rho * V**2 * GEOM.reference_area
    k = qA * cn_alpha * (x_cp - x_cg)
    c = qA / V * (2.0 * (GEOM.nose_cp - x_cg) ** 2
                  + fin_normal_force_slope(GEOM, mach) * (GEOM.fin_cp - x_cg) ** 2)
    omega_n = np.sqrt(k / J[1, 1])
    zeta = c / (2.0 * np.sqrt(k * J[1, 1]))
    expected_period = 2.0 * np.pi / (omega_n * np.sqrt(1.0 - zeta**2))
    assert period == pytest.approx(expected_period, rel=0.01)
