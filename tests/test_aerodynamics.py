import numpy as np
import pytest

from rocket6dof.aerodynamics import (MACH_MAX, ConstantDrag, Geometry, MachDrag,
                                     aerodynamic_loads, fin_normal_force_slope,
                                     normal_force_slope_and_cp, stability_margin)
from rocket6dof.integrators import integrate_rk4
from rocket6dof.quaternions import quat_derivative, quat_from_axis_angle, quat_to_rotation_matrix
from rocket6dof.rigid_body import angular_acceleration

# The L1 ROCKET, typed in again so the tests do not depend on the TOML loader.
L1 = dict(diameter=0.06, length=0.77, nose_length=0.17, fin_count=4, root_chord=0.117,
          tip_chord=0.030, span=0.045, sweep=0.08713, fin_station=0.653)
GEOM = Geometry(**L1)
DRAG = MachDrag(0.47, 0.16)


def test_fin_slope_at_mach_zero_is_barrowmans_formula():
    """At M = 0 the fin formula must equal Barrowman's, worked by hand:
    C_N = K 4N (s/d)^2 / (1 + sqrt(1 + (2 l_f / (a + b))^2)), K = 1 + r/(s + r)."""
    a, b, s, m, d, N = 0.117, 0.030, 0.045, 0.08713, 0.06, 4
    l_f = np.sqrt(s**2 + (m + b / 2 - a / 2) ** 2)            # mid-chord line length
    K = 1 + 0.03 / (s + 0.03)
    expected = K * 4 * N * (s / d) ** 2 / (1 + np.sqrt(1 + (2 * l_f / (a + b)) ** 2))
    assert fin_normal_force_slope(GEOM, 0.0) == pytest.approx(expected, rel=1e-12)


def test_cp_matches_openrocket():
    """OpenRocket (same rocket) puts the CP at 53.323 cm at Mach 0.056 and
    53.858 cm at Mach 0.570 (benchmark CSV). Agreement within 0.5 mm."""
    assert normal_force_slope_and_cp(GEOM, 0.056)[1] == pytest.approx(0.53323, abs=5e-4)
    assert normal_force_slope_and_cp(GEOM, 0.570)[1] == pytest.approx(0.53858, abs=5e-4)


def test_stability_margin_matches_openrocket():
    """OpenRocket: 2.31 calibres at rail exit (CG 39.463 cm, Mach 0.056)."""
    x_cp = normal_force_slope_and_cp(GEOM, 0.056)[1]
    assert stability_margin(x_cp, 0.39463, 0.06) == pytest.approx(2.31, abs=0.01)


def test_drag_models():
    assert ConstantDrag(0.5)(0.7) == 0.5
    assert DRAG(0.5) == pytest.approx(0.47 + 0.16 * 0.25)
    with pytest.raises(ValueError):
        ConstantDrag(0.0)
    with pytest.raises(ValueError, match="Mach"):
        DRAG(MACH_MAX + 0.01)


@pytest.mark.parametrize("field, value", [("diameter", 0.0), ("span", -0.01), ("fin_count", 2),
                                          ("fin_station", 0.70), ("nose_length", 1.0)])
def test_geometry_rejects_invalid_fields(field, value):
    with pytest.raises(ValueError):
        Geometry(**{**L1, field: value})


def loads(v_air, omega=(0.0, 0.0, 0.0), x_cg=0.40):
    return aerodynamic_loads(GEOM, DRAG, np.array(v_air, dtype=float), 1.225, 340.0, x_cg,
                             np.array(omega, dtype=float))


def test_axial_flow_gives_pure_drag():
    force, moment, mach = loads([100.0, 0.0, 0.0])
    assert force[0] == pytest.approx(-0.5 * 1.225 * 100.0**2 * GEOM.reference_area * DRAG(100 / 340))
    np.testing.assert_allclose(force[1:], 0.0)
    np.testing.assert_allclose(moment, 0.0)


def test_cross_flow_gives_restoring_moment():
    """Air-relative velocity with a +z part: a stable rocket (CP behind CG)
    must turn its nose towards +z, a negative moment about body y."""
    force, moment, _ = loads([100.0, 0.0, 5.0])
    assert force[2] < 0.0
    assert moment[1] < 0.0


def test_unstable_rocket_gets_overturning_moment():
    _, moment, _ = loads([100.0, 0.0, 5.0], x_cg=0.60)      # CG behind the CP
    assert moment[1] > 0.0


def test_damping_opposes_rotation():
    _, moment, _ = loads([100.0, 0.0, 0.0], omega=[0.0, 2.0, -1.0])
    assert moment[1] < 0.0 and moment[2] > 0.0


def test_small_pitch_oscillation_frequency():
    """Held at constant air speed, a stable rocket oscillates in pitch like a
    damped spring: omega_n^2 = q A C_N_alpha (x_cp - x_cg) / I. The simulated
    period must match 2 pi / (omega_n sqrt(1 - zeta^2)) within 1 %."""
    V, rho, a, x_cg = 100.0, 1.225, 340.0, 0.40
    J = np.diag([3e-4, 0.05, 0.05])

    def f(t, s):
        q, w = s[0:4], s[4:7]
        v_body = quat_to_rotation_matrix(q).T @ np.array([V, 0.0, 0.0])
        _, moment, _ = aerodynamic_loads(GEOM, DRAG, v_body, rho, a, x_cg, w)
        return np.concatenate((quat_derivative(q, w), angular_acceleration(J, w, moment)))

    q0 = quat_from_axis_angle([0.0, 1.0, 0.0], np.radians(1.0))   # 1 deg pitch offset
    t, s = integrate_rk4(f, np.concatenate((q0, [0.0, 0.0, 0.0])), 0.0, 1.0, 1e-4)
    w_y = s[:, 5]
    ups = np.where((w_y[:-1] < 0.0) & (w_y[1:] >= 0.0))[0]          # upward zero crossings
    crossings = t[ups] - w_y[ups] * (t[ups + 1] - t[ups]) / (w_y[ups + 1] - w_y[ups])
    period = np.mean(np.diff(crossings))

    cn_alpha, x_cp = normal_force_slope_and_cp(GEOM, V / a)
    qA = 0.5 * rho * V**2 * GEOM.reference_area
    k = qA * cn_alpha * (x_cp - x_cg)
    c = qA / V * (2.0 * (GEOM.nose_cp - x_cg) ** 2
                  + fin_normal_force_slope(GEOM, V / a) * (GEOM.fin_cp - x_cg) ** 2)
    omega_n = np.sqrt(k / J[1, 1])
    zeta = c / (2.0 * np.sqrt(k * J[1, 1]))
    assert period == pytest.approx(2 * np.pi / (omega_n * np.sqrt(1 - zeta**2)), rel=0.01)
