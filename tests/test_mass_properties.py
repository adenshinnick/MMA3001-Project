import numpy as np
import pytest

from rocket6dof.mass_properties import Part, combine, ellipsoid_nose, trapezoidal_fins, tube


def test_thin_tube_matches_textbook_formulas():
    """Solid cylinder: I_axial = m r^2 / 2, I_transverse = m (3 r^2 + L^2) / 12."""
    p = tube("rod", 1000.0, 0.5, 0.02, 0.0, 0.1)
    m = 1000.0 * np.pi * 0.02**2 * 0.5
    assert p.mass == pytest.approx(m)
    assert p.x_cg == pytest.approx(0.35)
    assert p.I_axial == pytest.approx(0.5 * m * 0.02**2)
    assert p.I_transverse == pytest.approx(m * (3 * 0.02**2 + 0.5**2) / 12)


def test_tube_rejects_bad_dimensions():
    with pytest.raises(ValueError):
        tube("bad", 1000.0, 0.5, 0.01, 0.02, 0.0)      # inner radius bigger than outer


def test_solid_ellipsoid_nose_matches_exact_volume_and_cg():
    """A wall thicker than the radius makes the nose solid: half a prolate
    spheroid, volume (2/3) pi R^2 L, CG 5L/8 from the tip."""
    L, R, rho = 0.17, 0.03, 1050.0
    p = ellipsoid_nose("solid", rho, L, R, wall=1.0)
    assert p.mass == pytest.approx(rho * 2.0 / 3.0 * np.pi * R**2 * L, rel=1e-5)
    assert p.x_cg == pytest.approx(5.0 * L / 8.0, rel=1e-4)
    # Axial inertia of half a spheroid: (2/5) m R^2.
    assert p.I_axial == pytest.approx(0.4 * p.mass * R**2, rel=1e-4)


def test_fin_mass():
    p = trapezoidal_fins("fins", 630.0, 4, 0.117, 0.03, 0.045, 0.0871, 0.003, 0.653, 0.03)
    assert p.mass == pytest.approx(630.0 * 0.003 * 0.5 * (0.117 + 0.03) * 0.045 * 4)


def test_combine_uses_parallel_axis_theorem():
    """Two 1 kg point masses 1 m apart: CG in the middle, I = 2 * 1 * 0.5^2."""
    mass, x_cg, I_axial, I_transverse = combine([Part("a", 1.0, 0.0), Part("b", 1.0, 1.0)])
    assert mass == 2.0
    assert x_cg == pytest.approx(0.5)
    assert I_axial == 0.0
    assert I_transverse == pytest.approx(0.5)


def test_combine_rejects_empty_list():
    with pytest.raises(ValueError):
        combine([])
