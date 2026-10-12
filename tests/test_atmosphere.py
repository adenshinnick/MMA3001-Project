import pytest

from rocket6dof.atmosphere import isa 

G0 = 9.80665 #standard gravity

#table values are given to approx 5 
# sig figs, so compare to a relative tolerance of 
# 1e-4

TABLE_RTOL = 1e-4

def test_sea_level_matches():
    T,p,rho,a = isa(0.0)
    assert T == pytest.approx(288.15, rel=TABLE_RTOL)
    assert p == pytest.approx(101325.0, rel=TABLE_RTOL)
    assert rho == pytest.approx(1.2250, rel=TABLE_RTOL)
    assert a == pytest.approx(340.29, rel=TABLE_RTOL)

def test_20km_matches_table():
    T, p, rho, _ = isa(20000.0)
    assert T == pytest.approx(216.65, rel=TABLE_RTOL)
    assert p == pytest.approx(5474.9, rel=TABLE_RTOL)
    assert rho == pytest.approx(0.088035, rel=TABLE_RTOL)

def test_tropopause_matches():
    T, p , rho, _ = isa(11000.0)
    assert T == pytest.approx(216.65, rel=TABLE_RTOL)
    assert p == pytest.approx(22632.1, rel=TABLE_RTOL)
    assert rho == pytest.approx(0.36392, rel=TABLE_RTOL)

def test_20km_matches():
    T, p, rho, _ = isa(20000.0)
    assert T == pytest.approx(216.65, rel=TABLE_RTOL)
    assert p == pytest.approx(5474.9, rel=TABLE_RTOL)
    assert rho == pytest.approx(0.088035, rel=TABLE_RTOL)

def test_continuous_at_tropopause():
    """Both layers must give the same values where they meet at 11 km."""
    below = isa(11000.0 - 1e-6)
    above = isa(11000.0 + 1e-6)
    assert below == pytest.approx(above, rel=1e-9)

@pytest.mark.parametrize("h",[500.0, 5000.0, 10000.0, 15000.0])
def test_hydrostatic_balance(h):
    """Pressure must fall with height as dp/dh = -rho*g0"""
    dh =1.0
    p_up = isa(h+dh)[1]
    p_down = isa(h-dh)[1]
    dp_dh = (p_up-p_down)/(2*dh) #central difference 
    rho = isa(h)[2]
    assert dp_dh == pytest.approx(-rho*G0, rel=1e-6)

def test_rejects_altitude_outside_model():
    with pytest.raises(ValueError):
        isa(25000.0)
    with pytest.raises(ValueError):
        isa(-1000.0)