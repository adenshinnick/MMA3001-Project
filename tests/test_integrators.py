import numpy as np
import pytest

from rocket6dof.integrators import integrate_rk4

g=9.81

def test_vacuum_ballistic():
    """ no drag and thrust: the path must be the exact analytic parabola"""
    def f(t, x): #state is [x,y,z,vx,vy,vz]; only gravity acts along -z]
        return np.array ([x[3],x[4],x[5],0.0, 0.0 ,-g])

    x0 = np.array([0.0,0.0,0.0,10.0,0.0,50.0])
    t,x = integrate_rk4(f,x0,0.0,3.0,0.1)

    T= t[-1]
    expected = np.array([
        10.0*T, 
        0.0,
        50.0*T-0.5*g*T**2,
        10.0,
        0.0,
        50.0-g*T,
    ])
    np.testing.assert_allclose(x[-1], expected, rtol=1e-10, atol=1e-10)

def test_rk4_converges():
    """halving the step should cut the error by about 2**4=16"""

    def f(t,x):
        return -x #dx/dt = -x has the exact solution exp(-t)

    def error_at_t1(h):
        _, x  = integrate_rk4(f,np.array([1.0]),0.0,1.0,h)
        return abs(x[-1, 0]-np.exp(-1.0))

    observed_order = np.log2(error_at_t1(0.1) / error_at_t1(0.05))
    assert 3.9< observed_order <4.1

def test_rejects_nonpositive_step():
    def f(t,x):
        return x

    with pytest.raises(ValueError):
        integrate_rk4(f,np.array([1.0]),0.0,1.0,0.0)
    

def test_euler_is_first_order():
    """Halving the step should only halve the error of explicit Euler."""
    from rocket6dof.integrators import euler_step

    def error_at_t1(h):
        x, t = np.array([1.0]), 0.0
        for _ in range(int(round(1.0 / h))):
            x = euler_step(lambda t, x: -x, t, x, h)
            t += h
        return abs(x[0] - np.exp(-1.0))

    observed_order = np.log2(error_at_t1(0.01) / error_at_t1(0.005))
    assert 0.95 < observed_order < 1.05
