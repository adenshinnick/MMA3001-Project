"""Time integration of ordinary differential equations dx/dt = f(t, x).

Fixed-step methods: classical RK4 (the main method) and explicit Euler
(the baseline in the integrator comparison).
"""
import numpy as np

def rk4_step(f,t,x,h):
    """Advance the state by one classical fourth-order Runge-Kutta step.

    Parameters
    ----------
    f : callable
        Right-hand side, called as ``f(t, x)``; returns dx/dt, same shape as x.
    t : float
        Current time [s].
    x : numpy.ndarray
        Current state.
    h : float
        Time step [s].

    Returns
    -------
    numpy.ndarray
        State at time ``t + h``.
    """
    k1 = f(t,x)  #slope at start
    k2 = f(t+h/2,x+h/2 *k1) #slope at midpoint using k1
    k3 = f(t+h/2, x+h/2*k2) #slope at midpoint using k2
    k4 = f(t+h, x+h*k3) #slope at end, using k3
    return x+h/6 * (k1 +2*k2 + 2*k3 + k4)

def euler_step(f, t, x, h):
    """Advance the state by one explicit (forward) Euler step.

    x(t + h) = x(t) + h f(t, x). First-order accurate: halving h only halves
    the error. Used as the baseline method in the integrator comparison.

    Parameters
    ----------
    f : callable
        Right-hand side, called as ``f(t, x)``.
    t : float
        Current time [s].
    x : numpy.ndarray
        Current state.
    h : float
        Time step [s].

    Returns
    -------
    numpy.ndarray
        State at time ``t + h``.
    """
    return x + h * f(t, x)


STEPPERS = {"rk4": rk4_step, "euler": euler_step}
"""Fixed-step methods by name, for ``simulate(method=...)``."""


def integrate_rk4(f,x0,t0,t_end,h):
    """Integrate dx/dt = f(t, x) from t0 to t_end with fixed-step RK4.

    Parameters
    ----------
    f : callable
        Right-hand side, called as ``f(t, x)``.
    x0 : array_like
        Initial state vector.
    t0 : float
        Start time [s].
    t_end : float
        End time [s]. If ``t_end - t0`` is not a whole number of steps,
        the run stops at the nearest whole number of steps; check ``t[-1]``.
    h : float
        Step size [s]. Must be positive.

    Returns
    -------
    t : numpy.ndarray
        Times, shape ``(n_steps + 1,)``.
    x : numpy.ndarray
        States, shape ``(n_steps + 1, len(x0))``. Row ``i`` is the state
        at time ``t[i]``.

    Raises
    ------
    ValueError
        If ``h`` is not positive or ``t_end`` is not after ``t0``.
    """
    if h <= 0:
        raise ValueError(f"Step size h must be positive, got {h}.")
    if t_end <= t0:
        raise ValueError(f"t_end ({t_end}) must be after t0 ({t0}).")

    x0 =np.asarray(x0, dtype = float)
    n_steps = int(round((t_end-t0)/h))

    t=np.empty(n_steps+1)
    x= np.empty((n_steps+1, x0.size))
    t[0]= t0
    x[0]= x0

    for i in range(n_steps):
        x[i+1] = rk4_step(f,t[i],x[i],h)
        t[i+1] = t0 + (i+1)*h #not t[i] + h, round off can't accumulate

    return t, x
