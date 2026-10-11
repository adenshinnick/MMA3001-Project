"""
time integration of ordinary differential equations 
"""
import numpy as np

def rk4_step(f,t,x,h):
    '''
    advance stat by on fourth order runge kutta step
    
    Parameters
    f(callable): returns time derivitive of state as an array, same shap as x
    t(float): current time
    x(array): current state
    h(float): time step
    
    returns
    numpy.ndarray
        State at time "t+h"
'''
    k1 = f(t,x)  #slope at start
    k2 = f(t+h/2,x+h/2 *k1) #slope at midpoint using k1
    k3 = f(t+h/2, x+h/2*k2) #slope at midpoint using k2
    k4 = f(t+h, x+h*k3) #slope at end, using k3
    return x+h/6 * (k1 +2*k2 + 2*k3 + k4)

def integrate_rk4(f,x0,t0,t_end,h):
    """integrate dx/dt from t0 to t_end
    parameters
    f:function
    x0: initial state vector (array)
    t0: start time (float)
    t_end: end time (float)
    h: step size (float)
    
    returns
    t: numpy.ndarray
    times,shape (n_steps+1,)
    x:numpy.ndarray
    states,shape (n_steps+1, len(x0)), row(i) is that state at time t[i]
    
    raises ValueError if h is not positive or t_end is not after t0
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
