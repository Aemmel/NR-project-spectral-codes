"""
standard Runge Kutta 4 method
"""

def RK4_step(t, u, dt, f):
    """
    evolve du/dt = f(u, t) one step
    with standard RK4 with timestep dt
    """

    k1 = f(u, t)
    k2 = f(u + dt / 2. * k1, t + dt / 2.)
    k3 = f(u + dt / 2. * k2, t + dt / 2.)
    k4 = f(u + dt * k3, t + dt / 2.)

    return u + dt / 6. * (k1 + 2*k2 + 2*k3 + k4)

## for testing:
def deriv(y, dx):
    return (y[2:] - y[:-2]) / (2. * dx)

def deriv_2(y, dx):
    return (y[2:] - 2.* y[1:-1] + y[:-2]) / (dx*dx)