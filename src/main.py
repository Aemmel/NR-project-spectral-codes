import numpy as np
import matplotlib.pyplot as plt

import time

import spec_decomposition
import RK4
from evolution_eq import ev_param, evolution_numba


# N = 320
# L0 = 30
# sd = spec_decomposition.spec_decom(L0, N)

def base_func(u, t, sd):
    ev = evolution_numba(u, sd)
    return ev.calc_rhs()

def evolve(N, t_min, t_max, dt, foo, ev_init=None, print_every=np.inf):
    if ev_init is None:
        u = ev_param(np.ones(N+1), np.zeros(N+1), np.ones(N+1), np.ones(N+1), np.zeros(N+1), np.zeros(N+1), np.zeros(N+1))
    else:
        u = ev_init
    
    params = {t_min : u}

    t = t_min
    t_printer = 0
    while t <= t_max:
        try:
            u = RK4.RK4_step(t, u, dt, foo)
        except np.linalg.LinAlgError:
            print("Oh oh.... NaNs and infs again :((")
            print("Happened at t=" + str(t))
            break
        t += dt
        t_printer += dt

        print("done with " + str(t))

        # if t > 0:
        #     plt.plot(u.alpha)
        #     plt.show()

        if t_printer > print_every:
            params[t] = u
            t_printer = 0
    
    # account for overcounting
    params[t - dt] = u

    return params

def RMS(f, sd):
    """ root mean square c.f. Eq. 55
    """
    return np.sqrt( 0.5 * sd.quadrature(f**2) )

def create_alpha_init_gauss(sd, a_iota, r0, sigma):
    return 1. + a_iota * sd.r_k**2 / (1 + sd.r_k**2) * ( np.exp(-(sd.r_k - r0)**2 / sigma) + np.exp(-(sd.r_k - r0)**2 / sigma) )

def create_alpha_dr_init_gauss(sd, a_iota, r0, sigma):
    """ d/dr alpha(0, r)
    """
    r = sd.r_k

    return 4. * np.exp(-(r-r0)**2 / sigma**2) * r * a_iota * ( -r * (1. + r**2) * (r - r0) + sigma**2 ) / ( (1. + r**2)**2 * sigma**2 )

def create_alpha_ddr_init_gauss(sd, a_iota, r0, sigma):
    """ d^2/dr^2 alpha(0, r)
    """
    r = sd.r_k

    return 4. * np.exp(-(r-r0)**2 / sigma**2) * a_iota * (2. * (r + r**3)**2 * (r-r0)**2 \
        - (r + r**3) * (5. * r + r**3 - 4 * r0) * sigma**2 + (1. - 3. * r**2) * sigma**4 ) / ( (1. + r**2)**3 * sigma**4 )


def monitor_constraints_over_time(N, L0):
    sd = spec_decomposition.spec_decom(L0, N)
    foo = lambda u, t: base_func(u, t, sd)

    alpha_init = create_alpha_init_gauss(N, sd, 0.001, 5, 1)
    u = ev_param(alpha_init, np.zeros(N+1), np.ones(N+1), np.ones(N+1), np.zeros(N+1), np.zeros(N+1), np.zeros(N+1))

    d = evolve(N, 0, 10, 5e-4, foo, u, 0.1)

    # for i in d:
    #     plt.plot(d[i].alpha)
    #     plt.title("t=" + str(i))
    #     plt.show()
    
    times = np.zeros(len(d))
    hamiltonian = np.zeros(len(d))
    momentum = np.zeros(len(d))
    Lambda = np.zeros(len(d))

    for i, key in enumerate(d):
        ev = evolution_numba(d[key], sd)
        times[i] = key
        hamiltonian[i] = RMS(ev.constraint_H(), sd)
        momentum[i] = RMS(ev.constraint_Mr(), sd)
        Lambda[i] = RMS(ev.constraint_Lambda(), sd)
    
    return times, hamiltonian, momentum, Lambda

def spec_decom_convergence():
    """ get convergence of spectral decomposition as tested on initial gaussian data of alpha
    """
    L0 = [50] # [10, 20, 30, 50, 100]
    N_range = range(20, 321, 20)
    # N_range = range(1, 321)

    for l in L0:
        for N in N_range:
            sd = spec_decomposition.spec_decom(l, N)
            alpha_init = create_alpha_init_gauss(sd, 0.01, 5, 1)
            u = ev_param(alpha_init, np.zeros(N+1), np.ones(N+1), np.ones(N+1), np.zeros(N+1), np.zeros(N+1), np.zeros(N+1))
            ev = evolution_numba(u, sd)
            modes_alpha = evolution_numba.get_modes_alpha(sd.even, ev.alpha)
            alpha_dr_N = evolution_numba.deriv_r_alpha(sd.even_dr, modes_alpha) # numerical derivative
            alpha_ddr_N = evolution_numba.deriv_r_alpha(sd.even_ddr, modes_alpha)

            alpha_dr = create_alpha_dr_init_gauss(sd, 0.01, 5, 1)
            alpha_ddr = create_alpha_ddr_init_gauss(sd, 0.01, 5, 1)

            print(np.log10(RMS(alpha_dr_N - alpha_dr, sd)))
            print(np.log10(RMS(alpha_ddr_N - alpha_ddr, sd)))


# times, H, Mr, L = monitor_constraints_over_time(250, 30)
# np.savetxt("out/constraints_N=250_L0=30_T=0to10_dt=5e-4.txt", [times, H, Mr, L])
spec_decom_convergence()

# N=320, dt=1e-4, T=15: took 73,8min
# N=80,  dt=1e-4, T=15: took 8min

# with Numba:

# N=320, dt=1e-4, T=15: took 68min
# N=80,  dt=1e-4, T=15: took 6.8min after improving: 5.3min
