import numpy as np
import matplotlib.pyplot as plt

import spec_decomposition
import RK4
from evolution_eq import evolution, ev_param

N = 200
sd = spec_decomposition.spec_decom(10, N)

def foo(u, t):
    ev = evolution(u, sd)
    return ev.calc_rhs()

a_iota = 0.01
r0 = 5
sigma = 1
alpha_init = np.ones(N+1) + a_iota * sd.r_k**2 / (1 + sd.r_k**2) * ( np.exp(-(sd.r_k - r0)**2 / sigma) + np.exp(-(sd.r_k - r0)**2 / sigma) )

u = ev_param(alpha_init, np.zeros(N+1), np.ones(N+1), np.ones(N+1), np.zeros(N+1), np.zeros(N+1))

ev = evolution(u, sd)

dt = 1e-2
t_min = 0
t_max = 1

plt.plot(sd.r_k, u.alpha)
plt.show()

l = [u]

t = t_min
while t < t_max:
    u = RK4.RK4_step(t, u, dt, foo)
    l.append(u)
    t += dt
    # print(u.chi)
    # print("done with t = " + str(t))
    # print("---------------------------------------")
    # print("\n")
    # plt.plot(u.alpha)
    # plt.show()

print(u.alpha)
plt.plot(sd.r_k, u.alpha)
plt.show()

n = 152
alphas = [i.alpha[n] for i in l]
Ks = [i.K[n] for i in l]
As = [i.A[n] for i in l]
Bs = [i.B[n] for i in l]
A_as = [i.A_a[n] for i in l]
chis = [i.chi[n] for i in l]

plt.plot(alphas[:10])
print(alphas[:10])
plt.show()
plt.plot(Ks)
plt.show()
plt.plot(As)
plt.show()
plt.plot(Bs)
plt.show()
plt.plot(A_as)
plt.show()
plt.plot(chis)
plt.show()