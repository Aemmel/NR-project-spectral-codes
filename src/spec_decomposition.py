"""
Spectral decomposition stuff
like definition of even and odd sines, integration points etc.
"""

import numpy as np

class spec_decom:
    def __init__(self, L0, N):
        self.L0 = L0
        self.r_k = self.quad_points(N)

        # init the even and odd sines
        # we need SB_0 up to (incl.) SB_(2N)
        self.SBN = np.zeros((2*N + 1, len(self.r_k)))
        self.SBN_dr = np.zeros_like(self.SBN)
        self.SBN_ddr = np.zeros_like(self.SBN)

        # precalculate all values of SBN and SBN_dr
        self.func_SBN(2*N)
        self.func_SBN_dr(2*N)
        self.func_SBN_ddr(2*N)

    # quadrature points based on
    # chebyshev Gauss Lobatto points for k=1,2,..,N+1
    # beEq. 48 and below
    def quad_points(self, N):
        k = np.array(range(1, N+2)) # N+2 excluded
        x_k = np.cos(np.pi * k / (2. * N + 2))

        return self.L0 * x_k / np.sqrt(1 - x_k**2)

    # even and odd sines SB_2n(r), SB_2n+1(r)
    # Eq. 38, 39
    def func_SBN(self, n):
        L0 = self.L0
        r = self.r_k
        # prevent calculating the same thing twice
        if np.count_nonzero(self.SBN[n]) == 0:
            if n == 0:
                self.SBN[n] = 1. / np.sqrt(1. + r**2 / L0**2)
            elif n == 1:
                self.SBN[n] = 2. * r / L0 * 1. / (1. + r**2 / L0**2)
            else:
                self.SBN[n] = 2. * r / L0 * 1. / np.sqrt(1. + r**2 / L0**2) * self.func_SBN(n-1) - self.func_SBN(n-2)
        
        return self.SBN[n]

    # d SB_n(r) / dr
    # calculated using Mathematica
    def func_SBN_dr(self, n):
        L0 = self.L0
        r = self.r_k
        # prevent calculating the same thing twice
        if np.count_nonzero(self.SBN_dr[n]) == 0:
            if n == 0:
                self.SBN_dr[n] = - r / L0**2 * (1. + r**2 / L0**2)**-1.5
            elif n == 1:
                self.SBN_dr[n] = 2. * L0 * (L0**2 - r**2) / (L0**2 + r**2)**2
            else:
                self.SBN_dr[n] = (2.*L0**2 * self.func_SBN(n-1) - (L0**2+r**2)
                    *(-2.*r*self.func_SBN_dr(n-1) + L0*np.sqrt(1 + r**2 / L0**2)*self.func_SBN_dr(n-2)) ) \
                    / (L0*(L0**2 + r**2)*np.sqrt(1 + r**2 / L0**2))
        
        return self.SBN_dr[n]
    
    # d^2 SB_n(r) / dr^2
    # calculated using Mathematica
    def func_SBN_ddr(self, n):
        # prevent calculating the same thing twice
        if np.count_nonzero(self.SBN_ddr[n]) == 0:
            L0 = self.L0
            r = self.r_k
            if n == 0:
                self.SBN_ddr[n] = (- L0**2 + 2. * r**2) / ( (L0**2 + r**2)**2 * np.sqrt(1 + r**2 / L0**2))
            elif n == 1:
                self.SBN_ddr[n] = 4. * L0 * r * (-3. * L0**2 + r**2) / (L0**2 + r**2)**3
            else:
                self.SBN_ddr[n] = ( - 6. * L0**2 * r * self.func_SBN(n-1) - (L0**2 + r**2) * ( -4. * L0**2 * self.func_SBN_dr(n-1) \
                    + (L0**2 + r**2)*(-2.*r*self.func_SBN_ddr(n-1) + L0 * np.sqrt(1+r**2 / L0**2) * self.func_SBN_ddr(n-2)) ) ) \
                    / (L0 * (L0**2 + r**2)**2 * np.sqrt(1 + r**2 / L0**2))

        return self.SBN_ddr[n]
