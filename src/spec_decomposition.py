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

        # save basis for values as copies, to allow greater numba speedup

        # for alpha, chi, K
        self.even = self.SBN[::2].transpose().copy()
        self.even_dr = self.SBN_dr[::2].transpose().copy()
        self.even_ddr = self.SBN_ddr[::2].transpose().copy()

        # for A, B, A_a
        self.even_m = (self.SBN[2::2, :-1] - self.SBN[0:-2:2, -1]).transpose().copy()
        self.even_m_dr = (self.SBN_dr[2::2, :-1] - self.SBN_dr[0:-2:2, -1]).transpose().copy()
        self.even_m_ddr = (self.SBN_ddr[2::2, :-1] - self.SBN_ddr[0:-2:2, -1]).transpose().copy()

        # for Lambda
        self.odd = self.SBN[1::2, :-1].transpose().copy()
        self.odd_dr = self.SBN_dr[1::2, :-1].transpose().copy()
        self.odd_ddr = self.SBN_ddr[1::2, :-1].transpose().copy()

    # quadrature points based on
    # chebyshev Gauss Lobatto points for k=1,2,..,N+1
    # beEq. 48 and below
    def quad_points(self, N):
        # k = np.array(range(1, N+2)) # N+2 excluded
        k = np.arange(1., N + 2.) # N+2 excluded
        x_k = np.cos(np.pi * k / (2. * N + 2))

        r_k = self.L0 * x_k / np.sqrt(1. - x_k**2)

        return r_k

    def quadrature(self, f):
        """ quadrature at quadrate points r_k
        provide: f(r_k)

        method used: https://en.wikipedia.org/wiki/Clenshaw%E2%80%93Curtis_quadrature#Integration_on_infinite_and_semi-infinite_intervals
        """
        N = len(self.r_k) - 1 # r_k is N+1 points
        n_k = np.pi * np.arange(1., N+2) / (2*N + 2)

        return self.L0 * np.pi / (2*N+2.) * np.sum( (f / np.sin(n_k)**2)[1:-1] )

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
                self.SBN_dr[n] = (2.*L0**2 * self.func_SBN(n-1) - (L0**2+r**2) \
                    * (-2. * r * self.func_SBN_dr(n-1) + L0 * np.sqrt(1 + r**2 / L0**2) * self.func_SBN_dr(n-2)) ) \
                    / (L0*(L0**2 + r**2)*np.sqrt(1. + r**2 / L0**2))
        
        return self.SBN_dr[n]
    
    # d^2 SB_n(r) / dr^2
    # calculated using Mathematica
    def func_SBN_ddr(self, n):
        # prevent calculating the same thing twice
        if np.count_nonzero(self.SBN_ddr[n]) == 0:
            L0 = self.L0
            r = self.r_k
            if n == 0:
                self.SBN_ddr[n] = (- L0**2 + 2. * r**2) / ( (L0**2 + r**2)**2 * np.sqrt(1. + r**2 / L0**2))
            elif n == 1:
                self.SBN_ddr[n] = 4. * L0 * r * (-3. * L0**2 + r**2) / (L0**2 + r**2)**3
            else:
                self.SBN_ddr[n] = ( - 6. * L0**2 * r * self.func_SBN(n-1) - (L0**2 + r**2) * ( -4. * L0**2 * self.func_SBN_dr(n-1) \
                    + (L0**2 + r**2)*(-2.*r*self.func_SBN_ddr(n-1) + L0 * np.sqrt(1. + r**2 / L0**2) * self.func_SBN_ddr(n-2)) ) ) \
                    / (L0 * (L0**2 + r**2)**2 * np.sqrt(1. + r**2 / L0**2))

        return self.SBN_ddr[n]
