"""
Encode the evolution equations Eq.2 to Eq.9
"""

from spec_decomposition import spec_decom
import numpy as np

from numba import jit

class ev_param:
    """ parameters that are carried over to the next timestep
    """
    def __init__(self, alpha, K, A, B, A_a, chi, Lambda):
        self.alpha = alpha
        self.K = K
        self.A = A
        self.B = B
        self.A_a = A_a
        self.chi = chi
        self.Lambda = Lambda

    def __add__(self, o):
        alpha = self.alpha + o.alpha
        K = self.K + o.K
        A = self.A + o.A
        B = self.B + o.B
        A_a = self.A_a + o.A_a
        chi = self.chi + o.chi
        Lambda = self.Lambda + o.Lambda

        return ev_param(alpha, K , A, B, A_a, chi, Lambda)

    def __rmul__(self, f):
        alpha = self.alpha * f
        K = self.K * f
        A = self.A * f
        B = self.B * f
        A_a = self.A_a * f
        chi = self.chi * f
        Lambda = self.Lambda * f

        return ev_param(alpha, K, A, B, A_a, chi, Lambda)
    
    def __str__(self):
        out = "alpha=" + str(self.alpha) + "\nK=" + str(self.K) + "\nA=" + str(self.A) + "\nB=" + str(self.B) + "\nA_a=" + str(self.A_a) + "\nchi=" + str(self.chi) + "\nLambda=" + str(self.Lambda)

        return out

    
class evolution_numba:
    def __init__(self, ev_p, spec):
        """ We have to provide evolution params

        Furthermore provide a spectral decomposition

        All equations from arXiv:2105.09094v2 
        """        
        # spectral decomposition data
        self.sd = spec
        self.r = self.sd.r_k

        ## initial data
        self.alpha = ev_p.alpha
        self.K = ev_p.K
        self.A = ev_p.A
        self.B = ev_p.B
        self.A_a = ev_p.A_a
        self.chi = ev_p.chi
        self.Lambda = ev_p.Lambda

        # fix boundary conditions at r = 0
        self.A[-1] = 1.
        self.B[-1] = 1.
        self.A_a[-1] = 0.

        ## from notes: step 2 to 5
        # self.Lambda = np.zeros(len(self.alpha)) # has to be done in calc_derivatives().. unfortunately
        self.psi = evolution_numba.calc_rhs_psi(self.chi)

        self.A_t_rr = self.A_a * self.A # tilde(A)_rr
        self.A_t_tt = - 1./2. * self.A_a * self.B # tilde(A)_theta,theta

        ## for the rest we need the derivative terms of alpha, psi, K, A, B, and Lambda
        # boundary conditions fixed in calc_derivatives()
        N = len(self.alpha) # N+1 in our previous nomenclature
        self.alpha_dr = np.zeros(N)
        self.chi_dr = np.zeros(N)
        self.psi_dr = np.zeros(N)
        self.K_dr = np.zeros(N)
        self.A_dr = np.zeros(N)
        self.B_dr = np.zeros(N)
        self.Lambda_dr = np.zeros(N)
        self.alpha_ddr = np.zeros(N)
        self.chi_ddr = np.zeros(N)
        self.psi_ddr = np.zeros(N)
        self.A_ddr = np.zeros(N)
        self.B_ddr = np.zeros(N)
        self.calc_derivatives()

        ## now compute D's and R's
        # step 6
        self.D_rr = evolution_numba.calc_rhs_D_rr(self.alpha_dr, self.alpha_ddr, self.A, self.A_dr, self.psi, self.psi_dr)
        self.D_tt = evolution_numba.calc_rhs_D_tt(self.r, self.alpha_dr, self.A, self.B, self.B_dr, self.psi, self.psi_dr)
        self.R_rr = evolution_numba.calc_rhs_R_rr(self.A, self.A_dr, self.A_ddr, self.B, self.B_dr, self.Lambda, self.Lambda_dr, self.r, self.psi, self.psi_dr, self.psi_ddr)
        self.R_tt = evolution_numba.calc_rhs_R_tt(self.A, self.A_dr, self.B, self.B_dr, self.B_ddr, self.Lambda, self.r, self.psi, self.psi_dr, self.psi_ddr)

        # step 7
        self.D_TF_rr = evolution_numba.calc_rhs_D_TF_rr(self.D_rr, self.D_tt, self.r, self.A, self.B)
        self.D_TF_tt = evolution_numba.calc_rhs_D_TF_tt(self.D_tt, self.A, self.B)
        self.R_TF_rr = evolution_numba.calc_rhs_R_TF_rr(self.R_rr, self.R_tt, self.r, self.A, self.B)
        self.R_TF_tt = evolution_numba.calc_rhs_R_TF_tt(self.R_tt, self.A, self.B)

        #step 8
        self.D = evolution_numba.calc_rhs_D(self.psi, self.A, self.B, self.D_rr, self.D_tt, self.r)

        # fix boundary conditions at r=0
        # self.D_rr[-1] = self.alpha_ddr[-1]
        # self.D_tt[-1] = 0
        # self.R_rr[-1] = 4. * self.psi_ddr[-1] # ? assumes dr^2 A = 0 at r=0
        # self.R_tt[-1] = 0
        # self.D_TF_rr[-1] = 2./3. * self.alpha_ddr[-1]
        # self.D_TF_tt[-1] = 0
        # self.R_TF_rr[-1] = 8./3. * self.psi_ddr[-1]
        # self.R_TF_tt[-1] = 0
        # self.D[-1] = self.alpha_ddr[-1] / self.psi[-1]**4

    def calc_derivatives(self):
        """ calculate all derivative terms
        """
        # get modes
        modes_alpha = evolution_numba.get_modes_alpha(self.sd.even, self.alpha)
        modes_chi = evolution_numba.get_modes_chi(self.sd.even, self.chi)
        modes_K = evolution_numba.get_modes_K(self.sd.even, self.K)

        # following modes are only N long, not N+1
        modes_A = evolution_numba.get_modes_A(self.sd.even_m, self.A)
        modes_B = evolution_numba.get_modes_B(self.sd.even_m, self.B)

        # calculate derivative terms from modes
        self.alpha_dr = evolution_numba.deriv_r_alpha(self.sd.even_dr, modes_alpha)
        self.chi_dr = evolution_numba.deriv_r_chi(self.sd.even_dr, modes_chi)
        self.K_dr = evolution_numba.deriv_r_K(self.sd.even_dr, modes_K)
        self.K_dr[-1] = 0 # BC
        self.A_dr[:-1] = evolution_numba.deriv_r_A(self.sd.even_m_dr, modes_A) # d/dr A(r_(N+1)) = 0
        self.B_dr[:-1] = evolution_numba.deriv_r_B(self.sd.even_m_dr, modes_B) # d/dr B(r_(N+1)) = 0
        self.alpha_ddr = evolution_numba.deriv_rr_alpha(self.sd.even_ddr, modes_alpha)
        self.chi_ddr = evolution_numba.deriv_rr_chi(self.sd.even_ddr, modes_chi)
        self.A_ddr[:-1] = evolution_numba.deriv_rr_A(self.sd.even_m_ddr, modes_A) # d^2/dr^2 A(r_(N+1)) = 0 ??
        self.B_ddr[:-1] = evolution_numba.deriv_rr_B(self.sd.even_m_ddr, modes_B) # d^2/dr^2 B(r_(N+1)) = 0 ??
        
        self.chi_dr[-1] = 0

        # self.Lambda = self.calc_rhs_Lambda()
        modes_Lambda = evolution_numba.get_modes_Lambda(self.sd.odd, self.Lambda)
        self.Lambda_dr[:-1] = evolution_numba.deriv_r_Lambda(self.sd.odd_dr, modes_Lambda) # d/dr Lambda(r_(N+1)) = 0

        # derivative of psi was defined with psi and derivative of chi
        self.psi_dr = evolution_numba.deriv_r_psi(self.psi, self.chi_dr)
        self.psi_ddr = evolution_numba.deriv_rr_psi(self.psi, self.chi_dr, self.chi_ddr)

        # boundary condition r = 0
        self.psi_dr[-1] = 0

    def calc_rhs(self):
        """ return d/dt alpha, ... for all ev_param's
        """
        dt_ev_p = ev_param(None, None, None, None, None, None, None)

        dt_ev_p.alpha = evolution_numba.calc_rhs_dt_alpha(self.alpha, self.K)
        dt_ev_p.A = evolution_numba.calc_rhs_dt_A(self.alpha, self.A_t_rr)
        dt_ev_p.B = evolution_numba.calc_rhs_dt_B(self.alpha, self.A_t_tt)
        dt_ev_p.K = evolution_numba.calc_rhs_dt_K(self.alpha, self.K, self.A_t_rr, self.A, self.A_t_tt, self.B, self.D)
        dt_ev_p.Lambda = evolution_numba.calc_rhs_dt_Lambda(self.alpha, self.A, self.A_t_tt, self.psi, self.psi_dr, self.K_dr, self.A_t_rr, self.A_dr, self.B, self.B_dr, self.r, self.alpha_dr)
        
        psi_dt = evolution_numba.calc_rhs_dt_psi(self.alpha, self.psi, self.K)
        # d/dt chi = 1 / psi * d/dt psi
        dt_ev_p.chi = 1. / self.psi * psi_dt

        A_t_rr_dt = evolution_numba.calc_rhs_dt_A_t_rr(self.psi, self.D_TF_rr, self.alpha, self.R_TF_rr, self.A_t_rr, self.K, self.A)

        dt_ev_p.A_a = ( A_t_rr_dt * self.A - self.A_t_rr * dt_ev_p.A ) / self.A**2

        return dt_ev_p

    def constraint_H(self):
        """ Hamiltonian constraint Eq. 17
        """
        K = self.K
        A_t_rr = self.A_t_rr
        A = self.A
        A_t_tt = self.A_t_tt
        B = self.B
        psi = self.psi
        Lambda_dr = self.Lambda_dr
        Lambda = self.Lambda
        A_dr = self.A_dr
        A_ddr = self.A_ddr
        B_dr = self.B_dr
        B_ddr = self.B_ddr
        r = self.r
        psi_dr = self.psi_dr
        psi_ddr = self.psi_ddr

        return 2./3. * K**2 - A_t_rr**2 / A**2 - 2. * A_t_tt**2 / B**2 \
            + 1./psi**4 * ( Lambda_dr + 1./2. * Lambda * A_dr / A + Lambda * B_dr / B + 2. * Lambda / r ) \
            - 8. / (A * psi**5) * (psi_ddr - 1./2. * A_dr * psi_dr / A + B_dr * psi_dr / B + 2. * psi_dr / r) \
            - 1. / (A * psi**4) * (1./2. * A_ddr / A - 3./4. * A_dr**2 / A**2 + B_ddr \
                - 1./2. * B_dr**2 / B**2 + 2. * B_dr / (r*B) + A_dr / (r*B) )
    
    def constraint_Mr(self):
        """ momentum constraint Eq. 18
        """
        # we need derivative of A_t_rr = A_a * A: A_t_rr_dr = A_a_dr * A + A_a * A_dr
        A_a_dr = np.zeros(len(self.alpha))
        modes_A_a = evolution_numba.get_modes_A_a(self.sd.even_m, self.A_a)
        A_a_dr[:-1] = evolution_numba.deriv_r_A_a(self.sd.even_m_dr, modes_A_a)
        A_t_rr_dr = A_a_dr * self.A + self.A_a * self.A_dr

        K_dr = self.K_dr
        A = self.A
        A_t_rr = self.A_t_rr
        psi = self.psi
        psi_dr = self.psi_dr
        A_dr = self.A_dr
        B = self.B
        B_dr = self.B_dr
        A_t_tt = self.A_t_tt
        r = self.r

        return 2./3. * K_dr - A_t_rr_dr / A - 6. * A_t_rr / A * psi_dr / psi + A_t_rr * A_dr / A**2 \
            - B_dr * A_t_rr / (A * B) + B_dr * A_t_tt / B**2 - 2. * A_t_rr / (r * A) + 2. * A_t_tt / (r * B)

    def constraint_Lambda(self):
        """ constraint for Lambda Eq. 20
        """
        return evolution_numba.calc_rhs_Lambda(self.A, self.A_dr, self.B, self.B_dr, self.r)

    ################################################################
    ## calculate RHs
    @jit(nopython=True)
    def calc_rhs_Lambda(A, A_dr, B, B_dr, r):
        """ calculate Lambda hat from A, B, quad points r (Eq. 20)
        """
        return 1. / A * ( A_dr / (2. * A) - B_dr / B - 2. / r * (1 - A / B) )

    @jit(nopython=True)
    def calc_rhs_psi(chi):
        """ calculate psi from chi (Eq. 19)
        """
        return np.exp(chi)

    @jit(nopython=True)
    def calc_rhs_chi(psi):
        """ calculate chi from psi (Eq. 19)
        """
        return np.log(psi)
    
    @jit(nopython=True)
    def calc_rhs_D_rr(alpha_dr, alpha_ddr, A, A_dr, psi, psi_dr):
        """ calculate D_rr from alpha, A, psi (Eq. 11)
        """
        return alpha_ddr - alpha_dr / 2. * (A_dr / A + 4. * psi_dr / psi)

    @jit(nopython=True)
    def calc_rhs_D_tt(r, alpha_dr, A, B, B_dr, psi, psi_dr):
        """ calculate D_theta,theta from alpha, A, B, psi, r (Eq. 12)
        """
        return  r * alpha_dr / A * ( B + r / 2. * (B_dr + 4. * B * psi_dr / psi) )

    @jit(nopython=True)
    def calc_rhs_R_rr(A, A_dr, A_ddr, B, B_dr, Lambda, Lambda_dr, r, psi, psi_dr, psi_ddr):
        """ calculate R_rr from A, B, Lambda hat, psi, r (Eq. 13)
        """
        return  3./4. * A_dr**2 / A**2 - B_dr**2 / (2. * B**2) + A * Lambda_dr + 1./2. * Lambda * A_dr \
            + 1./r * ( -4.*psi_dr / psi -  1. / B * (A_dr + 2. * B_dr) + 2. * A * B_dr / B**2) \
            - 4. * (psi_ddr * psi - psi_dr**2) / psi**2 + 2. * psi_dr / psi * ( A_dr / A - B_dr / B ) \
            - A_ddr / (2. * A) + 2. * (A - B) / (r**2 * B)

    @jit(nopython=True)
    def calc_rhs_R_tt(A, A_dr, B, B_dr, B_ddr, Lambda, r, psi, psi_dr, psi_ddr):
        """ calculate R_theta,theta from A, B, Lambda hat, psi, r (Eq. 14)
        """
        return r**2 * B / A * ( psi_dr / psi * A_dr / A - 2 * (psi_ddr * psi - psi_dr ** 2) / psi**2 - 4. * (psi_dr / psi)**2 ) \
            + r**2 / A * ( B_dr**2 / (2. * B) - 3. * psi_dr / psi * B_dr - 1./2. * B_ddr + 1./2. * Lambda * A * B_dr ) \
            + r * ( Lambda * B - B_dr / B - 6. * psi_dr / psi * B / A ) + B / A - 1.

    @jit(nopython=True)
    def calc_rhs_D_TF_rr(D_rr, D_tt, r, A, B):
        """ calculate D^TF_rr from D_rr, D_tt, A, B, r (Eq. 15)
        """
        return 2./3. * ( D_rr -  A * D_tt / (B * r**2))

    @jit(nopython=True)
    def calc_rhs_D_TF_tt(D_tt, A, B):
        """ calculate D^TF_theta,theta from D_tt, A, B (Eq. 16)
        """
        return 1./3. * ( D_tt - B * D_tt / A )

    @jit(nopython=True)
    def calc_rhs_R_TF_rr(R_rr, R_tt, r, A, B):
        """ calculate R^TF_rr from R_rr, R_tt, A, B, r (Eq. 15)
        """
        return 2./3. * ( R_rr -  A * R_tt / (B * r**2))

    @jit(nopython=True)
    def calc_rhs_R_TF_tt(R_tt, A, B):
        """ calculate R^TF_theta,theta from R_tt, A, B (Eq. 16)
        """
        return 1./3. * ( R_tt - B * R_tt / A )

    @jit(nopython=True)
    def calc_rhs_D(psi, A, B, D_rr, D_tt, r):
        """ calculate D from psi, D_rr, D_tt, A, B, r (Eq. 10)
        """
        return 1. / psi**4 * ( D_rr / A + 2. * D_tt / (r**2 * B) )
    
    @jit(nopython=True)
    def calc_rhs_dt_alpha(alpha, K):
        """ Eq. 2
        """
        return - alpha**2 * K

    @jit(nopython=True)
    def calc_rhs_dt_A(alpha, A_t_rr):
        """ Eq. 3
        """
        return -2. * alpha * A_t_rr

    @jit(nopython=True)
    def calc_rhs_dt_B(alpha, A_t_tt):
        """ Eq. 4
        """
        return -2. * alpha * A_t_tt

    @jit(nopython=True)
    def calc_rhs_dt_psi(alpha, psi, K):
        """ Eq. 5
        """
        return - 1./6. * alpha * psi * K
    
    @jit(nopython=True)
    def calc_rhs_dt_Lambda(alpha, A, A_t_tt, psi, psi_dr, K_dr, A_t_rr, A_dr, B, B_dr, r, alpha_dr):
        """ Eq. 6
        """
        return 2. * alpha / A * ( 6. * A_t_tt / A * psi_dr / psi - 2./3. * K_dr ) \
            + alpha / A * (A_t_rr * A_dr / A**2 - 2. * A_t_tt * B_dr / B**2 + 4. * A_t_tt * (A - B) / (r * B**2)) \
            - 2. * A_t_rr * alpha_dr / A**2

    @jit(nopython=True)
    def calc_rhs_dt_A_t_rr(psi, D_TF_rr, alpha, R_TF_rr, A_t_rr, K, A):
        """ Eq. 7
        """
        return 1. / psi**4 * ( - D_TF_rr + alpha * R_TF_rr ) \
            + alpha * ( A_t_rr * K - 2. * A_t_rr**2 / A )

    @jit(nopython=True)
    def calc_rhs_dt_A_t_tt(r, psi, D_TF_tt, alpha, R_TF_tt, A_t_tt, K, B):
        """ Eq. 8
        """
        return 1. / (r**2 * psi**4) * ( -D_TF_tt + alpha * R_TF_tt ) \
            + alpha * ( A_t_tt * K - 2. * A_t_tt**2 / B )

    @jit(nopython=True)
    def calc_rhs_dt_K(alpha, K, A_t_rr, A, A_t_tt, B, D):
        """ Eq. 9
        """
        return alpha * ( 1./3. * K**2 + A_t_rr**2 / A**2 + 2. * A_t_tt**2 / B**2 ) - D
    
    ################################################################
    ## get modes
    @jit(nopython=True)
    def get_modes_alpha(SBN, alpha):
        """ Eq. 42
        """
        return np.linalg.solve(SBN, alpha - 1)

    @jit(nopython=True)
    def get_modes_chi(SBN, chi):
        """ Eq. 43
        """
        return np.linalg.solve(SBN, chi)
    
    @jit(nopython=True)
    def get_modes_K(SBN, K):
        """ Eq. 44
        """
        return np.linalg.solve(SBN, K)

    @jit(nopython=True)
    def get_modes_A(SBN, A):
        """ Eq. 45
        """
        return np.linalg.solve(1./2. * SBN, A[:-1] - 1)
    
    @jit(nopython=True)
    def get_modes_B(SBN, B):
        """ Eq. 46
        """
        return np.linalg.solve(1./2. * SBN, B[:-1] - 1)
    
    @jit(nopython=True)
    def get_modes_A_a(SBN, A_a):
        """ Eq. 47
        """
        return np.linalg.solve(1./2. * SBN, A_a[:-1])
    
    @jit(nopython=True)
    def get_modes_Lambda(SBN, Lambda):
        """ Eq. 48
        """
        return np.linalg.solve(SBN, Lambda[:-1])
    
    ################################################################
    ## derivative terms
    @jit(nopython=True)
    def deriv_r_alpha(SBN_dr, modes):
        """ d/dr alpha
        """
        return SBN_dr@modes
    
    @jit(nopython=True)
    def deriv_r_chi(SBN_dr, modes):
        """ d/dr chi
        """
        return SBN_dr@modes

    @jit(nopython=True)
    def deriv_r_psi(psi, chi_dr):
        """ d/dr psi
        needs Info of psi and already d/dr chi
        """
        # d/dr psi = psi * d/dr chi
        return psi * chi_dr

    @jit(nopython=True)
    def deriv_r_K(SBN_dr, modes):
        """ d/dr K
        """
        return SBN_dr@modes
    
    @jit(nopython=True)
    def deriv_r_A(SBN_dr, modes):
        """ d/dr A
        """
        return (1./2. * SBN_dr)@modes
    
    @jit(nopython=True)
    def deriv_r_B(SBN_dr, modes):
        """ d/dr B
        """
        return (1./2. * SBN_dr)@modes
    
    @jit(nopython=True)
    def deriv_r_A_a(SBN_dr, modes):
        """ d/dr A_a
        """
        return (1./2. * SBN_dr)@modes

    @jit(nopython=True)
    def deriv_r_Lambda(SBN_dr, modes):
        """ d/dr Lambda
        """
        return SBN_dr@modes

    @jit(nopython=True)
    def deriv_rr_alpha(SBN_ddr, modes):
        """ d^2/dr^2 alpha
        """
        return SBN_ddr@modes
    
    @jit(nopython=True)
    def deriv_rr_chi(SBN_ddr, modes):
        """ d^2/dr^2 chi
        """
        return SBN_ddr@modes

    @jit(nopython=True)
    def deriv_rr_psi(psi, chi_dr, chi_ddr):
        """ d^2/dr^2 psi
        Needs psi, d/dr chi and d^2/dr^2 chi
        """
        # d^2/dr^2 psi = psi ( d^2/dr^2 chi + (d/dr chi)^2 )
        return psi*( chi_ddr + chi_dr**2 )
    
    @jit(nopython=True)
    def deriv_rr_A(SBN_ddr, modes):
        """ d^2/dr^2 A
        """
        return (1./2. * SBN_ddr)@modes
    
    @jit(nopython=True)
    def deriv_rr_B(SBN_ddr, modes):
        """ d^2/dr^2 B
        """
        return (1./2. * SBN_ddr)@modes


############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################


# class evolution:
#     def __init__(self, ev_p, spec):
#         """ We have to provide evolution params

#         Furthermore provide a spectral decomposition

#         All equations from arXiv:2105.09094v2 
#         """        
#         # spectral decomposition data
#         self.sd = spec
#         self.r = self.sd.r_k

#         ## initial data
#         self.alpha = ev_p.alpha
#         self.K = ev_p.K
#         self.A = ev_p.A
#         self.B = ev_p.B
#         self.A_a = ev_p.A_a
#         self.chi = ev_p.chi
#         self.Lambda = ev_p.Lambda

#         # fix boundary conditions at r = 0
#         self.A[-1] = 1.
#         self.B[-1] = 1.
#         self.A_a[-1] = 0.

#         ## from notes: step 2 to 5
#         # self.Lambda = np.zeros(len(self.alpha)) # has to be done in calc_derivatives().. unfortunately
#         self.psi = self.calc_rhs_psi()

#         self.A_t_rr = self.A_a * self.A # tilde(A)_rr
#         self.A_t_tt = - 1./2. * self.A_a * self.B # tilde(A)_theta,theta

#         ## for the rest we need the derivative terms of alpha, psi, K, A, B, and Lambda
#         # boundary conditions fixed in calc_derivatives()
#         N = len(self.alpha) # N+1 in our previous nomenclature
#         self.alpha_dr = np.zeros(N)
#         self.chi_dr = np.zeros(N)
#         self.psi_dr = np.zeros(N)
#         self.K_dr = np.zeros(N)
#         self.A_dr = np.zeros(N)
#         self.B_dr = np.zeros(N)
#         self.Lambda_dr = np.zeros(N)
#         self.alpha_ddr = np.zeros(N)
#         self.chi_ddr = np.zeros(N)
#         self.psi_ddr = np.zeros(N)
#         self.A_ddr = np.zeros(N)
#         self.B_ddr = np.zeros(N)
#         self.calc_derivatives()

#         ## now compute D's and R's
#         # step 6
#         self.D_rr = self.calc_rhs_D_rr() # cal D_rr
#         self.D_tt = self.calc_rhs_D_tt() # cal D_theta,theta
#         self.R_rr = self.calc_rhs_R_rr() # R_rr
#         self.R_tt = self.calc_rhs_R_tt() # R_theta,theta

#         # step 7
#         self.D_TF_rr = self.calc_rhs_D_TF_rr() # cal D^TF_rr
#         self.D_TF_tt = self.calc_rhs_D_TF_tt() # cal D^TF_theta,theta
#         self.R_TF_rr = self.calc_rhs_R_TF_rr() # R^TF_rr 
#         self.R_TF_tt = self.calc_rhs_R_TF_tt() # R^TF_theta,theta

#         #step 8
#         self.D = self.calc_rhs_D() # cal D

#         # fix boundary conditions at r=0
#         self.D_rr[-1] = self.alpha_ddr[-1]
#         self.D_tt[-1] = 0
#         self.R_rr[-1] = 4. * self.psi_ddr[-1] # ? assumes dr^2 A = 0 at r=0
#         self.R_tt[-1] = 0
#         self.D_TF_rr[-1] = 2./3. * self.alpha_ddr[-1]
#         self.D_TF_tt[-1] = 0
#         self.R_TF_rr[-1] = 8./3. * self.psi_ddr[-1]
#         self.R_TF_tt[-1] = 0
#         self.D[-1] = self.alpha_ddr[-1] / self.psi[-1]**4

#     def calc_derivatives(self):
#         """ calculate all derivative terms
#         """
#         # get modes
#         modes_alpha = self.get_modes_alpha()
#         modes_chi = self.get_modes_chi()
#         modes_K = self.get_modes_K()

#         # following modes are only N long, not N+1
#         modes_A = self.get_modes_A()
#         modes_B = self.get_modes_B()

#         # calculate derivative terms from modes
#         self.alpha_dr = self.deriv_r_alpha(modes_alpha)
#         self.chi_dr = self.deriv_r_chi(modes_chi)
#         self.K_dr = self.deriv_r_K(modes_K)
#         self.K_dr[-1] = 0 # BC
#         self.A_dr[:-1] = self.deriv_r_A(modes_A) # d/dr A(r_(N+1)) = 0
#         self.B_dr[:-1] = self.deriv_r_B(modes_B) # d/dr B(r_(N+1)) = 0
#         self.alpha_ddr = self.deriv_rr_alpha(modes_alpha)
#         self.chi_ddr = self.deriv_rr_chi(modes_chi)
#         self.A_ddr[:-1] = self.deriv_rr_A(modes_A) # d^2/dr^2 A(r_(N+1)) = 0 ??
#         self.B_ddr[:-1] = self.deriv_rr_B(modes_B) # d^2/dr^2 B(r_(N+1)) = 0 ??
        
#         self.chi_dr[-1] = 0

#         # self.Lambda = self.calc_rhs_Lambda()
#         modes_Lambda = self.get_modes_Lambda()
#         self.Lambda_dr[:-1] = self.deriv_r_Lambda(modes_Lambda) # d/dr Lambda(r_(N+1)) = 0

#         # derivative of psi was defined with psi and derivative of chi
#         self.psi_dr = self.deriv_r_psi()
#         self.psi_ddr = self.deriv_rr_psi()

#         # boundary condition r = 0
#         self.psi_dr[-1] = 0

#     def calc_rhs(self):
#         """ return d/dt alpha, ... for all ev_param's
#         """
#         dt_ev_p = ev_param(None, None, None, None, None, None, None)

#         dt_ev_p.alpha = self.calc_rhs_dt_alpha()
#         dt_ev_p.A = self.calc_rhs_dt_A()
#         dt_ev_p.B = self.calc_rhs_dt_B()
#         dt_ev_p.K = self.calc_rhs_dt_K()
#         dt_ev_p.Lambda = self.calc_rhs_dt_Lambda()
        
#         psi_dt = self.calc_rhs_dt_psi()
#         # d/dt chi = 1 / psi * d/dt psi
#         dt_ev_p.chi = 1. / self.psi * psi_dt

#         A_t_rr_dt = self.calc_rhs_dt_A_t_rr()

#         dt_ev_p.A_a = ( A_t_rr_dt * self.A - self.A_t_rr * dt_ev_p.A ) / self.A**2

#         return dt_ev_p

#     def solve_hamiltonian_constraint(self):
#         """ solve hamiltonian constraint (Eq. 17) for psi
#         """
#         pass

#     def solve_momentum_constraint(self):
#         """ solve momentum constraint (Eq. 18) for K
#         """
#         pass

#     ################################################################
#     ## calculate RHs
#     def calc_rhs_Lambda(self):
#         """ calculate Lambda hat from A, B, quad points r (Eq. 20)
#         """
#         ## Eq. 20 as constraint but also Eq. 6 as evolution?
        
#         A = self.A
#         A_dr = self.A_dr
#         B = self.B
#         B_dr = self.B_dr
#         r = self.r

#         return 1. / A * ( A_dr / (2. * A) - B_dr / B - 2. / r * (1 - A / B) )

#     def calc_rhs_psi(self):
#         """ calculate psi from chi (Eq. 19)
#         """
#         return np.exp(self.chi)

#     def calc_rhs_chi(self):
#         """ calculate chi from psi (Eq. 19)
#         """
#         return np.log(self.psi)
    
#     def calc_rhs_D_rr(self):
#         """ calculate D_rr from alpha, A, psi (Eq. 11)
#         """
#         alpha_dr = self.alpha_dr
#         alpha_ddr = self.alpha_ddr
#         A = self.A
#         A_dr = self.A_dr
#         psi = self.psi
#         psi_dr = self.psi_dr
        
#         return alpha_ddr - alpha_dr / 2. * (A_dr / A + 4. * psi_dr / psi)

#     def calc_rhs_D_tt(self):
#         """ calculate D_theta,theta from alpha, A, B, psi, r (Eq. 12)
#         """
#         r = self.r
#         alpha_dr = self.alpha_dr
#         A = self.A
#         B = self.B
#         B_dr = self.B_dr
#         psi = self.psi
#         psi_dr = self.psi_dr

#         return  r * alpha_dr / A * ( B + r / 2. * (B_dr + 4. * B * psi_dr / psi) )

#     def calc_rhs_R_rr(self):
#         """ calculate R_rr from A, B, Lambda hat, psi, r (Eq. 13)
#         """
#         A = self.A
#         A_dr = self.A_dr
#         A_ddr = self.A_ddr
#         B = self.B
#         B_dr = self.B_dr
#         Lambda = self.Lambda
#         Lambda_dr = self.Lambda_dr
#         r = self.r
#         psi = self.psi
#         psi_dr = self.psi_dr
#         psi_ddr = self.psi_ddr

#         return  3./4. * A_dr**2 / A**2 - B_dr**2 / (2. * B**2) + A * Lambda_dr + 1./2. * Lambda * A_dr \
#             + 1./r * ( -4.*psi_dr / psi -  1. / B * (A_dr + 2. * B_dr) + 2. * A * B_dr / B**2) \
#             - 4. * (psi_ddr * psi - psi_dr**2) / psi**2 + 2. * psi_dr / psi * ( A_dr / A - B_dr / B ) \
#             - A_ddr / (2. * A) + 2. * (A - B) / (r**2 * B)

#     def calc_rhs_R_tt(self):
#         """ calculate R_theta,theta from A, B, Lambda hat, psi, r (Eq. 14)
#         """
#         A = self.A
#         A_dr = self.A_dr
#         B = self.B
#         B_dr = self.B_dr
#         B_ddr = self.B_ddr
#         Lambda = self.Lambda
#         r = self.r
#         psi = self.psi
#         psi_dr = self.psi_dr
#         psi_ddr = self.psi_ddr

#         return r**2 * B / A * ( psi_dr / psi * A_dr / A - 2 * (psi_ddr * psi - psi_dr ** 2) / psi**2 - 4. * (psi_dr / psi)**2 ) \
#             + r**2 / A * ( B_dr**2 / (2. * B) - 3. * psi_dr / psi * B_dr - 1./2. * B_ddr + 1./2. * Lambda * A * B_dr ) \
#             + r * ( Lambda * B - B_dr / B - 6. * psi_dr / psi * B / A ) + B / A - 1.

#     def calc_rhs_D_TF_rr(self):
#         """ calculate D^TF_rr from D_rr, D_tt, A, B, r (Eq. 15)
#         """
#         D_rr = self.D_rr
#         D_tt = self.D_tt
#         r = self.r
#         A = self.A
#         B = self.B

#         return 2./3. * ( D_rr -  A * D_tt / (B * r**2))

#     def calc_rhs_D_TF_tt(self):
#         """ calculate D^TF_theta,theta from D_tt, A, B (Eq. 16)
#         """
#         D_tt = self.D_tt
#         A = self.A
#         B = self.B

#         return 1./3. * ( D_tt - B * D_tt / A )

#     def calc_rhs_R_TF_rr(self):
#         """ calculate R^TF_rr from R_rr, R_tt, A, B, r (Eq. 15)
#         """
#         R_rr = self.R_rr
#         R_tt = self.R_tt
#         r = self.r
#         A = self.A
#         B = self.B

#         return 2./3. * ( R_rr -  A * R_tt / (B * r**2))

#     def calc_rhs_R_TF_tt(self):
#         """ calculate R^TF_theta,theta from R_tt, A, B (Eq. 16)
#         """
#         R_tt = self.R_tt
#         A = self.A
#         B = self.B

#         return 1./3. * ( R_tt - B * R_tt / A )

#     def calc_rhs_D(self):
#         """ calculate D from psi, D_rr, D_tt, A, B, r (Eq. 10)
#         """
#         psi = self.psi
#         A = self.A
#         B = self.B
#         D_rr = self.D_rr
#         D_tt = self.D_tt
#         r = self.r

#         return 1. / psi**4 * ( D_rr / A + 2. * D_tt / (r**2 * B) )
    
#     def calc_rhs_dt_alpha(self):
#         """ Eq. 2
#         """
#         alpha = self.alpha
#         K = self.K

#         return - alpha**2 * K

#     def calc_rhs_dt_A(self):
#         """ Eq. 3
#         """
#         alpha = self.alpha
#         A_t_rr = self.A_t_rr

#         return -2. * alpha * A_t_rr

#     def calc_rhs_dt_B(self):
#         """ Eq. 4
#         """
#         alpha = self.alpha
#         A_t_tt = self.A_t_tt

#         return -2. * alpha * A_t_tt

#     def calc_rhs_dt_psi(self):
#         """ Eq. 5
#         """
#         alpha = self.alpha
#         psi = self.psi
#         K = self.K

#         return - 1./6. * alpha * psi * K
    
#     def calc_rhs_dt_Lambda(self):
#         """ Eq. 6
#         """
#         alpha = self.alpha
#         A = self.A
#         A_t_tt = self.A_t_tt
#         psi = self.psi
#         psi_dr = self.psi_dr
#         K_dr = self.K_dr
#         A_t_rr = self.A_t_rr
#         A_dr = self.A_dr
#         B = self.B
#         B_dr = self.B_dr
#         r = self.r
#         alpha_dr = self.alpha_dr

#         return 2. * alpha / A * ( 6. * A_t_tt / A * psi_dr / psi - 2./3. * K_dr ) \
#             + alpha / A * (A_t_rr * A_dr / A**2 - 2. * A_t_tt * B_dr / B**2 + 4. * A_t_tt * (A - B) / (r * B**2)) \
#             - 2. * A_t_rr * alpha_dr / A**2

#     def calc_rhs_dt_A_t_rr(self):
#         """ Eq. 7
#         """
#         psi = self.psi
#         D_TF_rr = self.D_TF_rr
#         alpha = self.alpha
#         R_TF_rr = self.R_TF_rr
#         A_t_rr = self.A_t_rr
#         K = self.K
#         A = self.A

#         return 1. / psi**4 * ( - D_TF_rr + alpha * R_TF_rr ) \
#             + alpha * ( A_t_rr * K - 2. * A_t_rr**2 / A )

#     def calc_rhs_dt_A_t_tt(self):
#         """ Eq. 8
#         """
#         r = self.r
#         psi = self.psi
#         D_TF_tt = self.D_TF_tt
#         alpha = self.alpha
#         R_TF_tt = self.R_TF_tt
#         A_t_tt = self.A_t_tt
#         K = self.K
#         B = self.B

#         return 1. / (r**2 * psi**4) * ( -D_TF_tt + alpha * R_TF_tt ) \
#             + alpha * ( A_t_tt * K - 2. * A_t_tt**2 / B )

#     def calc_rhs_dt_K(self):
#         """ Eq. 9
#         """
#         alpha = self.alpha
#         K = self.K
#         A_t_rr = self.A_t_rr
#         A = self.A
#         A_t_tt = self.A_t_tt
#         B = self.B
#         D = self.D

#         return alpha * ( 1./3. * K**2 + A_t_rr**2 / A**2 + 2. * A_t_tt**2 / B**2 ) - D
    
#     ################################################################
#     ## get modes
#     def get_modes_alpha(self):
#         """ Eq. 42
#         """
#         AL = self.sd.SBN[::2, :].transpose()

#         return np.linalg.solve(AL, self.alpha - 1)

#     def get_modes_chi(self):
#         """ Eq. 43
#         """
#         AL = self.sd.SBN[::2, :].transpose()

#         return np.linalg.solve(AL, self.chi)
    
#     def get_modes_K(self):
#         """ Eq. 44
#         """
#         AL = self.sd.SBN[::2, :].transpose()

#         return np.linalg.solve(AL, self.K)

#     def get_modes_A(self):
#         """ Eq. 45
#         """
#         AL = 1. / 2. * (self.sd.SBN[2::2, :-1] - self.sd.SBN[0:-2:2, :-1]) # do not include r_(N+1)
#         AL = AL.transpose()

#         return np.linalg.solve(AL, self.A[:-1] - 1)
    
#     def get_modes_B(self):
#         """ Eq. 46
#         """
#         AL = 1. / 2. * (self.sd.SBN[2::2, :-1] - self.sd.SBN[0:-2:2, :-1]) # do not include r_(N+1)
#         AL = AL.transpose()

#         return np.linalg.solve(AL, self.B[:-1] - 1)
    
#     def get_modes_A_a(self):
#         """ Eq. 47
#         """
#         AL = 1. / 2. * (self.sd.SBN[2::2, :-1] - self.sd.SBN[0:-2:2, :-1]) # do not include r_(N+1)
#         AL = AL.transpose()

#         return np.linalg.solve(AL, self.A_a[:-1])
    
#     def get_modes_Lambda(self):
#         """ Eq. 48
#         """
#         AL = self.sd.SBN[1::2, :-1] # do not include r_(N+1)
#         AL = AL.transpose()

#         return np.linalg.solve(AL, self.Lambda[:-1])
    
#     ################################################################
#     ## derivative terms
#     def deriv_r_alpha(self, modes):
#         """ d/dr alpha
#         """
#         DAL = self.sd.SBN_dr[::2].transpose()

#         return DAL@modes
    
#     def deriv_r_chi(self, modes):
#         """ d/dr chi
#         """
#         DAL = self.sd.SBN_dr[::2].transpose()

#         return DAL@modes

#     def deriv_r_psi(self):
#         """ d/dr psi
#         needs Info of psi and already d/dr chi
#         """
#         # d/dr psi = psi * d/dr chi
#         return self.psi * self.chi_dr

#     def deriv_r_K(self, modes):
#         """ d/dr K
#         """
#         DAL = self.sd.SBN_dr[::2].transpose()

#         return DAL@modes
    
#     def deriv_r_A(self, modes):
#         """ d/dr A
#         """
#         DAL = 1./2. * (self.sd.SBN_dr[2::2, :-1] - self.sd.SBN_dr[0:-2:2, :-1]) # do not include r_N+1
#         DAL = DAL.transpose()

#         return DAL@modes
    
#     def deriv_r_B(self, modes):
#         """ d/dr B
#         """
#         DAL = 1./2. * (self.sd.SBN_dr[2::2, :-1] - self.sd.SBN_dr[0:-2:2, :-1]) # do not include r_N+1
#         DAL = DAL.transpose()

#         return DAL@modes
    
#     def deriv_r_Lambda(self, modes):
#         """ d/dr Lambda
#         """
#         DAL = self.sd.SBN_dr[1::2, :-1].transpose() # do not include r_N+1

#         return DAL@modes

#     def deriv_rr_alpha(self, modes):
#         """ d^2/dr^2 alpha
#         """
#         DAL = self.sd.SBN_ddr[::2].transpose()

#         return DAL@modes
    
#     def deriv_rr_chi(self, modes):
#         """ d^2/dr^2 chi
#         """
#         DAL = self.sd.SBN_ddr[::2].transpose()

#         return DAL@modes

#     def deriv_rr_psi(self):
#         """ d^2/dr^2 psi
#         Needs psi, d/dr chi and d^2/dr^2 chi
#         """
#         # d^2/dr^2 psi = psi ( d^2/dr^2 chi + (d/dr chi)^2 )
#         return self.psi*( self.chi_ddr + self.chi_dr**2 )
    
#     def deriv_rr_A(self, modes):
#         """ d^2/dr^2 A
#         """
#         DDAL = 1./2. * (self.sd.SBN_ddr[2::2, :-1] - self.sd.SBN_ddr[0:-2:2, :-1]) # do not include r_N+1
#         DDAL = DDAL.transpose()

#         return DDAL@modes
    
#     def deriv_rr_B(self, modes):
#         """ d^2/dr^2 B
#         """
#         DDAL = 1./2. * (self.sd.SBN_ddr[2::2, :-1] - self.sd.SBN_ddr[0:-2:2, :-1]) # do not include r_N+1
#         DDAL = DDAL.transpose()

#         return DDAL@modes