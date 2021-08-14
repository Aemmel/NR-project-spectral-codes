"""
Encode the evolution equations Eq.2 to Eq.9
"""

from spec_decomposition import spec_decom
import numpy as np

class ev_param:
    """ parameters that are carried over to the next timestep
    """
    def __init__(self, alpha, K, A, B, A_a, chi):
        self.alpha = alpha
        self.K = K
        self.A = A
        self.B = B
        self.A_a = A_a
        self.chi = chi

    def __add__(self, o):
        alpha = self.alpha + o.alpha
        K = self.K + o.K
        A = self.A + o.A
        B = self.B + o.B
        A_a = self.A_a + o.A_a
        chi = self.chi + o.chi

        return ev_param(alpha, K , A, B, A_a, chi)

    def __rmul__(self, f):
        alpha = self.alpha * f
        K = self.K * f
        A = self.A * f
        B = self.B * f
        A_a = self.A_a * f
        chi = self.chi * f

        return ev_param(alpha, K, A, B, A_a, chi)
    
    def __str__(self):
        out = "alpha=" + str(self.alpha) + "\nK=" + str(self.K) + "\nA=" + str(self.A) + "\nB=" + str(self.B) + "\nA_a=" + str(self.A_a) + "\nchi=" + str(self.chi)

        return out

    
class evolution:
    def __init__(self, ev_p, spec):
        """ We have to provide evolution params

        Furthermore provide a spectral decomposition
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

        # fix boundary conditions at r = 0
        self.A[-1] = 1.
        self.B[-1] = 1.
        self.A_a[-1] = 0.

        ## from notes: step 2 to 5
        self.Lambda = np.zeros(len(self.alpha)) # has to be done in calc_derivatives().. unfortunately
        self.psi = self.calc_rhs_psi()

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
        self.D_rr = self.calc_rhs_D_rr() # cal D_rr
        self.D_tt = self.calc_rhs_D_tt() # cal D_theta,theta
        self.R_rr = self.calc_rhs_R_rr() # R_rr
        self.R_tt = self.calc_rhs_R_tt() # R_theta,theta

        # step 7
        self.D_TF_rr = self.calc_rhs_D_TF_rr() # cal D^TF_rr
        self.D_TF_tt = self.calc_rhs_D_TF_tt() # cal D^TF_theta,theta
        self.R_TF_rr = self.calc_rhs_R_TF_rr() # R^TF_rr 
        self.R_TF_tt = self.calc_rhs_R_TF_tt() # R^TF_theta,theta

        #step 8
        self.D = self.calc_rhs_D() # cal D

        # fix boundary conditions at r=0
        self.D_rr[-1] = self.alpha_ddr[-1]
        self.D_tt[-1] = 0
        self.R_rr[-1] = 4. * self.psi_ddr[-1] # ? assumes dr^2 A = 0 at r=0
        self.R_tt[-1] = 0
        self.D_TF_rr[-1] = 2./3. * self.alpha_ddr[-1]
        self.D_TF_tt[-1] = 0
        self.R_TF_rr[-1] = 8./3. * self.psi_ddr[-1]
        self.R_TF_tt[-1] = 0
        self.D[-1] = self.alpha_ddr[-1] / self.psi[-1]**4

    def calc_derivatives(self):
        """ calculate all derivative terms
        """
        # get modes
        modes_alpha = self.get_modes_alpha()
        modes_chi = self.get_modes_chi()
        modes_K = self.get_modes_K()

        # following modes are only N long, not N+1
        modes_A = self.get_modes_A()
        modes_B = self.get_modes_B()

        # calculate derivative terms from modes
        self.alpha_dr = self.deriv_r_alpha(modes_alpha)
        self.chi_dr = self.deriv_r_chi(modes_chi)
        self.K_dr = self.deriv_r_K(modes_K)
        self.K_dr[-1] = 0 # BC
        self.A_dr[:-1] = self.deriv_r_A(modes_A) # d/dr A(r_(N+1)) = 0
        self.B_dr[:-1] = self.deriv_r_B(modes_B) # d/dr B(r_(N+1)) = 0
        self.alpha_ddr = self.deriv_rr_alpha(modes_alpha)
        self.chi_ddr = self.deriv_rr_chi(modes_chi)
        self.A_ddr[:-1] = self.deriv_rr_A(modes_A) # d^2/dr^2 A(r_(N+1)) = 0 ??
        self.B_ddr[:-1] = self.deriv_rr_B(modes_B) # d^2/dr^2 B(r_(N+1)) = 0 ??
        
        self.chi_dr[-1] = 0

        self.Lambda = self.calc_rhs_Lambda()
        modes_Lambda = self.get_modes_Lambda()
        self.Lambda_dr[:-1] = self.deriv_r_Lambda(modes_Lambda) # d/dr Lambda(r_(N+1)) = 0

        # derivative of psi was defined with psi and derivative of chi
        self.psi_dr = self.deriv_r_psi()
        self.psi_ddr = self.deriv_rr_psi()

        # boundary condition r = 0
        self.psi_dr[-1] = 0

    def calc_rhs(self):
        """ return d/dt alpha, ... for all ev_param's
        """
        dt_ev_p = ev_param(None, None, None, None, None, None)

        dt_ev_p.alpha = self.calc_rhs_dt_alpha()
        dt_ev_p.A = self.calc_rhs_dt_A()
        dt_ev_p.B = self.calc_rhs_dt_B()
        dt_ev_p.K = self.calc_rhs_dt_K()
        
        psi_dt = self.calc_rhs_dt_psi()
        # d/dt chi = 1 / psi * d/dt psi
        dt_ev_p.chi = 1. / self.psi * psi_dt

        A_t_rr_dt = self.calc_rhs_dt_A_t_rr()

        dt_ev_p.A_a = ( A_t_rr_dt * self.A - self.A_t_rr * dt_ev_p.A ) / self.A**2

        return dt_ev_p

    def solve_hamiltonian_constraint(self):
        """ solve hamiltonian constraint (Eq. 17) for psi
        """
        pass

    def solve_momentum_constraint(self):
        """ solve momentum constraint (Eq. 18) for K
        """
        pass

    ################################################################
    ## calculate RHs
    def calc_rhs_Lambda(self):
        """ calculate Lambda hat from A, B, quad points r (Eq. 20)
        """
        ## Eq. 20 as constraint but also Eq. 6 as evolution?
        
        A = self.A
        A_dr = self.A_dr
        B = self.B
        B_dr = self.B_dr
        r = self.r

        return 1. / A * ( A_dr / (2. * A) - B_dr / B - 2. / r * (1 - A / B) )

    def calc_rhs_psi(self):
        """ calculate psi from chi (Eq. 19)
        """
        return np.exp(self.chi)

    def calc_rhs_chi(self):
        """ calculate chi from psi (Eq. 19)
        """
        return np.log(self.psi)
    
    def calc_rhs_D_rr(self):
        """ calculate D_rr from alpha, A, psi (Eq. 11)
        """
        alpha_dr = self.alpha_dr
        alpha_ddr = self.alpha_ddr
        A = self.A
        A_dr = self.A_dr
        psi = self.psi
        psi_dr = self.psi_dr
        
        return alpha_ddr - alpha_dr / 2. * (A_dr / A + 4. * psi_dr / psi)

    def calc_rhs_D_tt(self):
        """ calculate D_theta,theta from alpha, A, B, psi, r (Eq. 12)
        """
        r = self.r
        alpha_dr = self.alpha_dr
        A = self.A
        B = self.B
        B_dr = self.B_dr
        psi = self.psi
        psi_dr = self.psi_dr

        return  r * alpha_dr / A * ( B + r / 2. * (B_dr + 4. * B * psi_dr / psi) )

    def calc_rhs_R_rr(self):
        """ calculate R_rr from A, B, Lambda hat, psi, r (Eq. 13)
        """
        A = self.A
        A_dr = self.A_dr
        A_ddr = self.A_ddr
        B = self.B
        B_dr = self.B_dr
        Lambda = self.Lambda
        Lambda_dr = self.Lambda_dr
        r = self.r
        psi = self.psi
        psi_dr = self.psi_dr
        psi_ddr = self.psi_ddr

        return  3./4. * A_dr**2 / A**2 - B_dr**2 / (2. * B**2) + A * Lambda_dr + 1./2. * Lambda * A_dr \
            + 1./r * ( -4.*psi_dr / psi -  1. / B * (A_dr + 2. * B_dr) + 2. * A * B_dr / B**2) \
            - 4. * (psi_ddr * psi - psi_dr**2) / psi**2 + 2. * psi_dr / psi * ( A_dr / A - B_dr / B ) \
            - A_ddr / (2. * A) + 2. * (A - B) / (r**2 * B)

    def calc_rhs_R_tt(self):
        """ calculate R_theta,theta from A, B, Lambda hat, psi, r (Eq. 14)
        """
        A = self.A
        A_dr = self.A_dr
        B = self.B
        B_dr = self.B_dr
        B_ddr = self.B_ddr
        Lambda = self.Lambda
        r = self.r
        psi = self.psi
        psi_dr = self.psi_dr
        psi_ddr = self.psi_ddr

        return r**2 * B / A * ( psi_dr / psi * A_dr / A - 2 * (psi_ddr * psi - psi_dr ** 2) / psi**2 - 4. * (psi_dr / psi)**2 ) \
            + r**2 / A * ( B_dr**2 / (2. * B) - 3. * psi_dr / psi * B_dr - 1./2. * B_ddr + 1./2. * Lambda * A * B_dr ) \
            + r * ( Lambda * B - B_dr / B - 6. * psi_dr / psi * B / A ) + B / A - 1.

    def calc_rhs_D_TF_rr(self):
        """ calculate D^TF_rr from D_rr, D_tt, A, B, r (Eq. 15)
        """
        D_rr = self.D_rr
        D_tt = self.D_tt
        r = self.r
        A = self.A
        B = self.B

        return 2./3. * ( D_rr -  A * D_tt / (B * r**2))

    def calc_rhs_D_TF_tt(self):
        """ calculate D^TF_theta,theta from D_tt, A, B (Eq. 16)
        """
        D_tt = self.D_tt
        A = self.A
        B = self.B

        return 1./3. * ( D_tt - B * D_tt / A )

    def calc_rhs_R_TF_rr(self):
        """ calculate R^TF_rr from R_rr, R_tt, A, B, r (Eq. 15)
        """
        R_rr = self.R_rr
        R_tt = self.R_tt
        r = self.r
        A = self.A
        B = self.B

        return 2./3. * ( R_rr -  A * R_tt / (B * r**2))

    def calc_rhs_R_TF_tt(self):
        """ calculate R^TF_theta,theta from R_tt, A, B (Eq. 16)
        """
        R_tt = self.R_tt
        A = self.A
        B = self.B

        return 1./3. * ( R_tt - B * R_tt / A )

    def calc_rhs_D(self):
        """ calculate D from psi, D_rr, D_tt, A, B, r (Eq. 10)
        """
        psi = self.psi
        A = self.A
        B = self.B
        D_rr = self.D_rr
        D_tt = self.D_tt
        r = self.r

        return 1. / psi**4 * ( D_rr / A + 2. * D_tt / (r**2 * B) )
    
    def calc_rhs_dt_alpha(self):
        """ Eq. 2
        """
        alpha = self.alpha
        K = self.K

        return - alpha**2 * K

    def calc_rhs_dt_A(self):
        """ Eq. 3
        """
        alpha = self.alpha
        A_t_rr = self.A_t_rr

        return -2. * alpha * A_t_rr

    def calc_rhs_dt_B(self):
        """ Eq. 4
        """
        alpha = self.alpha
        A_t_tt = self.A_t_tt

        return -2. * alpha * A_t_tt

    def calc_rhs_dt_psi(self):
        """ Eq. 5
        """
        alpha = self.alpha
        psi = self.psi
        K = self.K

        return - 1./6. * alpha * psi * K

    def calc_rhs_dt_A_t_rr(self):
        """ Eq. 7
        """
        psi = self.psi
        D_TF_rr = self.D_TF_rr
        alpha = self.alpha
        R_TF_rr = self.R_TF_rr
        A_t_rr = self.A_t_rr
        K = self.K
        A = self.A

        return 1. / psi**4 * ( - D_TF_rr + alpha * R_TF_rr ) \
            + alpha * ( A_t_rr * K - 2. * A_t_rr**2 / A )

    def calc_rhs_dt_A_t_tt(self):
        """ Eq. 8
        """
        r = self.r
        psi = self.psi
        D_TF_tt = self.D_TF_tt
        alpha = self.alpha
        R_TF_tt = self.R_TF_tt
        A_t_tt = self.A_t_tt
        K = self.K
        B = self.B

        return 1. / (r**2 * psi**4) * ( -D_TF_tt + alpha * R_TF_tt ) \
            + alpha * ( A_t_tt * K - 2. * A_t_tt**2 / B )

    def calc_rhs_dt_K(self):
        """ Eq. 9
        """
        alpha = self.alpha
        K = self.K
        A_t_rr = self.A_t_rr
        A = self.A
        A_t_tt = self.A_t_tt
        B = self.B
        D = self.D

        return alpha * ( 1./3. * K**2 + A_t_rr**2 / A**2 + 2. * A_t_tt**2 / B**2 ) - D
    
    ################################################################
    ## get modes
    def get_modes_alpha(self):
        """ Eq. 42
        """
        AL = self.sd.SBN[::2, :].transpose()

        return np.linalg.solve(AL, self.alpha - 1)

    def get_modes_chi(self):
        """ Eq. 43
        """
        AL = self.sd.SBN[::2, :].transpose()

        return np.linalg.solve(AL, self.chi)
    
    def get_modes_K(self):
        """ Eq. 44
        """
        AL = self.sd.SBN[::2, :].transpose()

        return np.linalg.solve(AL, self.K)

    def get_modes_A(self):
        """ Eq. 45
        """
        AL = 1. / 2. * (self.sd.SBN[2::2, :-1] - self.sd.SBN[0:-2:2, :-1]) # do not include r_(N+1)
        AL = AL.transpose()

        return np.linalg.solve(AL, self.A[:-1] - 1)
    
    def get_modes_B(self):
        """ Eq. 46
        """
        AL = 1. / 2. * (self.sd.SBN[2::2, :-1] - self.sd.SBN[0:-2:2, :-1]) # do not include r_(N+1)
        AL = AL.transpose()

        return np.linalg.solve(AL, self.B[:-1] - 1)
    
    def get_modes_A_a(self):
        """ Eq. 47
        """
        AL = 1. / 2. * (self.sd.SBN[2::2, :-1] - self.sd.SBN[0:-2:2, :-1]) # do not include r_(N+1)
        AL = AL.transpose()

        return np.linalg.solve(AL, self.A_a[:-1])
    
    def get_modes_Lambda(self):
        """ Eq. 48
        """
        AL = self.sd.SBN[1::2, :-1] # do not include r_(N+1)
        AL = AL.transpose()

        return np.linalg.solve(AL, self.Lambda[:-1])
    
    ################################################################
    ## derivative terms
    def deriv_r_alpha(self, modes):
        """ d/dr alpha
        """
        DAL = self.sd.SBN_dr[::2].transpose()

        return DAL@modes
    
    def deriv_r_chi(self, modes):
        """ d/dr chi
        """
        DAL = self.sd.SBN_dr[::2].transpose()

        return DAL@modes

    def deriv_r_psi(self):
        """ d/dr psi
        needs Info of psi and already d/dr chi
        """
        # d/dr psi = psi * d/dr chi
        return self.psi * self.chi_dr

    def deriv_r_K(self, modes):
        """ d/dr K
        """
        DAL = self.sd.SBN_dr[::2].transpose()

        return DAL@modes
    
    def deriv_r_A(self, modes):
        """ d/dr A
        """
        DAL = 1./2. * (self.sd.SBN_dr[2::2, :-1] - self.sd.SBN_dr[0:-2:2, :-1]) # do not include r_N+1
        DAL = DAL.transpose()

        return DAL@modes
    
    def deriv_r_B(self, modes):
        """ d/dr B
        """
        DAL = 1./2. * (self.sd.SBN_dr[2::2, :-1] - self.sd.SBN_dr[0:-2:2, :-1]) # do not include r_N+1
        DAL = DAL.transpose()

        return DAL@modes
    
    def deriv_r_Lambda(self, modes):
        """ d/dr Lambda
        """
        DAL = self.sd.SBN_dr[1::2, :-1].transpose() # do not include r_N+1

        return DAL@modes

    def deriv_rr_alpha(self, modes):
        """ d^2/dr^2 alpha
        """
        DAL = self.sd.SBN_ddr[::2].transpose()

        return DAL@modes
    
    def deriv_rr_chi(self, modes):
        """ d^2/dr^2 chi
        """
        DAL = self.sd.SBN_ddr[::2].transpose()

        return DAL@modes

    def deriv_rr_psi(self):
        """ d^2/dr^2 psi
        Needs psi, d/dr chi and d^2/dr^2 chi
        """
        # d^2/dr^2 psi = psi ( d^2/dr^2 chi + (d/dr chi)^2 )
        return self.psi*( self.chi_ddr + self.chi_dr**2 )
    
    def deriv_rr_A(self, modes):
        """ d^2/dr^2 A
        """
        DDAL = 1./2. * (self.sd.SBN_ddr[2::2, :-1] - self.sd.SBN_ddr[0:-2:2, :-1]) # do not include r_N+1
        DDAL = DDAL.transpose()

        return DDAL@modes
    
    def deriv_rr_B(self, modes):
        """ d^2/dr^2 B
        """
        DDAL = 1./2. * (self.sd.SBN_ddr[2::2, :-1] - self.sd.SBN_ddr[0:-2:2, :-1]) # do not include r_N+1
        DDAL = DDAL.transpose()

        return DDAL@modes