from defaults import cosmo
import numpy as np
import emcee
import astropy.units as u
from tools import mc_errors
from scipy.integrate import simpson

from alma_stacking_pipeline.src.config_loader import nu_cii
from alma_stacking_pipeline.src.formulae import compute_m_atom_out, compute_m_out_dot, convert_LCII_to_SFR_lagache

# ------------------------------------------------------------------------------------------------------------
# models

def gaussian(x, amp, mean, sig):
    """
    single Gaussian function

    :param x: x-axis for which to evaluate the Gaussian
    :type x: arr
    :param amp: peak ampltiude of the Gaussian
    :type amp: float
    :param mean: centre of the Gaussian
    :type mean: float
    :param sig: standard deviation of the Gaussian
    :type sig: float
    :return: Gaussian model evaluated at x
    :rtype: arr
    """
    return amp*np.exp(-(x-mean)**2/2/sig**2)

def gaussian_c(x, amp, mean, sig, c):
    """
    single Gaussian function with continuum
    """
    return amp*np.exp(-(x-mean)**2/2/sig**2) + c

def gaussian_double(x, amp1, mean1, sig1, amp2, mean2, sig2):
    """
    double Gaussian function. Parameters are the same as for the model above, but not allowing for a second
    Gaussian
    """
    return amp1*np.exp(-(x-mean1)**2/2/sig1**2) + amp2*np.exp(-(x-mean2)**2/2/sig2**2)

def gaussian_double_fixed_centre(x, amp1, sig1, amp2, sig2):
    """
    double Gaussian function. Parameters are the same as for the model above, but not allowing for a second
    Gaussian
    """
    return amp1*np.exp(-x**2/2/sig1**2) + amp2*np.exp(-x**2/2/sig2**2)

# ------------------------------------------------------------------------------------------------------------

class line_model_single:
    def __init__(self, amp1=None, amp1_err=None, mean1=None, mean1_err=None,
                 sig1=None, sig1_err=None, z=None, sfr=None):
        self.amp1, self.amp1_err = amp1, amp1_err
        self.mean1, self.mean1_err = mean1, mean1_err
        self.sig1, self.sig1_err = sig1, sig1_err
        self.z = z
        self.sfr = sfr

        self.Dl = cosmo.luminosity_distance(self.z).to(u.Mpc).value

    def get_line_props(self):
        # Narrow line properties
        self.Icii_narrow = self.amp1*self.sig1*(2*np.pi)**0.5/1000     # flux of [CII] outflow (Jy km/s)
        self.Icii_narrow_err = -99.
        self.Lcii_narrow = 1.04e-03*self.Icii_narrow*nu_cii/(1+self.z)*self.Dl**2        # luminosity of [CII] outflow
        self.Lcii_narrow_err = -99.
        self.sfr_cii = convert_LCII_to_SFR_lagache(self.Lcii_narrow, self.z)

    def get_errors(self):
        f = lambda amp, sig : amp*sig*(2*np.pi)**0.5/1000
        self.Icii_narrow_err = mc_errors(f=f, params=[self.amp1,self.sig1], errors=[self.amp1_err,self.sig1_err])
        f = lambda Icii : 1.04e-03*Icii*nu_cii/(1+self.z)*self.Dl**2
        self.Lcii_narrow_err = mc_errors(f=f, params=[self.Icii_narrow], errors=[self.Icii_narrow_err])
        self.sfr_cii_err = mc_errors(f=convert_LCII_to_SFR_lagache, params=[self.Lcii_narrow,self.z],
                                     errors=[self.Lcii_narrow_err,0])

# ------------------------------------------------------------------------------------------------------------

class line_model_double:
    def __init__(self, amp1=None, amp1_err=None, amp2=None, amp2_err=None, mean1=None, mean1_err=None, mean2=None,
                 mean2_err=None, sig1=None, sig1_err=None, sig2=None, sig2_err=None, z=None, sfr=None):
        self.amp1, self.amp1_err = amp1, amp1_err
        self.amp2, self.amp2_err = amp2, amp2_err
        self.mean1, self.mean1_err = mean1, mean1_err
        self.mean2, self.mean2_err = mean2, mean2_err
        self.sig1, self.sig1_err = sig1, sig1_err
        self.sig2, self.sig2_err = sig2, sig2_err
        self.z = z
        self.sfr = sfr

        self.Dl = cosmo.luminosity_distance(self.z).to(u.Mpc).value

        # Line flux measurements
        self.Icii_narrow, self.Icii_narrow_err = None, None
        self.Icii_wings, self.Icii_wings_err = None, None
        self.Icii_out, self.Icii_out_err = None, None
        # Line luminosity measurements
        self.Lcii_narrow, self.Lcii_narrow_err = None, None
        self.Lcii_wings, self.Lcii_wings_err = None, None
        self.Lcii_out, self.Lcii_out_err = None, None
        # Broad wing measurements (Lutz+20)
        self.broad_wings_only = None
        self.broad_wing_flux = None
        self.broad_wing_ind = None
        # [CII] SFR
        self.sfr_cii, self.sfr_cii_err = None, None
        # Outflow properties
        self.v_out, self.v_out_err = None, None
        self.m_out, self.m_out_err = None, None
        self.m_out_wings, self.m_out_wings_err = None, None
        self.m_out_dot, self.m_out_dot_err = None, None
        self.m_out_dot_wings, self.m_out_dot_wings_err = None, None
        self.out_frac, self.out_frac_err = None, None
        self.mass_load_factor, self.mass_load_factor_err = None, None
        self.mass_load_factor_wings, self.mass_load_factor_wings_err = None, None
        self.e_out_dot, self.e_out_dot_err = None, None

    def get_line_props(self):
        # Narrow line properties
        self.Icii_narrow = self.amp1*self.sig1*(2*np.pi)**0.5/1000     # flux of [CII] outflow (Jy km/s)
        self.Icii_narrow_err = -99.
        self.Lcii_narrow = 1.04e-03*self.Icii_narrow*nu_cii/(1+self.z)*self.Dl**2        # luminosity of [CII] outflow
        self.Lcii_narrow_err = -99.
        self.sfr_cii = convert_LCII_to_SFR_lagache(self.Lcii_narrow, self.z)

        # Broad line properties
        self.Icii_out = self.amp2*self.sig2*(2*np.pi)**0.5/1000     # flux of [CII] outflow (Jy km/s)
        self.Icii_out_err = -99.
        self.Lcii_out = 1.04e-03*self.Icii_out*nu_cii/(1+self.z)*self.Dl**2        # luminosity of [CII] outflow
        self.Lcii_out_err = -99.

    def get_wing_props(self, X):
        # Lutz+20 model for isolating broad components
        self.broad_wings_only = gaussian_double(X, self.amp1, self.mean1, self.sig1, self.amp2, self.mean2, self.sig2).copy()
        self.broad_wing_ind = gaussian(X, self.amp2, self.mean2, self.sig2) >= 0.5 * \
                         gaussian_double(X, self.amp1, self.mean1, self.sig1, self.amp2, self.mean2, self.sig2)
        self.broad_wings_only[self.broad_wing_ind == False] = 0
        self.broad_wing_flux = simpson(self.broad_wings_only, x=X)
        self.broad_wings_only[self.broad_wing_ind == False] = np.nan

        self.Icii_wings = self.broad_wing_flux/1000
        self.Lcii_wings = 1.04e-03*self.Icii_wings*nu_cii/(1+self.z)*self.Dl**2

    def get_outflow_props(self):
        # Outflow velocity, measured according to Lutz+20
        self.v_out = abs(self.mean2) + 4.292*self.sig2/2
        # Atomic outflow mass, from Ginolfi+20
        self.m_out = compute_m_atom_out(self.Lcii_out)
        self.m_out_err = -99.
        # Atomic outflow mass, from Lutz+20
        self.m_out_wings = compute_m_atom_out(self.Lcii_wings)
        self.m_out_wings_err = -99.
        # Mass outflow rates
        self.m_out_dot = compute_m_out_dot(v_out=self.v_out, M_out=self.m_out, R_out=6)
        self.m_out_dot_err = -99.
        self.m_out_dot_wings = compute_m_out_dot(v_out=self.v_out, M_out=self.m_out_wings, R_out=6)
        self.m_out_dot_wings_err = -99.
        #
        self.out_frac = (self.amp2*self.sig2)/(self.amp1*self.sig1)
        self.out_frac_err = -99.
        # Mass loading factors
        self.mass_load_factor = self.m_out_dot/self.sfr
        self.mass_load_factor_err = -99.
        self.mass_load_factor_wings = self.m_out_dot_wings/self.sfr
        self.mass_load_factor_wings_err = -99.
        # Outflow energy rate
        self.e_out_dot = 0.5*self.m_out_dot*self.v_out**2

    def get_errors(self):
        f = lambda amp, sig : amp*sig*(2*np.pi)**0.5/1000
        self.Icii_narrow_err = mc_errors(f=f, params=[self.amp1, self.sig1], errors=[self.amp1_err, self.sig1_err])
        self.Icii_out_err = mc_errors(f=f, params=[self.amp2,self.sig2], errors=[self.amp2_err,self.sig2_err])
        self.Icii_wings_err = self.Icii_wings * (self.Icii_out_err / self.Icii_out)
        f = lambda Icii : 1.04e-03*Icii*nu_cii/(1+self.z)*self.Dl**2
        self.Lcii_narrow_err = mc_errors(f=f, params=[self.Icii_narrow], errors=[self.Icii_narrow_err])
        self.Lcii_out_err = mc_errors(f=f, params=[self.Icii_out], errors=[self.Icii_out_err])
        self.Lcii_wings_err = mc_errors(f=f, params=[self.Icii_wings], errors=[self.Icii_wings_err])
        f = lambda mean2, sig2 : abs(mean2) + 4.292*sig2/2
        self.v_out_err = mc_errors(f=f, params=[self.mean2,self.sig2], errors=[self.mean2_err,self.sig2_err])
        self.m_out_err = mc_errors(f=compute_m_atom_out, params=[self.Lcii_out], errors=[self.Lcii_out_err])
        self.m_out_dot_err = mc_errors(f=compute_m_out_dot, params=[self.v_out,self.m_out,6],
                                       errors=[0,self.m_out_err,0])
        self.m_out_wings_err = mc_errors(f=compute_m_atom_out, params=[self.Lcii_wings], errors=[self.Lcii_wings_err])
        self.m_out_dot_wings_err = mc_errors(f=compute_m_out_dot, params=[self.v_out,self.m_out_wings,6],
                                       errors=[0,self.m_out_wings_err,0])
        f = lambda amp2, sig2, amp1, sig1 : (amp2*sig2)/(amp1*sig1)
        self.out_frac_err = mc_errors(f=f, params=[self.amp2,self.sig2,self.amp1,self.sig1],
                                      errors=[self.amp2_err,self.sig2_err,self.amp1_err,self.sig1_err])
        f = lambda m_out_dot, sfr : m_out_dot/sfr
        self.mass_load_factor_err = mc_errors(f=f, params=[self.m_out_dot,self.sfr],
                                              errors=[self.m_out_dot_err,0])
        self.sfr_cii_err = mc_errors(f=convert_LCII_to_SFR_lagache, params=[self.Lcii_narrow,self.z],
                                     errors=[self.Lcii_narrow_err,0])
        f = lambda m_out_dot, v_out : 0.5*m_out_dot*v_out**2
        self.e_out_dot_err = mc_errors(f=f, params=[self.m_out_dot,self.v_out],
                                       errors=[self.m_out_dot_err,self.v_out_err])

    def get_broad_wings(self, X):
        return broad_wing_ind, broad_wings_only, broad_wing_flux

# ------------------------------------------------------------------------------------------------------------

def log_prior_single(theta, bounds):
    amp1, mean1, sig1 = theta
    bounds_lower, bounds_upper = bounds
    amp1_l, mean1_l, sig1_l = bounds_lower
    amp1_u, mean1_u, sig1_u = bounds_upper
    if amp1_l < amp1 < amp1_u and mean1_l < mean1 < mean1_u and sig1_l < sig1 < sig1_u:
        return 0.0
    return -np.inf

def log_likelihood_single(theta, x, y, yerr):
    amp1, mean1, sig1 = theta
    model = gaussian(x, amp1, mean1, sig1)
    sigma2 = yerr**2# + model**2 * np.exp(2 * log_f)
    return -0.5 * np.sum((y - model) ** 2 / sigma2)# + np.log(sigma2))

def log_probability_single(theta, bounds, x, y, yerr):
    lp = log_prior_single(theta, bounds)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_single(theta, x, y, yerr)

# ------------------------------------------------------------------------------------------------------------

def log_prior_double(theta, bounds):
    amp1, mean1, sig1, amp2, mean2, sig2 = theta
    bounds_lower, bounds_upper = bounds
    amp1_l, mean1_l, sig1_l, amp2_l, mean2_l, sig2_l = bounds_lower
    amp1_u, mean1_u, sig1_u, amp2_u, mean2_u, sig2_u = bounds_upper
    if amp2 > 0.5*amp1:
        return -np.inf
    if sig2 < 1.2*sig1:
        return -np.inf
    if amp1_l < amp1 < amp1_u and mean1_l < mean1 < mean1_u and sig1_l < sig1 < sig1_u\
            and amp2_l < amp2 < amp2_u and mean2_l < mean2 < mean2_u and sig2_l < sig2 < sig2_u:
        return 0.0
    return -np.inf

def log_likelihood_double(theta, x, y, yerr):
    amp1, mean1, sig1, amp2, mean2, sig2 = theta
    model = gaussian_double(x, amp1, mean1, sig1, amp2, mean2, sig2)
    sigma2 = yerr**2# + model**2 * np.exp(2 * log_f)
    return -0.5 * np.sum((y - model) ** 2 / sigma2)# + np.log(sigma2))

def log_probability_double(theta, bounds, x, y, yerr):
    lp = log_prior_double(theta, bounds)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_double(theta, x, y, yerr)

# ------------------------------------------------------------------------------------------------------------