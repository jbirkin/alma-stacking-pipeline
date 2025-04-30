import numpy as np

# ------------------------------------------------------------------------------------------------------------

# Relations from Schaerer+20 (ALPINE)
def LCII_SFR_schaerer(SFR):
    return 10**(7.03 + 1*np.log10(SFR))

def SFR_schaerer(LCII):
    return 10**((np.log10(LCII) - 7.03) / 1)

# Relations from Lagache+18
def LCII_SFR_lagache(SFR, z):
    return (1.4-0.07*z)*np.log10(SFR)+7.1-0.07*z

def SFR_lagache(LCII, z):
    return 10**((np.log10(LCII) - (7.1-0.07*z))/(1.4-0.07*z))

# Relations from de Looze+14
def LCII_SFR_de_looze(SFR):
    return (np.log10(SFR)+6.99)/1.01

def SFR_de_looze(LCII):
    return 10**(-6.99 + 1.01*np.log10(LCII))

def SFR_de_looze_highz(LCII):
    return 10**(-8.52 + 1.18*np.log10(LCII))

# ------------------------------------------------------------------------------------------------------------

def m_atom_out(Lcii, X_C=1.4e-04, ncrit=3e03, n=3e03, Tgas=100):
    """
    Estimate atomic mass of outflow using [CII] luminosity
    (from Ginolfi+20 originally Hailey-Dunsheath+10)

    :param Lcii: [CII] luminosity, in units of Lsol
    :type Lcii: float
    :param X_C: C+ abundance per Hydrogen atom
    :type X_C: float
    :param ncrit: [CII] critical density
    :type ncrit: float
    :param n: gas number density
    :type n: float
    :param Tgas: gas temperature
    :type Tgas: float
    :return: outflow atomic mass
    :rtype: float
    """
    return 0.77 * Lcii * (1.4e-04/X_C) * (1+2*np.exp(-91/Tgas)+ncrit/n)/(2*np.exp(-91/Tgas))

def m_out_dot(v_out, M_out, R_out):
    """
    Estimate mass outflow rate
    (from Ginolfi+20)

    :param v_out: outflow velocity in km/s
    :type v_out: float
    :param M_out: mass of outflow in Msol
    :type M_out: float
    :param R_out: spatial extent of outflow region, in kpc
    :type R_out: float
    :return: mass outflow rate, in Msol/yr
    :rtype: float
    """
    return (v_out*60*60*24*365) * M_out / (R_out*3.09e16)