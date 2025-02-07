import numpy as np

# ------------------------------------------------------------------------------------------------------------
# Relations from Schaerer+20 (ALPINE)

def convert_SFR_to_LCII_schaerer(SFR):
    """
    Convert Star Formation Rate (SFR) to [CII] luminosity using Schaerer+20 relation.

    :param SFR: Star Formation Rate (in solar masses per year)
    :return: [CII] luminosity (L_CII) in solar luminosities
    """
    return 10 ** (7.03 + np.log10(SFR))

def convert_LCII_to_SFR_schaerer(LCII):
    """
    Convert [CII] luminosity to Star Formation Rate (SFR) using Schaerer+20 relation.

    :param LCII: [CII] luminosity (in solar luminosities)
    :return: Star Formation Rate (SFR) in solar masses per year
    """
    return 10 ** ((np.log10(LCII) - 7.03))


# ------------------------------------------------------------------------------------------------------------
# Relations from Lagache+18

def convert_SFR_to_LCII_lagache(SFR, z):
    """
    Convert SFR to [CII] luminosity using Lagache+18 relation.

    :param SFR: Star Formation Rate (in solar masses per year)
    :param z: Redshift
    :return: [CII] luminosity (L_CII) in solar luminosities
    """
    slope = 1.4 - 0.07 * z
    intercept = 7.1 - 0.07 * z
    return slope * np.log10(SFR) + intercept

def convert_LCII_to_SFR_lagache(LCII, z):
    """
    Convert [CII] luminosity to SFR using Lagache+18 relation.

    :param LCII: [CII] luminosity (in solar luminosities)
    :param z: Redshift
    :return: Star Formation Rate (SFR) in solar masses per year
    """
    slope = 1.4 - 0.07 * z
    intercept = 7.1 - 0.07 * z
    return 10 ** ((np.log10(LCII) - intercept) / slope)

# ------------------------------------------------------------------------------------------------------------
# Relations from de Looze+14

def convert_SFR_to_LCII_de_looze(SFR):
    """
    Convert SFR to [CII] luminosity using de Looze+14 relation.
    """
    return (np.log10(SFR) + 6.99) / 1.01

def convert_LCII_to_SFR_de_looze(LCII):
    """
    Convert [CII] luminosity to SFR using de Looze+14 relation.
    """
    return 10**(-6.99 + 1.01 * np.log10(LCII))

def convert_LCII_to_SFR_de_looze_highz(LCII):
    """
    Convert [CII] luminosity to SFR using de Looze+14 relation for high-redshift galaxies.
    """
    return 10**(-8.52 + 1.18 * np.log10(LCII))


# ------------------------------------------------------------------------------------------------------------
# Atomic mass outflow calculations

def compute_m_atom_out(Lcii, X_C=1.4e-4, ncrit=3e3, n=3e3, Tgas=100):
    """
    Compute atomic mass outflow rate from [CII] luminosity.

    :param Lcii: [CII] luminosity (in solar luminosities)
    :param X_C: Carbon abundance fraction (default: 1.4e-4)
    :param ncrit: Critical density (default: 3000 cm^-3)
    :param n: Gas density (default: 3000 cm^-3)
    :param Tgas: Gas temperature (default: 100 K)
    :return: Atomic mass outflow rate (in solar masses per year)
    """
    partition_function = 1 + 2 * np.exp(-91 / Tgas) + ncrit / n
    return 0.77 * Lcii * (1.4e-4 / X_C) * partition_function / (2 * np.exp(-91 / Tgas))

def compute_m_out_dot(v_out, M_out, R_out):
    """
    Compute mass outflow rate.

    :param v_out: Outflow velocity (km/s)
    :param M_out: Outflow mass (solar masses)
    :param R_out: Outflow radius (parsecs)
    :return: Mass outflow rate (solar masses per year)
    """
    sec_per_year = 60 * 60 * 24 * 365  # Convert seconds to years
    parsec_to_cm = 3.09e16  # Conversion factor
    return (v_out * sec_per_year) * M_out / (R_out * parsec_to_cm)