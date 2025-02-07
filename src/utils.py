import numpy as np
import warnings

# ------------------------------------------------------------------------------------------------------------

def find_nearest(array, value):
    array = np.asarray(array)
    if isinstance(value, list):
        idx = [(np.abs(array - value[i])).argmin() for i in range(len(value))]
    else:
        idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def mc_errors(f, params, errors, n_random=5000):
    params, errors = np.array(params), np.array(errors)

    model_MC = []
    for i in range(n_random):
        params_MC = params + errors * np.random.normal(size=np.shape(params))
        model_MC.append(f(*params_MC))
    model_MC = np.array(model_MC)

    return (np.nanpercentile(model_MC, 84, axis=0) - np.nanpercentile(model_MC, 16, axis=0)) / 2

def get_median_properties(tab):
    # Get median redshift of the subsample
    z16, z_med, z84 = np.percentile(tab["z"], [16,50,84])
    # Get median SFR of the subsample, ignoring those with no measurements
    sfr16, sfr_med, sfr84 = 10**np.median(tab["logSFR"][tab["logSFR"]>0], [16,50,84])
    # Get median stellar mass of the subsample, ignoring those with no measurements
    mstar16, mstar_med, mstar84 = 10**np.median(tab["logMstar"][tab["logMstar"]>0], [16,50,84])
    
    return z16, z_med, z84, sfr16, sfr_med, sfr84, mstar16, mstar_med, mstar84

def disable_ax_ticks(ax, wcs=False):
    if not wcs:
        ax.set_xticks([])
        ax.set_yticks([])
    elif wcs:
        ax.coords[0].set_ticks_visible(False)
        ax.coords[0].set_ticklabel_visible(False)
        ax.coords[1].set_ticks_visible(False)
        ax.coords[1].set_ticklabel_visible(False)

def moments(v, I_v, dI_v, sig_clip=999., n_mc=1000):
    """
    Calculate the first three moments of a given spectrum

    Parameters
    ----------
    v: array
        velocities of the spectrum
    I_v: array
        intensities as a function of v
    dI_v: array
        uncertainties on I_v
    sig_clip: float
        sigma clip to apply to negative values

    Returns
    -------
    out: array
        first three moments of the spectrum
    """
    warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")

    v_c = v.copy()[I_v>-sig_clip*dI_v]
    I_v_c = I_v.copy()[I_v>-sig_clip*dI_v]
    dI_v_c = dI_v.copy()[I_v>-sig_clip*dI_v]

    dv = abs(np.nanmedian(v[0:-1]-v[1:]))                       # velocity channel width

    M0 = np.nansum(I_v_c)*dv                                    # zeroeth moment and uncertainty
    # dM0 = np.nansum((dI_v_c*dv)**2)**0.5
    M1 = np.nansum(I_v_c * v_c) * dv/M0                         # first moment and uncertainty
    # dM1 = np.nansum(dI_v_c**2 * (v_c-M1)**2)**0.5
    M2 = (np.nansum(I_v_c * (v_c - M1)**2) * dv/M0)**0.5        # second moment and uncertainty
    # dM2 = 1/(2*M2) * np.nansum(dI_v_c**2 * ((v_cv_c - M1)**2 - M2**2)**2)**0.5

    # MC errors
    M0_mcs, M1_mcs, M2_mcs = [], [], []
    for i in range(n_mc):
        I_v_mc = I_v_c + np.random.normal()*dI_v_c
        M0_mc = np.nansum(I_v_mc)*dv
        M1_mc = np.nansum(I_v_mc * v_c) * dv/M0_mc
        M2_mc = (np.nansum(I_v_mc * (v_c - M1_mc)**2) * dv/M0_mc)**0.5
        M0_mcs.append(M0_mc)
        M1_mcs.append(M1_mc)
        M2_mcs.append(M2_mc)
    dM0 = np.nanstd(M0_mcs)
    dM1 = np.nanstd(M1_mcs)
    dM2 = np.nanstd(M2_mcs)

    return M0, dM0, M1, dM1, M2, dM2
