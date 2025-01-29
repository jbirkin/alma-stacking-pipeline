import numpy as np

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