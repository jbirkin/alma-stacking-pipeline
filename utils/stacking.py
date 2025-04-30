import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from astropy.stats import bootstrap

from astro_cubes.tools import mc_errors
from astro_cubes.defaults import colors
from src.utils.tables import cristal_dir, ctab, cosmo
from src.utils.line_models import gaussian, gaussian_double
from src.utils.cristal_cube import cristal_spec

# ------------------------------------------------------------------------------------------------------------

def prelim_stack(vels, fluxes):
    """
    Stack spectra with equal weights to all fluxes

    :param vels: velocities to stack
    :type vels: array
    :param fluxes: fluxes to stack
    :type fluxes: array
    :return: velocity axis of the stack, corresponding fluxes, and the number of sources in each channel
    :rtype: array, array, array
    """
    stack_vel, stack_flux, stack_bins = [], [], []
    v_min, v_max = np.min(vels), np.max(vels)
    bins = np.linspace(v_min, v_max, int((v_max - v_min) * 0.1))
    for i in range(len(bins) - 1):
        flux_i = fluxes[(vels > bins[i]) & (vels < bins[i + 1])]
        stack_vel.append((bins[i] + bins[i + 1]) / 2)
        stack_flux.append(np.median(flux_i))
        stack_bins.append(len(flux_i))

    stack_vel, stack_flux, stack_bins = np.array(stack_vel), np.array(stack_flux), np.array(stack_bins)
    return stack_vel, stack_flux, stack_bins

def uniform_weighted_stack(vels, fluxes, errs, vel_res=50., mode="sum"):
    """
    Stack spectra with uniform weighting

    :param vels: velocities to stack
    :type vels: array
    :param fluxes: fluxes to stack
    :type fluxes: array
    :param errs: errors to stack
    :type errs: array
    :param vel_res: velocity resolution for the stack
    :type vel_res: float
    :return: velocity axis of the stack, corresponding fluxes, and the number of sources in each channel
    :rtype: array, array, array
    """
    stack_vel, stack_flux, stack_err, stack_bins = [], [], [], []

    vels_flat = np.concatenate(vels)
    fluxes_flat = np.concatenate(fluxes)
    errs_flat = np.concatenate(errs)

    v_min, v_max = np.nanmin(vels_flat), np.nanmax(vels_flat)
    bins = np.arange(v_min, v_max, vel_res)

    for i in range(len(bins) - 1):
        flux_i = fluxes_flat[(vels_flat > bins[i]) & (vels_flat < bins[i + 1])]
        errs_i = errs_flat[(vels_flat > bins[i]) & (vels_flat < bins[i + 1])]
        bad = np.where((flux_i==0) | (errs_i==0))
        flux_i = np.delete(flux_i, bad)
        errs_i = np.delete(errs_i, bad)
        weights_i = np.ones(len(errs_i))
        if len(flux_i)==0:
            continue
        stack_vel.append((bins[i] + bins[i + 1]) / 2)
        if mode=="sum":
            stack_flux.append(np.sum(flux_i*weights_i)/np.sum(weights_i))
            # stack_err.append(np.sum(errs_i**2)**0.5)
            # stack_err.append((1/np.sum(weights_i))**0.5)
        elif mode=="median":
            stack_flux.append(np.median(flux_i*weights_i)/np.sum(weights_i))
            # stack_err.append(np.median(errs_i**2)**0.5)

        # estimate uncertainty for the bin
        n_random = 100
        if mode == "sum":
            f = lambda flux, weight : np.sum(flux * weight) / np.sum(weight)
        elif mode == "median":
            f = lambda flux, weight: np.median(flux * weight) / np.sum(weight)
        mc_error = mc_errors(f=f, params=[flux_i,weights_i], errors=[errs_i,np.zeros(len(weights_i))],
                              n_random=n_random)
        stack_err.append(mc_error)

        # get the number of sources in the bin
        count = 0
        for j in range(np.shape(vels)[0]):
            # print(vels[j], bins[i], bins[i+1], ((vels[j] > bins[i]) & (vels[j] < bins[i+1])).any())
            if ((vels[j] > bins[i]) & (vels[j] < bins[i+1])).any():
                count+=1
        stack_bins.append(count)

    stack_vel, stack_flux = np.array(stack_vel), np.array(stack_flux)
    stack_err, stack_bins = np.array(stack_err), np.array(stack_bins)

    return stack_vel, stack_flux, stack_err, stack_bins

def variance_weighted_stack(vels, fluxes, errs, vel_res=50., mode="sum"):
    """
    Stack spectra weighting the flux in each channel by 1/err^2

    :param vels: velocities to stack
    :type vels: array
    :param fluxes: fluxes to stack
    :type fluxes: array
    :param errs: errors to stack
    :type errs: array
    :param vel_res: velocity resolution for the stack
    :type vel_res: float
    :return: velocity axis of the stack, corresponding fluxes, and the number of sources in each channel
    :rtype: array, array, array
    """
    stack_vel, stack_flux, stack_err, stack_bins = [], [], [], []

    vels_flat = np.concatenate(vels)
    fluxes_flat = np.concatenate(fluxes)
    errs_flat = np.concatenate(errs)

    v_min, v_max = np.nanmin(vels_flat), np.nanmax(vels_flat)
    bins = np.arange(v_min, v_max, vel_res)

    for i in range(len(bins) - 1):
        flux_i = fluxes_flat[(vels_flat > bins[i]) & (vels_flat < bins[i + 1])]
        errs_i = errs_flat[(vels_flat > bins[i]) & (vels_flat < bins[i + 1])]
        # flux_i = np.delete(flux_i, flux_i==0)
        # errs_i = np.delete(errs_i, errs_i==0)
        bad = np.where((flux_i==0) | (errs_i==0))
        flux_i = np.delete(flux_i, bad)
        errs_i = np.delete(errs_i, bad)
        weights_i = 1/errs_i**2
        if len(flux_i)==0:
            continue
        stack_vel.append((bins[i] + bins[i + 1]) / 2)
        if mode=="sum":
            stack_flux.append(np.sum(flux_i*weights_i)/np.sum(weights_i))
            # stack_err.append(np.sum(errs_i**2)**0.5)
            # stack_err.append((1/np.sum(weights_i))**0.5)
        elif mode=="median":
            stack_flux.append(np.median(flux_i*weights_i)/np.sum(weights_i))
            # stack_err.append(np.median(errs_i**2)**0.5)

        # estimate uncertainty for the bin
        n_random = 100
        if mode == "sum":
            f = lambda flux, weight : np.sum(flux * weight) / np.sum(weight)
        elif mode == "median":
            f = lambda flux, weight: np.median(flux * weight) / np.sum(weight)
        mc_error = mc_errors(f=f, params=[flux_i,weights_i], errors=[errs_i,np.zeros(len(weights_i))],
                              n_random=n_random)
        stack_err.append(mc_error)

        # get the number of sources in the bin
        count = 0
        for j in range(np.shape(vels)[0]):
            # print(vels[j], bins[i], bins[i+1], ((vels[j] > bins[i]) & (vels[j] < bins[i+1])).any())
            if ((vels[j] > bins[i]) & (vels[j] < bins[i+1])).any():
                count+=1
        stack_bins.append(count)

    stack_vel, stack_flux = np.array(stack_vel), np.array(stack_flux)
    stack_err, stack_bins = np.array(stack_err), np.array(stack_bins)

    return stack_vel, stack_flux, stack_err, stack_bins

def variance_weighted_stack_with_centroid_MC(vels, fluxes, errs, centroid_errs, vel_res=50., mode="sum", n_mc=100):
    """
    Stack spectra with variance weighting, including centroid and flux uncertainties via MC.
    """
    all_stack_fluxes = []

    # Define fixed global binning from original velocities
    vels_flat = np.concatenate(vels)
    v_min, v_max = np.nanmin(vels_flat), np.nanmax(vels_flat)
    bins = np.arange(v_min, v_max, vel_res)
    stack_vel = (bins[:-1] + bins[1:]) / 2

    for mc in range(n_mc):
        vels_mc, fluxes_mc = [], []

        # Perturb centroids and fluxes
        for v, f, e, cent_err in zip(vels, fluxes, errs, centroid_errs):
            shift = np.random.normal(loc=0, scale=cent_err)
            vels_mc.append(v + shift)
            fluxes_mc.append(f + np.random.normal(size=f.shape) * e)

        # Flatten arrays for stacking
        vels_flat_mc = np.concatenate(vels_mc)
        fluxes_flat_mc = np.concatenate(fluxes_mc)
        errs_flat = np.concatenate(errs)  # Errors don't change, only fluxes are perturbed
        stack_flux_mc = []

        for i in range(len(bins) - 1):
            # Select data in the velocity bin
            in_bin = (vels_flat_mc > bins[i]) & (vels_flat_mc < bins[i + 1])
            flux_i = fluxes_flat_mc[in_bin]
            errs_i = errs_flat[in_bin]

            # Remove bad values
            bad = np.where((flux_i == 0) | (errs_i == 0))
            flux_i = np.delete(flux_i, bad)
            errs_i = np.delete(errs_i, bad)

            weights_i = 1 / errs_i**2

            if len(flux_i) == 0:
                stack_flux_mc.append(np.nan)
                continue

            if mode == "sum":
                stack_flux_mc.append(np.sum(flux_i * weights_i) / np.sum(weights_i))
            elif mode == "median":
                stack_flux_mc.append(np.median(flux_i * weights_i) / np.sum(weights_i))

        all_stack_fluxes.append(stack_flux_mc)

    # Convert to array
    all_stack_fluxes = np.array(all_stack_fluxes)

    # Final stacked flux and error (stddev across MC runs)
    stack_flux = np.nanmean(all_stack_fluxes, axis=0)
    stack_err = np.nanstd(all_stack_fluxes, axis=0)

    return stack_vel, stack_flux, stack_err

def binned_stack_IDs(IDs, spec_dir, vel_res_stack=50, fwhm_norm=True, flux_norm="conserve", use_rms=False,
                     fwhm_norm_med=260, mode="sum", verbose=True):
    ind = []
    for i in range(len(ctab)):
        if ctab["CRISTAL_ID_full"][i] in IDs:
            ind.append(i)
    ctab_sub = ctab[ind]

    if verbose:
        print("Stacking the following sources:")
        print(np.array(ctab_sub["CRISTAL_ID_full"]))

    # --------------------------------------------------------------------------------------------------

    vels, fluxes, errs = [], [], []
    norm_values = []
    for j in range(len(ctab_sub)):
        ID, z = ctab_sub["CRISTAL_ID_full"][j], ctab_sub["z"][j]
        vel, nu, flux, err = np.load(cristal_dir+"spectra/"+spec_dir+ID+"_centered.npy")
        fwhm_cii = ctab_sub["FWHM_CII"][j]
        Spec = cristal_spec(nu=nu, vel=vel, flux=flux, err=err, ID=ID, z=z)
        Spec.get_moments()
        Spec.get_rms()

        # fwhm_norm_value = (fwhm_norm_med/2.35) / Spec.M2
        fwhm_norm_value = fwhm_norm_med / fwhm_cii
        if np.isnan(fwhm_norm_value):
            fwhm_norm_value = 1#fwhm_norm_med/2.35
        unity_norm_value = np.nanmax(Spec.flux)

        if fwhm_norm:
            vels.append(Spec.vel * fwhm_norm_value)
            if flux_norm=="conserve":
                fluxes.append(Spec.flux / fwhm_norm_value)
                if use_rms:
                    errs.append(Spec.rms / fwhm_norm_value)
                else:
                    errs.append(Spec.err / fwhm_norm_value)
                norm_values.append(fwhm_norm_value)
            elif flux_norm=="unity":
                fluxes.append(Spec.flux / unity_norm_value)
                if use_rms:
                    errs.append(Spec.rms / unity_norm_value)
                else:
                    errs.append(Spec.err / unity_norm_value)
                norm_values.append(unity_norm_value)
            else:
                fluxes.append(Spec.flux)
                if use_rms:
                    errs.append(Spec.rms)
                else:
                    errs.append(Spec.err)
        else:
            vels.append(Spec.vel)
            fluxes.append(Spec.flux)
            if use_rms:
                errs.append(Spec.rms)
            else:
                errs.append(Spec.err)
    vels = np.array(vels, dtype=object)
    fluxes = np.array(fluxes, dtype=object)
    errs = np.array(errs, dtype=object)

    stack_vel, stack_flux, stack_err, stack_bins = variance_weighted_stack(vels, fluxes, errs,
                                                                           mode=mode, vel_res=vel_res_stack)
    # stack_vel, stack_flux, stack_err, stack_bins = uniform_weighted_stack(vels, fluxes, errs,
    #                                                                        mode=mode, vel_res=vel_res_stack)

    stack_vel_m = stack_vel.copy()[np.isnan(stack_flux) == False]
    stack_flux_m = stack_flux.copy()[np.isnan(stack_flux) == False]
    stack_err_m = stack_err.copy()[np.isnan(stack_flux) == False]
    stack_bins_m = stack_bins.copy()[np.isnan(stack_flux) == False]

    return stack_vel_m, stack_flux_m, stack_err_m, stack_bins_m, vels, fluxes, errs, norm_values


def binned_stack_IDs_centroid_errs(IDs, spec_dir, vel_res_stack=50, fwhm_norm=True, flux_norm="conserve", use_rms=False,
                     fwhm_norm_med=260, mode="sum", verbose=True, n_mc=100):
    ind = []
    for i in range(len(ctab)):
        if ctab["CRISTAL_ID_full"][i] in IDs:
            ind.append(i)
    ctab_sub = ctab[ind]

    if verbose:
        print("Stacking the following sources:")
        print(np.array(ctab_sub["CRISTAL_ID_full"]))

    vels, fluxes, errs, centroid_errs = [], [], [], []
    norm_values = []

    for j in range(len(ctab_sub)):
        ID, z = ctab_sub["CRISTAL_ID_full"][j], ctab_sub["z"][j]
        vel, nu, flux, err = np.load(cristal_dir + "spectra/" + spec_dir + ID + "_centered.npy")
        fwhm_cii = ctab_sub["FWHM_CII"][j]
        Spec = cristal_spec(nu=nu, vel=vel, flux=flux, err=err, ID=ID, z=z)
        Spec.get_moments()
        Spec.get_rms()

        # Extract centroid uncertainty (adjust column name as needed)
        dM1 = ctab_sub["M1_err"][j]
        centroid_errs.append(dM1)

        fwhm_norm_value = fwhm_norm_med / fwhm_cii
        if np.isnan(fwhm_norm_value):
            fwhm_norm_value = 1
        unity_norm_value = np.nanmax(Spec.flux)

        if fwhm_norm:
            vels.append(Spec.vel * fwhm_norm_value)
            if flux_norm == "conserve":
                fluxes.append(Spec.flux / fwhm_norm_value)
                if use_rms:
                    errs.append(Spec.rms / fwhm_norm_value)
                else:
                    errs.append(Spec.err / fwhm_norm_value)
                norm_values.append(fwhm_norm_value)
            elif flux_norm == "unity":
                fluxes.append(Spec.flux / unity_norm_value)
                if use_rms:
                    errs.append(Spec.rms / unity_norm_value)
                else:
                    errs.append(Spec.err / unity_norm_value)
                norm_values.append(unity_norm_value)
            else:
                fluxes.append(Spec.flux)
                if use_rms:
                    errs.append(Spec.rms)
                else:
                    errs.append(Spec.err)
        else:
            vels.append(Spec.vel)
            fluxes.append(Spec.flux)
            if use_rms:
                errs.append(Spec.rms)
            else:
                errs.append(Spec.err)

    vels = np.array(vels, dtype=object)
    fluxes = np.array(fluxes, dtype=object)
    errs = np.array(errs, dtype=object)
    centroid_errs = np.array(centroid_errs)

    # Use the MC stacking function with centroid perturbations
    stack_vel, stack_flux, stack_err = variance_weighted_stack_with_centroid_MC(
        vels, fluxes, errs, centroid_errs, vel_res=vel_res_stack, mode=mode, n_mc=n_mc
    )

    # Since MC doesn't track stack_bins, set as NaNs (or modify to count if needed)
    stack_bins = np.full_like(stack_vel, np.nan)

    # Remove NaNs
    mask = ~np.isnan(stack_flux)
    stack_vel_m = stack_vel[mask]
    stack_flux_m = stack_flux[mask]
    stack_err_m = stack_err[mask]
    stack_bins_m = stack_bins[mask]

    return stack_vel_m, stack_flux_m, stack_err_m, stack_bins_m, vels, fluxes, errs, norm_values

def bic(data, model, error, n_param):
    return np.sum((data-model)**2/error**2)+n_param*np.log(len(data))

def bic_check(bic_single, bic_double):
    if bic_single-bic_double>0:
        return "double"
    else:
        return "single"

# ------------------------------------------------------------------------------------------------------------

class cristal_stack:
    def __init__(self, spec_dir=None, N_gal=None, ID_list=None, bad_list=None, table=None, fwhm_norm=None, flux_norm=None, stack_vel=None,
                 stack_flux=None, stack_err=None, stack_bins=None, vels=None, fluxes=None, errs=None,
                 norm_med=None, popt=None, perr=None, popt_b=None, perr_b=None, bic_single=None,
                 bic_double=None, bic_result=None, outfile=None):
        self.spec_dir = spec_dir
        self.N_gal = N_gal
        self.ID_list = ID_list
        self.bad_list = bad_list
        self.table = table
        self.fwhm_norm = fwhm_norm
        self.flux_norm = flux_norm
        self.stack_vel = stack_vel
        self.stack_flux = stack_flux
        self.stack_err = stack_err
        self.stack_bins = stack_bins
        self.vels = vels
        self.fluxes = fluxes
        self.errs = errs
        self.norm_med = norm_med
        self.popt = popt
        self.perr = perr
        self.popt_b = popt_b
        self.perr_b = perr_b
        self.bic_single = bic_single
        self.bic_double = bic_double
        self.bic_result = bic_result
        self.outfile = outfile