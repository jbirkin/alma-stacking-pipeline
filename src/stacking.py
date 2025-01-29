import numpy as np
from scipy.optimize import curve_fit

from alma_stacking_pipeline.src.config import cristal_tab
from alma_stacking_pipeline.src.utils import mc_errors
from alma_stacking_pipeline.src.data.alma import alma_spec
from alma_stacking_pipeline.src.line_models import gaussian, gaussian_double

# ------------------------------------------------------------------------------------------------------------

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


def binned_stack_IDs(IDs, tab, spec_dir, vel_res_stack=50, fwhm_norm=True, flux_norm="conserve", use_rms=False,
                     fwhm_norm_med=260, mode="sum", verbose=True):
    ind = []
    for i in range(len(tab)):
        if tab["CRISTAL_ID_full"][i] in IDs:
            ind.append(i)
    tab_sub = tab[ind]

    if verbose:
        print("Stacking the following sources:")
        print(np.array(tab_sub["CRISTAL_ID_full"]))

    # --------------------------------------------------------------------------------------------------

    vels, fluxes, errs = [], [], []
    norm_values = []
    for j in range(len(tab_sub)):
        ID, z = tab_sub["CRISTAL_ID_full"][j], tab_sub["z"][j]
        vel, nu, flux, err = np.load(spec_dir+ID+"_centered.npy")
        Spec = alma_spec(nu=nu, vel=vel, flux=flux, err=err, ID=ID, z=z)
        Spec.get_moments()
        Spec.get_rms()

        fwhm_norm_value = (fwhm_norm_med/2.35) / Spec.M2
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

def bic(data, model, error, n_param):
    return np.sum((data-model)**2/error**2)+n_param*np.log(len(data))

def bic_check(bic_single, bic_double):
    if bic_single-bic_double>0:
        return "double"
    else:
        return "single"