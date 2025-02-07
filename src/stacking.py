import numpy as np
from scipy.optimize import curve_fit
import emcee
import pickle
from pathlib import Path
from dataclasses import dataclass

from alma_stacking_pipeline.src.config_loader import cristal_tab, work_dir
from alma_stacking_pipeline.src.utils import mc_errors
from alma_stacking_pipeline.src.data.alma import alma_spec
from alma_stacking_pipeline.src.line_models import gaussian, gaussian_double


# ------------------------------------------------------------------------------------------------------------

@dataclass
class AlmaStack:
    spec_dir: str = None
    N_gal: int = None
    ID_list: list = None
    table: object = None
    fwhm_norm: bool = None
    flux_norm: str = None
    stack_vel: np.ndarray = None
    stack_flux: np.ndarray = None
    stack_err: np.ndarray = None
    stack_bins: np.ndarray = None
    vels: list = None
    fluxes: list = None
    errs: list = None
    norm_med: float = None
    popt: np.ndarray = None
    perr: np.ndarray = None
    popt_b: np.ndarray = None
    perr_b: np.ndarray = None
    bic_single: float = None
    bic_double: float = None
    bic_result: str = None
    outfile: str = None


# ------------------------------------------------------------------------------------------------------------

def bic(data, model, error, n_param):
    error = np.where(error == 0, 1e-6, error)  # Avoid division by zero
    return np.sum((data - model) ** 2 / error ** 2) + n_param * np.log(len(data))


def bic_check(bic_single, bic_double):
    if bic_single - bic_double > 0:
        return "double"
    else:
        return "single"

# ------------------------------------------------------------------------------------------------------------

def stack_spectra(vels, fluxes, errs, vel_res=50., mode="sum", weight_func=None):
    stack_vel, stack_flux, stack_err, stack_bins = [], [], [], []

    vels_flat, fluxes_flat, errs_flat = map(np.concatenate, (vels, fluxes, errs))
    v_min, v_max = np.nanmin(vels_flat), np.nanmax(vels_flat)
    bins = np.arange(v_min, v_max, vel_res)

    for i in range(len(bins) - 1):
        flux_i = fluxes_flat[(vels_flat > bins[i]) & (vels_flat < bins[i + 1])]
        errs_i = errs_flat[(vels_flat > bins[i]) & (vels_flat < bins[i + 1])]

        bad = np.where((flux_i == 0) | (errs_i == 0))
        flux_i, errs_i = np.delete(flux_i, bad), np.delete(errs_i, bad)

        if len(flux_i) == 0:
            continue

        weights_i = weight_func(errs_i) if weight_func else np.ones(len(errs_i))
        stack_vel.append((bins[i] + bins[i + 1]) / 2)
        if mode == "sum":
            stack_flux.append(np.sum(flux_i * weights_i) / np.sum(weights_i))
            stack_err.append(mc_errors(lambda f, w: np.sum(f * w) / np.sum(w), [flux_i, weights_i], [errs_i,
                            np.zeros(len(weights_i))], 100))
        elif mode == "median":
            stack_flux.append(np.sum(flux_i * weights_i) / np.median(weights_i))
            stack_err.append(mc_errors(lambda f, w: np.sum(f * w) / np.median(w), [flux_i, weights_i], [errs_i,
                            np.zeros(len(weights_i))], 100))
        stack_bins.append(sum(((vel > bins[i]) & (vel < bins[i + 1])).any() for vel in vels))

    return map(np.array, (stack_vel, stack_flux, stack_err, stack_bins))


# ------------------------------------------------------------------------------------------------------------

def uniform_weighted_stack(vels, fluxes, errs, vel_res=50., mode="sum"):
    return stack_spectra(vels, fluxes, errs, vel_res, mode, lambda errs: np.ones_like(errs))


def variance_weighted_stack(vels, fluxes, errs, vel_res=50., mode="sum"):
    return stack_spectra(vels, fluxes, errs, vel_res, mode, lambda errs: 1 / errs ** 2)


# ------------------------------------------------------------------------------------------------------------

def binned_stack_IDs(IDs, tab, spec_dir, vel_res_stack=50, fwhm_norm=True, flux_norm="conserve", use_rms=False,
                     fwhm_norm_med=260, mode="sum", verbose=True, outfile="stack"):
    ind = [i for i, ID in enumerate(tab["CRISTAL_ID_full"]) if ID in IDs]
    tab_sub = tab[ind]

    if verbose:
        print("Stacking the following sources:")
        print(np.array(tab_sub["CRISTAL_ID_full"]))

    vels, fluxes, errs, norm_values = [], [], [], []
    # for j in range(len(tab_sub)):
    for ID, z, fwhm_cii in zip(tab_sub["CRISTAL_ID_full"], tab_sub["z"], tab_sub["FWHM_CII"]):
        # ID, z, fwhm_cii = tab_sub["CRISTAL_ID_full"][j], tab_sub["z"][j], tab_sub["FWHM_CII"][j]
        vel, nu, flux, err = np.load(f"{work_dir}{spec_dir}{ID}_centered.npy")
        Spec = alma_spec(nu=nu, vel=vel, flux=flux, err=err, ID=ID, z=z)
        Spec.get_moments()
        Spec.get_rms()

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

    norm_med = np.nanmedian(norm_values)

    stack_vel, stack_flux, stack_err, stack_bins = variance_weighted_stack(vels, fluxes, errs,
                                                                           mode=mode, vel_res=vel_res_stack)
    stack_flux *= norm_med
    stack_err *= norm_med

    stack_vel_m = stack_vel.copy()[np.isnan(stack_flux) == False]
    stack_flux_m = stack_flux.copy()[np.isnan(stack_flux) == False]
    stack_err_m = stack_err.copy()[np.isnan(stack_flux) == False]
    stack_bins_m = stack_bins.copy()[np.isnan(stack_flux) == False]

    # Get median redshift of the subsample
    z_med = np.median(tab_sub["z"])
    Spec_stack = alma_spec(vel=stack_vel_m, flux=stack_flux_m, err=stack_err_m, z=z_med)  # generate spec class for
    # stack

    # Fit single-Gaussian model to the composite, using emcee
    popt, perr = Spec_stack.fit_gauss_emcee(bounds=[[np.max(Spec_stack.flux) / 2, -100, 50 / 2.35],
                                                    [np.max(Spec_stack.flux) * 1.5, 100, 400 / 2.35]],
                                            n_walkers=20
                                            )
    # Fit double-Gaussian model to the composite, using emcee
    popt_b, perr_b = Spec_stack.fit_gauss_broad_emcee(
        bounds=[[np.max(Spec_stack.flux) * 0.5, -200, 80 / 2.35, 0, -200, 400 / 2.35],
                [np.max(Spec_stack.flux) * 1.5, 200, 400 / 2.35, np.max(Spec_stack.flux) * 0.5, 200, 1000 / 2.35]],
        n_walkers=30  # 200
    )

    # Measure the BIC for both models, then work out which is best
    bic_single = bic(data=stack_flux, model=gaussian(stack_vel, *popt), error=stack_err, n_param=3)
    bic_double = bic(data=stack_flux, model=gaussian_double(stack_vel, *popt_b), error=stack_err, n_param=6)
    bic_result = bic_check(bic_single, bic_double)

    # Make a AlmaStack object
    Stack = AlmaStack(spec_dir=spec_dir, N_gal=len(IDs), ID_list=IDs, table=tab_sub,
                      fwhm_norm=fwhm_norm, flux_norm=flux_norm, stack_vel=stack_vel, stack_flux=stack_flux,
                      stack_err=stack_err, stack_bins=stack_bins, vels=vels, fluxes=fluxes, errs=errs,
                      norm_med=norm_med, popt=popt, perr=perr, popt_b=popt_b, perr_b=perr_b,
                      bic_single=bic_single, bic_double=bic_double, bic_result=bic_result, outfile=outfile)
    # Save the cristal_spec to a file
    with open(f'{outfile}.pkl', 'wb') as file:
        pickle.dump(Stack, file)

    return stack_vel_m, stack_flux_m, stack_err_m, stack_bins_m, vels, fluxes, errs, norm_values
