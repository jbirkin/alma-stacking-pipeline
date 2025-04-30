import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pickle
import sys

from astro_cubes.defaults import colors
from astro_cubes.tools import find_nearest, disable_ax_ticks
from src.utils.line_models import gaussian, gaussian_double, line_model_single, line_model_double
from src.utils.tables import ctab, cristal_dir, cube_dir
from src.utils.cristal_cube import make_cristal_cube, make_cristal_cube_cutout

outfile = f"{sys.argv[1]}"

# Load stack from the file
with open(f'{outfile}.pkl', 'rb') as file:
    Stack = pickle.load(file)

spec_dir = Stack.spec_dir
N_gal = Stack.N_gal
ID_list = Stack.ID_list
bad_list = Stack.bad_list
ctab_sub = Stack.table
fwhm_norm = Stack.fwhm_norm
flux_norm = Stack.flux_norm
stack_vel = Stack.stack_vel
stack_flux = Stack.stack_flux
stack_err = Stack.stack_err
stack_bins = Stack.stack_bins
vels = Stack.vels
fluxes = Stack.fluxes
errs = Stack.errs
norm_med = Stack.norm_med
popt = Stack.popt
perr = Stack.perr
popt_b = Stack.popt_b
perr_b = Stack.perr_b
bic_single = Stack.bic_single
bic_double = Stack.bic_double
bic_result = Stack.bic_result
outfile = Stack.outfile

# Get median redshift of the subsample
z_med = np.median(ctab_sub["z"])
# Get median SFR of the subsample, ignoring those with no measurements
sfr_med = 10**np.median(ctab_sub["logSFR"][ctab_sub["logSFR"]>0])
sfr16 = np.nanpercentile(10**ctab_sub["logSFR"][ctab_sub["logSFR"]>0], 16)
sfr84 = np.nanpercentile(10**ctab_sub["logSFR"][ctab_sub["logSFR"]>0], 84)
# Get median stellar mass of the subsample, ignoring those with no measurements
mstar_med = 10**np.median(ctab_sub["logMstar"][ctab_sub["logMstar"]>0])
mstar16 = np.nanpercentile(10**ctab_sub["logMstar"][ctab_sub["logMstar"]>0], 16)
mstar84 = np.nanpercentile(10**ctab_sub["logMstar"][ctab_sub["logMstar"]>0], 84)

# --------------------------------------------------------------------------------------------------
# Plot the results

# Establish grid size from the list of sources
nx = 3
ny = 5 if N_gal > 12 else 4 if N_gal > 9 else 3
stack_only = False

# Initiate the figure and gridspec
if stack_only:
    fig = plt.figure(figsize=(14, 14))
    gs_stack = gridspec.GridSpec(3, 1, height_ratios=[0.2, 1, 0.2], hspace=0)
else:
    fig = plt.figure(figsize=(42, 14))
    gs_main = gridspec.GridSpec(1, 3, wspace=0, width_ratios=[0.33,0.33,0.33])

    # Three sub grids: 1) Stack 2) Individual [CII] maps 3) Individual [CII] spectra
    gs_stack = gridspec.GridSpecFromSubplotSpec(3, 1, gs_main[0], height_ratios=[0.2, 1, 0.2], hspace=0)
    gs_maps = gridspec.GridSpecFromSubplotSpec(nrows=nx, ncols=ny, subplot_spec=gs_main[1],
                                               wspace=0, hspace=0)
    gs_spec = gridspec.GridSpecFromSubplotSpec(nrows=nx, ncols=ny, subplot_spec=gs_main[2],
                                               wspace=0, hspace=0)

# --------------------------------------------------------------------------------------------------
# Left panel: stacks

# Choose appropriate plotting range
v_min, v_max = -800, 800

# Axis showing the number of galaxies per bin
ax = plt.subplot(gs_stack[0])
ax.set_xlim([v_min, v_max])
ax.step(stack_vel, stack_bins, where="mid")
ax.set_xticks([])
ax.set_ylabel(r"N$_{\rm gal}$", fontsize=25)

# Axis showing the stack
ax = plt.subplot(gs_stack[1])
ax.set_xlim([v_min, v_max])
ax.set_ylim([np.min(stack_flux[(stack_vel > v_min) & (stack_vel < v_max)]) * 1.3,
             np.max(stack_flux[(stack_vel > v_min) & (stack_vel < v_max)]) * 1.2])
X = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 1000)

# Plot the stack and errors
data, = ax.step(stack_vel, stack_flux, where="mid", lw=2.5)
ax.step(stack_vel, stack_err, where="mid", c="k")
# Display individual spectra used in the stack
for i in range(np.shape(vels)[0]):
    ax.step(vels[i], fluxes[i] * norm_med, where="mid", c="gray", alpha=0.05)

# Set linewidth of the two models depending on which is favoured
lw_single = 3.5 if bic_result == "single" else 2
lw_double = 3.5 if bic_result == "double" else 2
# Plot the two models
bestfit = gaussian(X, *popt)
bestfit_b = gaussian_double(X, *popt_b)
gauss = ax.errorbar(X, bestfit, c=colors[1], lw=lw_single, zorder=2)
gauss_with_broad = ax.errorbar(X, bestfit_b, c=colors[2], lw=lw_double, zorder=2)
# Plot the individual components of the double Gaussian
ax.errorbar(X, gaussian(X, *popt_b[:3]), c=colors[2], lw=lw_double, ls="--", alpha=0.5, zorder=2)
ax.errorbar(X, gaussian(X, *popt_b[3:]), c=colors[2], lw=lw_double, ls="--", alpha=0.5, zorder=2)

# Add zeros lines
ax.axhline(0, c="k", ls="-", alpha=0.5)
ax.axvline(0, c="k", ls="--", alpha=0.5)
ax.set_ylabel(r"Flux density / mJy", fontsize=25)

# Add some informative labels and a legend
ax.text(0.02, 0.98, "Variance-weighted stack", ha="left", va="top",
        transform=ax.transAxes, color="k", fontsize=25)
ax.text(0.02, 0.93, "{:.0f} galaxies".format(N_gal), ha="left", va="top",
        transform=ax.transAxes, color="k", fontsize=25)
ax.text(0.02, 0.88, r"$\Delta$BIC = {:.1f}".format(bic_single - bic_double), ha="left", va="top",
        transform=ax.transAxes, color="k", fontsize=25)
ax.legend(loc=1, handles=[data, gauss, gauss_with_broad], labels=["Data", "Gaussian", "Gaussian+broad"], fontsize=25)

# If no broad component is required, don't need to do much
if bic_result == "single":
    ax.text(0.98, 0.7, r"FWHM = {:.0f} kms$^{{-1}}$".format(popt[2]*2*(2*np.log(2))**0.5), ha="right",
            va="top",
            transform=ax.transAxes, color="k", fontsize=25)
    line_model = line_model_single(amp1=popt[0], amp1_err=perr[0],
                                   mean1=popt[1], mean1_err=perr[1],
                                   sig1=popt[2], sig1_err=perr[2], z=z_med, sfr=sfr_med)
    line_model.get_line_props()
    line_model.get_errors()

# If a broad component IS required, calculate the outflow properties,
elif bic_result == "double":
    # Generate a line_model_double class and calculate the outflow properties
    line_model = line_model_double(amp1=popt_b[0], amp1_err=perr_b[0], amp2=popt_b[3],
                    amp2_err=perr_b[3], mean1=popt_b[1], mean1_err=perr_b[1], mean2=popt_b[4],
                    mean2_err=perr_b[4], sig1=popt_b[2], sig1_err=perr_b[2], sig2=popt_b[5],
                    sig2_err=perr_b[5], z=z_med, sfr=sfr_med)
    line_model.get_line_props()
    line_model.get_wing_props(X)
    line_model.get_outflow_props()
    line_model.get_errors()

    # Highlight the wings used to derive mass outflow rates and velocity
    # ax.fill_between(X, np.zeros(len(X)), line_model.broad_wings_only, color=colors[2], zorder=3, ls="--", alpha=0.5)

    # Add some informative labels
    ax.text(0.98, 0.77, r"FWHM$_{{\rm narrow}}$ = {:.0f} kms$^{{-1}}$".format(popt_b[2]*2*(2*np.log(2))**0.5),
            ha="right", va="top", transform=ax.transAxes, color="k", fontsize=25)
    ax.text(0.98, 0.72, r"FWHM$_{{\rm broad}}$ = {:.0f} kms$^{{-1}}$".format(popt_b[5]*2*(2*np.log(2))**0.5),
            ha="right", va="top", transform=ax.transAxes, color="k", fontsize=25)
    ax.text(0.98, 0.67, r"$\dot{{M}}_{{\rm out}}$ = {:.0f} M$_\odot$yr$^{{-1}}$".format(line_model.m_out_dot),
            ha="right", va="top", transform=ax.transAxes, color="k", fontsize=25)
    ax.text(0.98, 0.62, r"$\eta_m$ = {:.2f}".format(line_model.mass_load_factor),
            ha="right", va="top", transform=ax.transAxes, color="k", fontsize=25)

# Calculate the residuals from both fits
resid = stack_flux - gaussian(stack_vel, *popt)
resid_b = stack_flux - gaussian_double(stack_vel, *popt_b)

# Axis showing the residuals
ax = plt.subplot(gs_stack[2])
ax.set_xlim([v_min, v_max])
ax.set_ylim([-0.1 * norm_med, 0.1 * norm_med])  # 1.1*np.min(resid), 1.1*np.max(resid)])
ax.step(stack_vel, resid, where="mid", c=colors[1], lw=2)
ax.step(stack_vel, resid_b, where="mid", c=colors[2], lw=2)
ax.step(stack_vel, 3 * stack_err, where="mid", ls="--", color="k")
ax.fill_between(stack_vel, resid, color=colors[1], alpha=0.3, step="mid")
ax.axhline(0, c="k", ls="--", alpha=0.5)
ax.set_xlabel(r"Velocity / kms$^{-1}$", fontsize=25)
ax.set_ylabel(r"Residual / mJy", fontsize=25)

# --------------------------------------------------------------------------------------------------
# Centre panel: [CII] maps

# Set to the chosen mask sigma. Will be used to determin which regions of the maps are greyed out
nsig_gal = 3

if not stack_only:
    for i in range(len(ID_list)):
        # Identify table row to use, get the correct file and redshift/coordinates
        j = np.where(ctab["CRISTAL_ID_full"]==ID_list[i])[0][0]
        file = cube_dir+ctab["File"][j]
        file = file.replace("10kms", "{}kms".format(ctab["vel_res_med"][i]))
        z, ra, dec = ctab["z"][j], ctab["RA"][j], ctab["Dec"][j]
        fwhm_cii = ctab["FWHM_CII"][i]

        # Load cube and collapse to 2-D
        Cube = make_cristal_cube(file, ID_list[i], z, ra, dec, fwhm_cii)
        Cube_cutout = make_cristal_cube_cutout(Cube, size=4, centre=Cube.c)
        Cube_cutout.convert_to_jypixel()
        flux_map = Cube_cutout.get_flux_map()

        # Mask to show which regions of the galaxy are being used
        mask = np.load(cristal_dir+"data/masks/"+spec_dir+Cube.ID+"_mask.npy")
        flux_map_masked = flux_map.copy()
        flux_map_masked[mask==False] = np.nan
        # Mask to show which regions are considered "galaxy", but are not used in the stack
        mask_gal = np.load(f"{cristal_dir}data/masks/contours_{nsig_gal}sig_vel_res_med/{Cube.ID}_mask.npy")
        flux_map_masked_gal = flux_map.copy()
        flux_map_masked_gal[mask_gal==False] = np.nan                   # Remove non-galaxy regions
        flux_map_masked_gal[(mask_gal==True) & (mask==False)] = 1       # Galaxy regions that aren't used in the stack
        flux_map_masked_gal[(mask_gal==True) & (mask==True)] = np.nan   # Remove regions used in the stack

        # Add to plot
        ax = plt.subplot(gs_maps[i], projection=Cube_cutout.wcs2d)
        disable_ax_ticks(ax, wcs=True)
        # Main flux map
        ax.imshow(flux_map, cmap="plasma", zorder=1, alpha=0.5)
        # Map of galaxies regions that aren't used in the stack (grey)
        ax.imshow(flux_map_masked_gal, cmap="gray", zorder=1.5, alpha=0.5, aspect="auto")
        # Regions that ARE used in the stack
        im = ax.imshow(flux_map_masked, cmap="plasma", zorder=2, aspect="auto")
        ax.text(0.02, 0.98, ID_list[i], ha="left", va="top", transform=ax.transAxes,
                color="k", fontsize=15)
        ax.set_adjustable("datalim")

# --------------------------------------------------------------------------------------------------
# Right panel: individual [CII] spectra

if not stack_only:
    for i in range(len(ID_list)):
        # Load individual spectra
        vel, nu, flux, err = np.load(cristal_dir+"spectra/"+spec_dir+ID_list[i]+"_centered.npy")

        ax = plt.subplot(gs_spec[i])
        ax.set_xlim([v_min, v_max])     # use same x limits as stack plot
        ax.set_ylim([-0.5, 1.4])
        ax.step(vel, flux/np.max(flux), where="mid", lw=1.5)        # normalize spectra so they match with stack
        ax.text(0.02, 0.98, ID_list[i], ha="left", va="top", transform=ax.transAxes, fontsize=15)
        ax.axhline(0, c="k")        # show zero lines for clarity
        ax.axvline(0, c="k")
        ax.set_xticks([])           # remove tick labels to avoid clutter
        ax.set_yticks([])

# --------------------------------------------------------------------------------------------------

# Save a .txt file with the results in an easily readable format
# This is somewhat obsolete as I will read results from the .npy file when plotting
# It is useful for easily checking results however
results = "Using directory: {}\n\n{} sources: {}\n\nExcluded: {}\n\nMedian redshift: {:.3f}\nMedian " \
          "SFR: {:.1f}\n\nBIC (single): {:.1f}, BIC (double): {:.1f}, deltaBIC: {:.1f}, preferred model: {}".format(
    spec_dir, len(ID_list), ID_list, bad_list, z_med, sfr_med, bic_single, bic_double, bic_single-bic_double,
    bic_result.upper())
with open(cristal_dir+"stacks/"+outfile+"_results.txt", "w") as text_file:
    text_file.write(results)

# Save a .npy file with the results in a way that can be read for plotting
results = {"stack_name": spec_dir, "N_gal": len(ID_list), "z_med": z_med, "SFR_med": sfr_med,
           "SFR_16": sfr16, "SFR_84": sfr84, "SFR_err_l": sfr_med-sfr16, "SFR_err_u": sfr84-sfr_med,
           "Mstar_med": mstar_med, "Mstar_16": mstar16, "Mstar_84": mstar84,
           "Mstar_err_l": mstar_med-mstar16, "Mstar_err_u": mstar84-mstar_med,
           "BIC_single": bic_single, "BIC_double": bic_double, "delta_BIC": bic_single-bic_double,
           "BIC_result": bic_result.upper(),
           "SFR_CII_lagache": line_model.sfr_cii,
           "SFR_CII_lagache_err": line_model.sfr_cii_err}

# Add outflow properties to the results file
if bic_result.lower() == "double":
    results["FWHM_narrow"] = popt_b[2]*2*(2*np.log(2))**0.5
    results["FWHM_narrow_err"] = perr_b[2]*2*(2*np.log(2))**0.5
    results["FWHM_broad"] = popt_b[5]*2*(2*np.log(2))**0.5
    results["FWHM_broad_err"] = perr_b[5]*2*(2*np.log(2))**0.5
    results["M_out_dot"] = line_model.m_out_dot
    results["M_out_dot_err"] = line_model.m_out_dot_err
    results["M_out_dot_wings"] = line_model.m_out_dot_wings
    results["M_out_dot_wings_err"] = line_model.m_out_dot_wings_err
    results["load_factor"] = line_model.mass_load_factor
    results["load_factor_err"] = line_model.mass_load_factor_err
    results["load_factor_wings"] = line_model.mass_load_factor_wings
    results["load_factor_wings_err"] = line_model.mass_load_factor_wings_err
    results["out_frac"] = line_model.out_frac
    results["out_frac_err"] = line_model.out_frac_err
    results["E_out_dot"] = line_model.e_out_dot
    results["E_out_dot_err"] = line_model.e_out_dot_err
    results["v_out"] = line_model.v_out
    results["v_out_err"] = line_model.v_out_err
# Or just set -99. if no broad component required
else:
    results["FWHM_narrow"] = popt[2]*2*(2*np.log(2))**0.5
    results["FWHM_narrow_err"] = perr[2]*2*(2*np.log(2))**0.5
    results["FWHM_broad"] = -99.
    results["FWHM_broad_err"] = -99.
    results["M_out_dot"] = -99.
    results["M_out_dot_err"] = -99.
    results["load_factor"] = -99.
    results["load_factor_err"] = -99.
    results["out_frac"] = -99.
    results["out_frac_err"] = -99.
# Save the file
np.save(cristal_dir+"stacks/"+outfile+"_results.npy", results)

# Save the plot
np.save(cristal_dir+"stacks/"+outfile+".npy",
        np.array([stack_vel, stack_flux, stack_err, stack_bins, vels, fluxes, errs], dtype="object"))
plt.savefig(cristal_dir+"stacks/"+outfile+".pdf", bbox_inches="tight", dpi=600)
plt.close()