from defaults import colors
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from astropy.io import ascii
import pickle

from alma_stacking_pipeline.src.config_loader import work_dir
from alma_stacking_pipeline.src.utils import disable_ax_ticks
from alma_stacking_pipeline.src.line_models import gaussian, gaussian_double, line_model_single, line_model_double
from alma_stacking_pipeline.src.data.alma import make_alma_cube, make_alma_cube_cutout

# --------------------------------------------------------------------------------------------------

def plot_mask(mask, Cube, outfile):
    plt.figure()
    ax = plt.subplot(111, projection=Cube.wcs2d)
    disable_ax_ticks(ax, wcs=True)
    ax.imshow(mask)
    ax.text(0.02, 0.98, Cube.ID, ha="left", va="top", transform=ax.transAxes, color="w",
            fontsize=25)
    ax.text(0.02, 0.02, f"z={Cube.z:.3f}", ha="left", va="bottom",
            transform=ax.transAxes, color="w", fontsize=25)
    plt.savefig(outfile, bbox_inches="tight")
    plt.close()

def plot_stack(infile):
    # Load stack from the file
    try:
        with open(infile, "rb") as file:
            Stack = pickle.load(file)
    except FileNotFoundError:
        print(f"Error: File {infile} not found.")
        return
    except pickle.UnpicklingError:
        print(f"Error: Could not load {infile}. Invalid pickle format.")
        return

    # Get median redshift of the subsample
    z_med = np.median(Stack.table["z"])
    # Get median SFR of the subsample, ignoring those with no measurements
    sfr_med = 10 ** np.median(Stack.table["logSFR"][Stack.table["logSFR"] > 0])
    # Difference in BIC between two models
    delta_bic = Stack.bic_single - Stack.bic_double

    # --------------------------------------------------------------------------------------------------
    # Plot the results

    # Establish grid size from the list of sources
    nx = 3
    ny = 5 if Stack.N_gal > 9 else 3

    # Initiate the figure and gridspec
    plt.figure(figsize=(42, 14))
    gs_main = gridspec.GridSpec(1, 3, wspace=0, width_ratios=[0.33, 0.33, 0.33])

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
    ax_stack0 = plt.subplot(gs_stack[0])
    ax_stack0.set_xlim([v_min, v_max])
    ax_stack0.step(Stack.stack_vel, Stack.stack_bins, where="mid")
    ax_stack0.set_xticks([])
    ax_stack0.set_ylabel(r"N$_{\rm gal}$", fontsize=25)

    # Axis showing the stack
    ax_stack1= plt.subplot(gs_stack[1])
    ax_stack1.set_xlim([v_min, v_max])
    ax_stack1.set_ylim([np.min(Stack.stack_flux[(Stack.stack_vel > v_min) & (Stack.stack_vel < v_max)]) * 1.3,
                 np.max(Stack.stack_flux[(Stack.stack_vel > v_min) & (Stack.stack_vel < v_max)]) * 1.2])
    X = np.linspace(ax_stack1.get_xlim()[0], ax_stack1.get_xlim()[1], 1000)

    # Plot the stack and errors
    data, = ax_stack1.step(Stack.stack_vel, Stack.stack_flux, where="mid", lw=2.5)
    ax_stack1.step(Stack.stack_vel, Stack.stack_err, where="mid", c="k")
    # Display individual spectra used in the stack
    for vel, flux in zip(Stack.vels, Stack.fluxes):
        ax_stack1.step(vel, flux * Stack.norm_med, where="mid", c="gray", alpha=0.05)

    # Set linewidth of the two models depending on which is favoured
    lw_single = 3.5 if Stack.bic_result == "single" else 2
    lw_double = 3.5 if Stack.bic_result == "double" else 2
    # Plot the two models
    bestfit = gaussian(X, *Stack.popt)
    bestfit_b = gaussian_double(X, *Stack.popt_b)
    gauss = ax_stack1.errorbar(X, bestfit, c=colors[1], lw=lw_single, zorder=2)
    gauss_with_broad = ax_stack1.errorbar(X, bestfit_b, c=colors[2], lw=lw_double, zorder=2)
    # Plot the individual components of the double Gaussian
    ax_stack1.errorbar(X, gaussian(X, *Stack.popt_b[:3]), c=colors[2], lw=lw_double, ls="--", alpha=0.5, zorder=2)
    ax_stack1.errorbar(X, gaussian(X, *Stack.popt_b[3:]), c=colors[2], lw=lw_double, ls="--", alpha=0.5, zorder=2)

    # Add zeros lines
    ax_stack1.axhline(0, c="k", ls="-", alpha=0.5)
    ax_stack1.axvline(0, c="k", ls="--", alpha=0.5)
    ax_stack1.set_ylabel(r"Flux density / mJy", fontsize=25)

    # Add some informative labels and a legend
    ax_stack1.text(0.02, 0.98, "Variance-weighted stack", ha="left", va="top",
            transform=ax_stack1.transAxes, color="k", fontsize=25)
    ax_stack1.text(0.02, 0.93, f"{Stack.N_gal:.0f} galaxies", ha="left", va="top",
            transform=ax_stack1.transAxes, color="k", fontsize=25)
    ax_stack1.text(0.02, 0.88, rf"$\Delta$BIC = {delta_bic:.1f}", ha="left", va="top",
            transform=ax_stack1.transAxes, color="k", fontsize=25)
    ax_stack1.legend(loc=1, handles=[data, gauss, gauss_with_broad], labels=["Data", "Gaussian", "Gaussian+broad"],
              fontsize=25)

    # If no broad component is required, don't need to do much
    if Stack.bic_result == "single":
        fwhm_narrow = Stack.popt[2] * 2 * (2 * np.log(2)) ** 0.5
        ax_stack1.text(0.98, 0.7, rf"FWHM = {fwhm_narrow:.0f} kms$^{{-1}}$", ha="right",
                va="top", transform=ax_stack1.transAxes, color="k", fontsize=25)
        line_model = line_model_single(amp1=Stack.popt[0], amp1_err=Stack.perr[0],
                                       mean1=Stack.popt[1], mean1_err=Stack.perr[1],
                                       sig1=Stack.popt[2], sig1_err=Stack.perr[2], z=z_med, sfr=sfr_med)
        line_model.get_line_props()
        line_model.get_errors()

    # If a broad component IS required, calculate the outflow properties,
    elif Stack.bic_result == "double":
        fwhm_narrow = Stack.popt_b[2] * 2 * (2 * np.log(2)) ** 0.5
        fwhm_broad = Stack.popt_b[5] * 2 * (2 * np.log(2)) ** 0.5
        # Generate a line_model_double class and calculate the outflow properties
        line_model = line_model_double(amp1=Stack.popt_b[0], amp1_err=Stack.perr_b[0], amp2=Stack.popt_b[3],
                                       amp2_err=Stack.perr_b[3], mean1=Stack.popt_b[1], mean1_err=Stack.perr_b[1], mean2=Stack.popt_b[4],
                                       mean2_err=Stack.perr_b[4], sig1=Stack.popt_b[2], sig1_err=Stack.perr_b[2], sig2=Stack.popt_b[5],
                                       sig2_err=Stack.perr_b[5], z=z_med, sfr=sfr_med)
        line_model.get_line_props()
        line_model.get_wing_props(X)
        line_model.get_outflow_props()
        line_model.get_errors()

        # Add some informative labels
        ax_stack1.text(0.98, 0.77, rf"FWHM$_{{\rm narrow}}$ = {fwhm_narrow:.0f} kms$^{{-1}}$",
                ha="right", va="top", transform=ax_stack1.transAxes, color="k", fontsize=25)
        ax_stack1.text(0.98, 0.72, rf"FWHM$_{{\rm broad}}$ = {fwhm_broad:.0f} kms$^{{-1}}$",
                ha="right", va="top", transform=ax_stack1.transAxes, color="k", fontsize=25)
        ax_stack1.text(0.98, 0.67, rf"$\dot{{M}}_{{\rm out}}$ = {line_model.m_out_dot:.0f} M$_\odot$yr$^{{-1}}$",
                ha="right", va="top", transform=ax_stack1.transAxes, color="k", fontsize=25)
        ax_stack1.text(0.98, 0.62, rf"$\eta_m$ = {line_model.mass_load_factor:.2f}",
                ha="right", va="top", transform=ax_stack1.transAxes, color="k", fontsize=25)

    # Calculate the residuals from both fits
    resid = Stack.stack_flux - gaussian(Stack.stack_vel, *Stack.popt)
    resid_b = Stack.stack_flux - gaussian_double(Stack.stack_vel, *Stack.popt_b)

    # Axis showing the residuals
    ax_stack2 = plt.subplot(gs_stack[2])
    ax_stack2.set_xlim([v_min, v_max])
    ax_stack2.set_ylim([-0.1*Stack.norm_med, 0.1*Stack.norm_med])
    ax_stack2.step(Stack.stack_vel, resid, where="mid", c=colors[1], lw=2)
    ax_stack2.step(Stack.stack_vel, resid_b, where="mid", c=colors[2], lw=2)
    ax_stack2.step(Stack.stack_vel, 3 * Stack.stack_err, where="mid", ls="--", color="k")
    ax_stack2.fill_between(Stack.stack_vel, resid, color=colors[1], alpha=0.3, step="mid")
    ax_stack2.axhline(0, c="k", ls="--", alpha=0.5)
    ax_stack2.set_xlabel(r"Velocity / kms$^{-1}$", fontsize=25)
    ax_stack2.set_ylabel(r"Residual / mJy", fontsize=25)

    # --------------------------------------------------------------------------------------------------
    # Centre panel: [CII] maps

    for i in range(len(Stack.ID_list)):
        # Identify table row to use, get the correct file and redshift/coordinates
        j = np.where(Stack.table["CRISTAL_ID_full"] == Stack.ID_list[i])[0][0]
        file = f"{work_dir}data/cubes/{Stack.table['File'][i]}".replace("_fake", "")
        z, ra, dec, fwhm_cii = Stack.table["z"][j], Stack.table["RA"][j], Stack.table["Dec"][j], Stack.table[
            "FWHM_CII"][i]

        # Load cube and collapse to 2-D
        Cube = make_alma_cube(file, Stack.ID_list[i], z, ra, dec, fwhm_cii)
        Cube_cutout = make_alma_cube_cutout(Cube, size=4, centre=Cube.c)
        Cube_cutout.convert_to_jypixel()
        flux_map = Cube_cutout.get_flux_map()

        # Mask to show which regions of the galaxy are being used
        mask = np.load(f"{work_dir}data/masks/contours/{Cube.ID}_mask.npy")
        flux_map_masked = flux_map.copy()
        flux_map_masked[mask == False] = np.nan
        # Mask to show which regions are considered "galaxy", but are not used in the stack
        mask_gal = np.load(f"{work_dir}data/masks/contours/{Cube.ID}_mask.npy")
        flux_map_masked_gal = flux_map.copy()
        flux_map_masked_gal[mask_gal == False] = np.nan  # Remove non-galaxy regions
        flux_map_masked_gal[
            (mask_gal == True) & (mask == False)] = 1  # Galaxy regions that aren't used in the stack
        flux_map_masked_gal[(mask_gal == True) & (mask == True)] = np.nan  # Remove regions used in the stack

        # Add to plot
        ax = plt.subplot(gs_maps[i], projection=Cube_cutout.wcs2d)
        disable_ax_ticks(ax, wcs=True)
        # Main flux map
        ax.imshow(flux_map, cmap="plasma", zorder=1, alpha=0.5)
        # Map of galaxies regions that aren't used in the stack (grey)
        ax.imshow(flux_map_masked_gal, cmap="gray", zorder=1.5, alpha=0.5, aspect="auto")
        # Regions that ARE used in the stack
        ax.imshow(flux_map_masked, cmap="plasma", zorder=2, aspect="auto")
        ax.text(0.02, 0.98, Stack.ID_list[i], ha="left", va="top", transform=ax.transAxes,
                color="k", fontsize=15)
        ax.set_adjustable("datalim")

    # --------------------------------------------------------------------------------------------------
    # Right panel: individual [CII] spectra

    for i in range(len(Stack.ID_list)):
        # Load individual spectra
        vel, nu, flux, err = np.load(f"{work_dir}{Stack.spec_dir}{Stack.ID_list[i]}_centered.npy")

        ax = plt.subplot(gs_spec[i])
        ax.set_xlim([v_min, v_max])  # use same x limits as stack plot
        ax.set_ylim([-0.5, 1.4])
        ax.step(vel, flux / np.max(flux), where="mid", lw=1.5)  # normalize spectra so they match with stack
        ax.text(0.02, 0.98, Stack.ID_list[i], ha="left", va="top", transform=ax.transAxes, fontsize=15)
        ax.axhline(0, c="k")  # show zero lines for clarity
        ax.axvline(0, c="k")
        ax.set_xticks([])  # remove tick labels to avoid clutter
        ax.set_yticks([])

    plt.savefig(f"{work_dir}data/stacks/tmp.pdf", bbox_inches="tight", dpi=600)
    plt.close()