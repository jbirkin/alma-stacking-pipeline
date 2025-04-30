import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MultipleLocator
import os

from src.utils.tables import cristal_dir, ctab, cube_dir, nu_cii
from src.utils.cristal_cube import cristal_spec
from src.utils.line_models import gaussian
from src.utils.stacking import variance_weighted_stack

# ------------------------------------------------------------------------------------------------------------

# spec_type = "mask_high_sfr_vel_res_med"

def stack_resids(spec_type):
    spec_dir = f"{cristal_dir}spectra/{spec_type}/"
    norm_values = []
    vels, fluxes, errs = [], [], []

    for i in range(len(ctab)):
        ID = ctab["CRISTAL_ID_full"][i]
        if ctab["in_stack?"][i]==False or os.path.exists(f"{spec_dir}{ID}_centered.npy")==False:
            continue
        z, ra, dec = ctab["z"][i], ctab["RA"][i], ctab["Dec"][i]
        vel, nu, flux, err = np.load(f"{spec_dir}{ID}_centered.npy", allow_pickle=True)
        norm_values.append(np.nanmax(flux))
        flux /= np.nanmax(flux)

        Spec = cristal_spec(vel=vel, nu=nu, flux=flux, err=err, z=z)

        popt, perr = Spec.fit_gauss()
        Spec.get_residuals()

        # plt.figure()
        # plt.step(Spec.vel, Spec.resid)
        # plt.show()

        np.save(f"{cristal_dir}spectra/resid_{spec_type}/{ID}_centered.npy", np.array([vel, nu, Spec.resid, err]))

    for j in range(len(ctab)):
        ID, z = ctab["CRISTAL_ID_full"][j], ctab["z"][j]
        if not ctab["in_stack?"][j] or not os.path.exists(f"{spec_dir}{ID}_centered.npy"):
            continue
        vel, nu, flux, err = np.load(f"{cristal_dir}spectra/resid_{spec_type}/{ID}_centered.npy")
        Spec = cristal_spec(nu=nu, vel=vel, flux=flux, err=err, ID=ID, z=z)
        Spec.get_rms()

        vels.append(Spec.vel)
        fluxes.append(Spec.flux)
        errs.append(Spec.rms)

    vels = np.array(vels, dtype=object)
    fluxes = np.array(fluxes, dtype=object)
    errs = np.array(errs, dtype=object)

    stack_vel, stack_flux, stack_err, stack_bins = variance_weighted_stack(vels, fluxes, errs,
                                                                           mode="sum", vel_res=50)

    return stack_vel, stack_flux, stack_err, stack_bins

stack_vel_2sig, stack_flux_2sig, stack_err_2sig, stack_bins_2sig = stack_resids("contours_2sig_vel_res_med")
stack_vel_lowsfr, stack_flux_lowsfr, stack_err_lowsfr, stack_bins_lowsfr = stack_resids("mask_low_sfr_vel_res_med")
stack_vel_highsfr, stack_flux_highsfr, stack_err_highsfr, stack_bins_highsfr = stack_resids("mask_high_sfr_vel_res_med")

xmin, xmax = -800, 800
plt.figure(figsize=(15,15))
gs = gridspec.GridSpec(nrows=3, ncols=1, hspace=0)

ax0 = plt.subplot(gs[0])
ax0.set_xlim([xmin,xmax])
ax0.set_ylim([-0.12,0.12])
ax0.step(stack_vel_2sig, stack_flux_2sig, where="mid")
# ax0.step(stack_vel, stack_err)
ax0.axhline(0, c="k", alpha=0.7)
ax0.text(0.02, 0.98, "Full sample (15 galaxies)", ha="left", va="top", transform=ax0.transAxes)
ax0.yaxis.set_major_locator(MultipleLocator(0.05))

ax1 = plt.subplot(gs[1])
ax1.set_xlim([xmin,xmax])
ax1.set_ylim([-0.12,0.12])
ax1.step(stack_vel_lowsfr, stack_flux_lowsfr, where="mid")
# ax1.step(stack_vel, stack_err)
ax1.axhline(0, c="k", alpha=0.7)
ax1.text(0.02, 0.98, "Low-SFR sample (8 galaxies)", ha="left", va="top", transform=ax1.transAxes)
ax1.yaxis.set_major_locator(MultipleLocator(0.05))

ax2 = plt.subplot(gs[2])
ax2.set_xlim([xmin,xmax])
ax2.set_ylim([-0.12,0.12])
ax2.step(stack_vel_highsfr, stack_flux_highsfr, where="mid")
# ax2.step(stack_vel, stack_err)
ax2.axhline(0, c="k", alpha=0.7)
ax2.text(0.02, 0.98, "High-SFR sample (9 galaxies)", ha="left", va="top", transform=ax2.transAxes)
ax2.yaxis.set_major_locator(MultipleLocator(0.05))

ax2.set_xlabel(r"Velocity / kms$^{-1}$", fontsize=25)
for ax in [ax0,ax1,ax2]:
    ax.set_ylabel(r"Residual / mJy", fontsize=25)
plt.savefig(f"{cristal_dir}stacks/resids.pdf")
plt.close()