import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MultipleLocator
from scipy.optimize import curve_fit
from astropy.coordinates import SkyCoord
import astropy.units as u
import sys
import glob
import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")

from astro_cubes.defaults import colors
from astro_cubes.tools import rebin_single, disable_ax_ticks, boot_median, moments, find_nearest
from src.utils.cristal_cube import cristal_cube, make_cristal_cube, make_cristal_cube_cutout, mask_cristal_cube
from src.utils.cristal_cube import cristal_spec
from src.utils.flux_extraction import extract_from_sig_mask, get_errors
from src.utils.tables import cristal_dir, cube_dir, ctab, ctab_file, nu_cii
from src.utils.line_models import gaussian, gaussian_double
from src.utils.stacking import variance_weighted_stack

# --------------------------------------------------------------------------------------------------

vel_res = "med"
# spec_dir = "spectra/contours_3sig_vel_res_" + vel_res + "/"
spec_dir = "spectra/"+str(sys.argv[1])+"/"
filelist = glob.glob(cristal_dir+spec_dir+"*.npy")

# --------------------------------------------------
# make individual spectra

for i in range(len(ctab)):
    if not ctab["Detect?"][i]:
        continue
    ID = ctab["CRISTAL_ID_full"][i]
    if cristal_dir+spec_dir+ID+".npy" not in filelist:
        continue
    file = cube_dir + ctab["File"][i]
    z, ra, dec = ctab["z"][i], ctab["RA"][i], ctab["Dec"][i]
    fwhm_cii = ctab["FWHM_CII"][i]

    Cube = make_cristal_cube(file, ID, z, ra, dec, fwhm_cii)
    vel, nu, flux, err = np.load(cristal_dir + spec_dir + ID + ".npy")
    Spec = cristal_spec(vel=vel, flux=flux, err=err, z=z)
    # fit an approximate Gaussian
    popt, pcov = curve_fit(gaussian, vel, flux,
                           p0=[np.max(flux), 0, 100],
                           bounds=[[np.max(flux)/2, -100, 50],
                                   [np.max(flux), 100, 150]])
    # use to define limits for calculating moments
    i_min = find_nearest(vel, popt[1] - 1.5 * 2.35 * popt[2])[0]
    i_max = find_nearest(vel, popt[1] + 1.5 * 2.35 * popt[2])[0]

    M0, dM0, M1, dM1, M2, dM2 = moments(vel[i_min:i_max],
                                        flux[i_min:i_max],
                                        err[i_min:i_max])
    Spec.get_rms()

    # velocity space
    fig = plt.figure(figsize=(7.5,7.5))
    ax = plt.subplot(111)
    ax.set_xlim([-1000,1000])
    ax.set_ylim([np.nanmin(flux) * 1.2, np.nanmax(flux) * 1.5])
    X = np.linspace(-1000, 1000, 1000)
    ax.step(vel, flux, where="mid")
    ax.step(vel, err, where="mid")
    ax.step(vel, Spec.rms, where="mid", c=colors[3])
    ax.axvline(0, c="k", ls="--", alpha=0.5)
    ax.axvline(-M2*2.35/2+M1, c=colors[0], lw=3, ls="--")
    ax.axvline(M2*2.35/2+M1, c=colors[0], lw=3, ls="--")
    ax.text(0.98, 0.98, "M0={:.1f}±{:.1f}".format(M0, dM0), ha="right", va="top", transform=ax.transAxes)
    ax.text(0.98, 0.93, "M1={:.1f}±{:.1f}".format(M1, dM2), ha="right", va="top", transform=ax.transAxes)
    ax.text(0.98, 0.88, "M2={:.1f}±{:.1f}".format(M2, dM2), ha="right", va="top", transform=ax.transAxes)
    ax.text(0.02, 0.98, Cube.ID + " (z={:.3f})".format(Cube.z), ha="left", va="top",
            transform=ax.transAxes, color="k", fontsize=17.5)
    ax.set_xlabel(r"Velocity / kms$^{-1}$")
    ax.set_ylabel(r"Flux density / mJy")
    plt.savefig(cristal_dir + spec_dir + ID + ".pdf", bbox_inches="tight")
    plt.close()

    # frequency space
    fig = plt.figure(figsize=(7.5,7.5))
    ax = plt.subplot(111)
    ax.set_xlim([nu.min(),nu.max()])
    ax.set_ylim([np.nanmin(flux) * 1.2, np.nanmax(flux) * 1.5])
    X = np.linspace(nu.min(),nu.max(), 1000)
    ax.step(nu, flux, where="mid")
    ax.step(nu, err, where="mid")
    ax.step(nu, Spec.rms, where="mid", c=colors[3])
    ax.axvline(0, c="k", ls="--", alpha=0.5)
    # ax.axvline(-M2*2.35/2+M1, c=colors[0], lw=3, ls="--")
    # ax.axvline(M2*2.35/2+M1, c=colors[0], lw=3, ls="--")
    # ax.text(0.98, 0.98, "M0={:.1f}±{:.1f}".format(M0, dM0), ha="right", va="top", transform=ax.transAxes)
    # ax.text(0.98, 0.93, "M1={:.1f}±{:.1f}".format(M1, dM2), ha="right", va="top", transform=ax.transAxes)
    # ax.text(0.98, 0.88, "M2={:.1f}±{:.1f}".format(M2, dM2), ha="right", va="top", transform=ax.transAxes)
    ax.text(0.02, 0.98, Cube.ID + " (z={:.3f})".format(Cube.z), ha="left", va="top",
            transform=ax.transAxes, color="k", fontsize=17.5)
    ax.set_xlabel(r"Frequency / GHz")
    ax.set_ylabel(r"Flux density / mJy")
    plt.savefig(cristal_dir + spec_dir + ID + "_freq.pdf", bbox_inches="tight")
    plt.close()

# --------------------------------------------------

# --------------------------------------------------
# make plot of all spectra

nx, ny = 7, 5
fig = plt.figure(figsize=(22.5*nx/5,22.5*ny/5))
gs = gridspec.GridSpec(ny, nx, wspace=0, hspace=0)

ctab_det = ctab[ctab["Detect?"]]
for i in range(len(ctab_det)):
    ID = ctab_det["CRISTAL_ID_full"][i]
    if cristal_dir+spec_dir+ID+"_centered.npy" not in filelist:
        continue
    file = cube_dir + ctab_det["File"][i]
    z, ra, dec = ctab_det["z"][i], ctab_det["RA"][i], ctab["Dec"][i]
    fwhm_cii = ctab["FWHM_CII"][i]

    Cube = make_cristal_cube(file, ID, z, ra, dec, fwhm_cii)
    vel, nu, flux, err = np.load(cristal_dir + spec_dir + ID + "_centered.npy")
    Spec = cristal_spec(vel=vel, flux=flux, err=err, z=z)
    Spec.get_rms()

    ax = plt.subplot(gs[i])
    ax.set_xlim([-1000,1000])
    ax.set_ylim([np.nanmin(flux) * 1.2, np.nanmax(flux) * 1.5])
    # ax.grid("on")
    X = np.linspace(-1000, 1000, 1000)
    ax.step(vel, flux, where="mid")
    # ax.step(vel, err, where="mid")
    ax.step(vel, Spec.rms, where="mid", c=colors[3])
    ax.axvline(0, c="k", ls="--", alpha=0.5)
    # ax.axvline(-M2*2.35/2, c=colors[0], lw=3, ls="--")
    # ax.axvline(M2*2.35/2, c=colors[0], lw=3, ls="--")
    ax.axhline(0, c="k", ls="--", alpha=0.5)
    ax.set_xticklabels([])
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.set_yticklabels([])
    ax.text(0.02, 0.98, Cube.ID + " (z={:.3f})".format(Cube.z), ha="left", va="top",
            transform=ax.transAxes, color="k", fontsize=17.5)

# plt.tight_layout()
plt.savefig(cristal_dir+spec_dir+"all.pdf", bbox_inches="tight")
plt.close()

# --------------------------------------------------

# ------------------------------------------------------------------------------------------------------------