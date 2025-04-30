import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u
from photutils import CircularAperture
import glob
from scipy.optimize import curve_fit
from tqdm import tqdm
import pyregion
import os, glob
import sys

from astro_cubes.defaults import colors
from astro_cubes.tools import find_nearest, disable_ax_ticks
from cristal.cristal_cube import cristal_cube, make_cristal_cube, make_cristal_cube_cutout
from cristal.tables import cristal_dir, cube_dir, ctab, nu_cii, cosmo
from cristal.formulae import LCII_SFR_lagache, LCII_SFR_de_looze

import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in log10")
from astropy.wcs import FITSFixedWarning
warnings.filterwarnings('ignore', category=FITSFixedWarning, append=True)

# ------------------------------------------------------------------------------------------------------------

# general source masks (to ensure no overlap between neighbouring galaxies)
print("Making source masks...")

for i in tqdm(range(len(ctab))):
    for vel_res in ["low","med","high"]:
        if not ctab["Detect?"][i]:
            continue
        ID = ctab["CRISTAL_ID_full"][i]
        file = cube_dir + ctab["File"][i]
        file = file.replace("10kms", "{}kms".format(ctab[f"vel_res_{vel_res}"][i]))
        z, ra, dec = ctab["z"][i], ctab["RA"][i], ctab["Dec"][i]
        fwhm_cii = ctab["FWHM_CII"][i]

        # load cube and collapse to 2-D T
        Cube = make_cristal_cube(file, ID, z, ra, dec, fwhm_cii)
        Cube_cutout = make_cristal_cube_cutout(Cube, size=4, centre=Cube.c)
        Cube_cutout.convert_to_jypixel()                        # convert cube to Jy/pixel
        flux_map = Cube_cutout.get_flux_map()                   # make L[CII] map

        # DS9 region masks
        # hdu = fits.open(cristal_dir + "data/line_maps/CRISTAL-07a_natural.fits")[0]
        with open(cristal_dir+"data/masks/regions/"+ID+".reg", "r") as file:
            data = file.read()
        reg = pyregion.read_region_as_imagecoord(data, Cube_cutout.wcs.to_header())
        source_mask = np.array(pyregion.get_mask(reg, flux_map), dtype=int)

        # plot mask
        plt.figure()
        ax = plt.subplot(111, projection=Cube_cutout.wcs2d)
        disable_ax_ticks(ax, wcs=True)
        ax.imshow(source_mask)
        ax.text(0.02, 0.98, ID, ha="left", va="top", transform=ax.transAxes, color="w",
               fontsize=25)
        ax.text(0.02, 0.02, "z={:.3f}".format(Cube.z), ha="left", va="bottom",
                transform=ax.transAxes, color="w", fontsize=25)
        plt.savefig(cristal_dir+"data/masks/source_masks/"+ID+f"_mask_vel_res_{vel_res}.pdf", bbox_inches="tight")
        plt.close()

        # save the mask
        np.save(cristal_dir+"data/masks/source_masks/"+ID+f"_mask_vel_res_{vel_res}.npy", source_mask)

print("Done.")

# ------------------------------------------------------------------------------------------------------------

# SFR masks
print("Making SFR-based masks...")

print("Removing old masks...")
SFR_threshold = float(sys.argv[1])
for f in glob.glob(cristal_dir+"data/masks/mask_*/*"):
    os.remove(f)
print("Done.")

for i in tqdm(range(len(ctab))):
    if not ctab["Detect?"][i]:
        continue
    ID = ctab["CRISTAL_ID_full"][i]
    file = cube_dir + ctab["File"][i]
    file = file.replace("10kms", "{}kms".format(ctab["vel_res_med"][i]))
    z, ra, dec = ctab["z"][i], ctab["RA"][i], ctab["Dec"][i]
    fwhm_cii = ctab["FWHM_CII"][i]
    threshold = 10 ** LCII_SFR_lagache(SFR=SFR_threshold, z=z)

    # load cube and collapse to 2-D
    Cube = make_cristal_cube(file, ID, z, ra, dec, fwhm_cii)
    Cube_cutout = make_cristal_cube_cutout(Cube, size=4, centre=Cube.c)
    Cube_cutout.convert_to_jypixel()                        # convert cube to Jy/pixel
    flux_map = Cube_cutout.get_flux_map()                   # make L[CII] map
    sfr_map = Cube_cutout.get_sfr_map()                     # make SFR map
    sigma_cii_map = flux_map / Cube.pix_area

    # make masks
    mask_low_sfr = np.zeros(np.shape(flux_map))                         # generate zero arrays for mask
    mask_high_sfr = np.zeros(np.shape(flux_map))
    sig_flux_map = np.std(flux_map)                                     # get RMS of L[CII] map
    mask_low_sfr[sigma_cii_map < threshold] = 1    # 3-sigma pixels = "galaxy" regions
    mask_high_sfr[sigma_cii_map > threshold] = 1            # pixels above SFR threshold = "high-SFR' regions

    # DS9 region masks
    mask_gal = np.load(f"{cristal_dir}data/masks/contours_2sig_vel_res_med/{Cube.ID}_mask.npy")

    mask_low_sfr = (mask_low_sfr==1) & (mask_gal==1)
    mask_high_sfr = (mask_high_sfr==1) & (mask_gal==1)

    # plot masks
    plt.figure()
    ax = plt.subplot(111, projection=Cube_cutout.wcs2d)
    disable_ax_ticks(ax, wcs=True)
    ax.imshow(mask_low_sfr)
    ax.text(0.02, 0.98, ID, ha="left", va="top", transform=ax.transAxes, color="w",
           fontsize=25)
    ax.text(0.02, 0.02, "z={:.3f}".format(Cube.z), ha="left", va="bottom",
            transform=ax.transAxes, color="w", fontsize=25)
    plt.savefig(cristal_dir+"data/masks/mask_low_sfr_vel_res_med/"+ID+"_mask.pdf", bbox_inches="tight")
    plt.close()

    plt.figure()
    ax = plt.subplot(111, projection=Cube_cutout.wcs2d)
    disable_ax_ticks(ax, wcs=True)
    ax.imshow(mask_high_sfr)
    ax.text(0.02, 0.98, ID, ha="left", va="top", transform=ax.transAxes, color="w",
           fontsize=25)
    ax.text(0.02, 0.02, "z={:.3f}".format(Cube.z), ha="left", va="bottom",
            transform=ax.transAxes, color="w", fontsize=25)
    plt.savefig(cristal_dir+"data/masks/mask_high_sfr_vel_res_med/"+ID+"_mask.pdf", bbox_inches="tight")
    plt.close()

    # save the masks
    np.save(cristal_dir+"data/masks/mask_low_sfr_vel_res_med/"+ID+"_mask.npy", mask_low_sfr)
    np.save(cristal_dir+"data/masks/mask_high_sfr_vel_res_med/"+ID+"_mask.npy", mask_high_sfr)

print("Done.")

# ------------------------------------------------------------------------------------------------------------

# inner/outer masks
print("Making inner/outer masks...")

for i in tqdm(range(len(ctab))):
    if not ctab["in_stack?"][i]:
        continue
    ID = ctab["CRISTAL_ID_full"][i]
    file = cube_dir + ctab["File"][i]
    file = file.replace("10kms", "{}kms".format(ctab["vel_res_med"][i]))
    z, ra, dec = ctab["z"][i], ctab["RA"][i], ctab["Dec"][i]
    fwhm_cii = ctab["FWHM_CII"][i]

    # load cube and collapse to 2-D
    Cube = make_cristal_cube(file, ID, z, ra, dec, fwhm_cii)
    Cube_cutout = make_cristal_cube_cutout(Cube, size=4, centre=Cube.c)
    Cube_cutout.convert_to_jypixel()                        # convert cube to Jy/pixel
    # flux_map = Cube_cutout.get_flux_map()                   # make L[CII] map
    # sfr_map = Cube_cutout.get_sfr_map()                     # make SFR map
    # sigma_cii_map = flux_map / Cube.pix_area
    i_min = find_nearest(Cube_cutout.nu, nu_cii/(1+Cube_cutout.z)*(1+Cube_cutout.fwhm_cii/3e05))[0]
    i_max = find_nearest(Cube_cutout.nu, nu_cii/(1+Cube_cutout.z)*(1-Cube_cutout.fwhm_cii/3e05))[0]
    img = np.nansum(Cube_cutout.cube[i_min:i_max], axis=0)

    # DS9 region masks
    mask_gal = np.load(f"{cristal_dir}data/masks/contours_2sig_vel_res_med/{Cube.ID}_mask.npy")

    # sig_flux_map = np.std(img)
    # sigma_cii_map[(flux_map<2*sig_flux_map) | (source_mask==0)] = np.nan
    # print(ID, sig_flux_map)

    img[mask_gal == 0] = np.nan
    sig_flux_map = np.nanstd(img)
    # img[img < 2 * sig_flux_map] = np.nan
    img_med = np.nanmedian(img.ravel())

    # make masks
    mask_outer = np.zeros(np.shape(img))                         # generate zero arrays for mask
    mask_inner = np.zeros(np.shape(img))
    mask_outer[img < img_med] = 1    # 2-sigma pixels = "galaxy" regions
    mask_inner[img > img_med] = 1            # pixels above SFR threshold = "high-SFR' regions

    # plot masks
    plt.figure()
    ax = plt.subplot(111, projection=Cube_cutout.wcs2d)
    disable_ax_ticks(ax, wcs=True)
    ax.imshow(mask_outer)
    ax.text(0.02, 0.98, ID, ha="left", va="top", transform=ax.transAxes, color="w",
           fontsize=25)
    ax.text(0.02, 0.02, "z={:.3f}".format(Cube.z), ha="left", va="bottom",
            transform=ax.transAxes, color="w", fontsize=25)
    plt.savefig(cristal_dir+"data/masks/mask_outer_vel_res_med/"+ID+"_mask.pdf", bbox_inches="tight")
    plt.close()

    plt.figure()
    ax = plt.subplot(111, projection=Cube_cutout.wcs2d)
    disable_ax_ticks(ax, wcs=True)
    ax.imshow(mask_inner)
    ax.text(0.02, 0.98, ID, ha="left", va="top", transform=ax.transAxes, color="w",
           fontsize=25)
    ax.text(0.02, 0.02, "z={:.3f}".format(Cube.z), ha="left", va="bottom",
            transform=ax.transAxes, color="w", fontsize=25)
    plt.savefig(cristal_dir+"data/masks/mask_inner_vel_res_med/"+ID+"_mask.pdf", bbox_inches="tight")
    plt.close()

    # save the masks
    np.save(cristal_dir+"data/masks/mask_outer_vel_res_med/"+ID+"_mask.npy", mask_outer)
    np.save(cristal_dir+"data/masks/mask_inner_vel_res_med/"+ID+"_mask.npy", mask_inner)

print("Done.")