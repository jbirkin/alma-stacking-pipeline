import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from photutils import CircularAperture, aperture_photometry
from tqdm import tqdm
import sys
import pyregion
import os, glob

from astro_cubes.tools import find_nearest
from astro_cubes.defaults import colors
from src.utils.cristal_cube import cristal_cube, make_cristal_cube, make_cristal_cube_cutout, mask_cristal_cube
from src.utils.flux_extraction import extract_from_sig_mask, get_errors
from src.utils.tables import cristal_dir, cube_dir, ctab, nu_cii

print("Extracting spectra...")

# spec_method = "contours"
spec_method = str(sys.argv[1])
if spec_method=="masks":
    mask_dir = str(sys.argv[2])+"/"

# --------------------------------------------------------------------------------------------------
# Using contours

if spec_method=="contours":
    print("Using contours...")

    for nsig in [2,3]:
        print(str(nsig) + "-sigma")
        for i in tqdm(range(len(ctab))):
            if not ctab["in_stack?"][i]:
                continue
            for vel_res in ["med"]:
                ID = ctab["CRISTAL_ID_full"][i]
                file = cube_dir + ctab["File"][i]
                z, ra, dec = ctab["z"][i], ctab["RA"][i], ctab["Dec"][i]
                fwhm_cii = ctab["FWHM_CII"][i]
                file = file.replace("10kms", "{}kms".format(ctab["vel_res_" + vel_res][i]))

                Cube = make_cristal_cube(file, ID, z, ra, dec, fwhm_cii)
                Cube_cutout = make_cristal_cube_cutout(Cube, size=4, centre=Cube.c)
                # DS9 region masks
                source_mask = np.load(cristal_dir + "data/masks/source_masks/"+ID+f"_mask_vel_res_{vel_res}.npy")

                Cube_masked = make_cristal_cube_cutout(Cube, size=4, centre=Cube.c)
                for j in range(len(Cube_masked.cube)):
                    Cube_masked.cube[j][source_mask!=1] = np.nan

                flux, mask_sig = extract_from_sig_mask(Cube_masked, nsig=nsig)
                err, apertures = get_errors(Cube, n_random=100, buffer=Cube.cube.shape[2]/4)

                # save mask
                np.save(cristal_dir+"spectra/contours_"+str(nsig)+"sig_vel_res_"+vel_res+"/"+ID,
                        np.array([Cube.vel_axis, Cube.nu, flux, err]))
                np.save(cristal_dir+"data/masks/contours_"+str(nsig)+"sig_vel_res_"+vel_res+"/"+ID+"_mask",
                        mask_sig)

    print("Done.")

# --------------------------------------------------------------------------------------------------
# Using masks

elif spec_method=="masks":
    print(f"Using pre-made masks from {mask_dir}...")

    print("Removing old spectra...")
    for f in glob.glob(f"{cristal_dir}spectra/{mask_dir}/*"):
        os.remove(f)
    print("Done.")

    for i in tqdm(range(len(ctab))):
        if not ctab["in_stack?"][i]:
            continue
        for vel_res in ["med"]:
            ID = ctab["CRISTAL_ID_full"][i]
            mask = np.load(f"{cristal_dir}data/masks/{mask_dir}{ID}_mask.npy")

            file = cube_dir + ctab["File"][i]
            z, ra, dec = ctab["z"][i], ctab["RA"][i], ctab["Dec"][i]
            fwhm_cii = ctab["FWHM_CII"][i]
            file = file.replace("10kms", f"{ctab['vel_res_'+vel_res][i]}kms")

            Cube = make_cristal_cube(file, ID, z, ra, dec, fwhm_cii)
            Cube_cutout = make_cristal_cube_cutout(Cube, size=4, centre=Cube.c)
            beam_aperture = Cube_cutout.get_beam_aperture()
            if len(np.where(mask.ravel()==True)[0])<beam_aperture.area:#np.sum(mask)==0:
                continue

            Cube_masked = Cube_cutout.cube.copy()
            for j in range(len(Cube_cutout.cube)):
                Cube_masked[j][mask!=1] = np.nan

            # do aperture photometry
            flux_masked = np.nansum(Cube_masked, axis=(1,2))
            err, apertures = get_errors(Cube, n_random=100, buffer=Cube.cube.shape[2]/4)
            bmaj, bmin, bpa = Cube.beams[i][0:3]
            pix_per_beam = np.pi * bmaj * bmin / (4 * np.log(2)) / Cube.pix_scale ** 2
            flux_masked /= pix_per_beam
            flux_masked *= 1000

            if np.sum(flux_masked)>0:
                np.save(f"{cristal_dir}spectra/{mask_dir}/{ID}",
                        np.array([Cube.vel_axis, Cube.nu, flux_masked, err]))
                # print(cristal_dir+"spectra/"+mask_dir.replace("/","")+"_vel_res_"+vel_res+"/"+ID)

    print("Done.")

# --------------------------------------------------------------------------------------------------