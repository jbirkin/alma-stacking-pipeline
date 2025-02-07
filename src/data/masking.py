import numpy as np
import glob
from tqdm import tqdm
import pyregion
import os, glob
import yaml
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in log10")

from alma_stacking_pipeline.src.config_loader import work_dir, cristal_tab, nu_cii
from alma_stacking_pipeline.src.utils import find_nearest
from alma_stacking_pipeline.src.plotting import plot_mask
from alma_stacking_pipeline.src.data.alma import make_alma_cube, make_alma_cube_cutout
from alma_stacking_pipeline.src.formulae import convert_SFR_to_LCII_lagache

# ------------------------------------------------------------------------------------------------------------

def make_source_masks(tab):
    # general source masks (to ensure no overlap between neighbouring galaxies)
    print("Making source masks...")

    for ID, file, z, ra, dec, fwhm_cii in tqdm(zip(tab["CRISTAL_ID_full"], tab["File"], tab["z"], tab["RA"],
                                                   tab["Dec"], tab["FWHM_CII"]), total=len(tab)):
        cube_file = Path(work_dir) / "data/cubes" / file
        cube_file = str(cube_file).replace("_fake", "")

        mask_path = Path(work_dir) / "data/masks/regions" / f"{ID}.reg"
        if not mask_path.exists():
            print(f"Warning: Mask file {mask_path} not found. Skipping {ID}.")
            continue

        # load cube and collapse to 2-D
        Cube = make_alma_cube(cube_file, ID, z, ra, dec, fwhm_cii)
        Cube_cutout = make_alma_cube_cutout(Cube, size=4, centre=Cube.c)
        Cube_cutout.convert_to_jypixel()                        # convert cube to Jy/pixel
        flux_map = Cube_cutout.get_flux_map()                   # make L[CII] map

        # DS9 region masks
        with mask_path.open("r") as file:
            data = file.read()
        reg = pyregion.read_region_as_imagecoord(data, Cube_cutout.wcs.to_header())
        source_mask = np.array(pyregion.get_mask(reg, flux_map), dtype=int)

        plot_mask(mask=source_mask, Cube=Cube_cutout,
                  outfile="tmp.pdf")
        # save the mask
        np.save(f"{work_dir}data/masks/source_masks/{ID}_mask.npy", source_mask)

    print("Done.")

# ------------------------------------------------------------------------------------------------------------

def make_sfr_masks(tab, SFR_threshold=1.7):
    print(f"Making SFR-based masks with threshold {SFR_threshold:.2f}...")

    for ID, file, z, ra, dec, fwhm_cii, detect in tqdm(zip(tab["CRISTAL_ID_full"], tab["File"], tab["z"], tab["RA"],
                                                           tab["Dec"], tab["FWHM_CII"], tab["Detect?"]), total=len(tab)):
        if not detect:
            continue

        cube_file = Path(work_dir) / "data/cubes" / file
        cube_file = str(cube_file).replace("_fake", "")
        threshold = 10 ** convert_SFR_to_LCII_lagache(SFR=SFR_threshold, z=z)

        # load cube and collapse to 2-D
        Cube = make_alma_cube(cube_file, ID, z, ra, dec, fwhm_cii)
        Cube_cutout = make_alma_cube_cutout(Cube, size=4, centre=Cube.c)
        Cube_cutout.convert_to_jypixel()                        # convert cube to Jy/pixel
        flux_map = Cube_cutout.get_flux_map()                   # make L[CII] map
        sigma_cii_map = flux_map / Cube.pix_area

        # make masks
        mask_low_sfr = np.zeros(np.shape(flux_map))                         # generate zero arrays for mask
        mask_high_sfr = np.zeros(np.shape(flux_map))
        sig_flux_map = np.std(flux_map)                                     # get RMS of L[CII] map
        mask_low_sfr[(flux_map>2*sig_flux_map) & (sigma_cii_map < threshold)] = 1    # 3-sigma pixels = "galaxy" regions
        mask_high_sfr[sigma_cii_map > threshold] = 1            # pixels above SFR threshold = "high-SFR' regions

        # DS9 region masks
        source_mask_path = Path(work_dir) / "data/masks/source_masks" / f"{ID}_mask.npy"
        if not source_mask_path.exists():
            print(f"Warning: Source mask {source_mask_path} not found. Skipping {ID}.")
            continue
        source_mask = np.load(source_mask_path)

        mask_low_sfr = (mask_low_sfr==1) & (source_mask==1)
        mask_high_sfr = (mask_high_sfr==1) & (source_mask==1)

        # plot masks
        plot_mask(mask=mask_low_sfr, Cube=Cube_cutout, outfile="tmp.pdf")
        plot_mask(mask=mask_high_sfr, Cube=Cube_cutout, outfile="tmp.pdf")

        # save the masks
        np.save(f"{work_dir}data/masks/mask_low_sfr_vel_res_med/{ID}_mask.npy", mask_low_sfr)
        np.save(f"{work_dir}data/masks/mask_high_sfr_vel_res_med/{ID}_mask.npy", mask_high_sfr)

    print("Done.")