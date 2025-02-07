import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm
import os, glob
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice.")

from alma_stacking_pipeline.src.config_loader import cristal_tab, work_dir, nu_cii
from alma_stacking_pipeline.src.utils import find_nearest, moments
from alma_stacking_pipeline.src.data.alma import make_alma_cube, make_alma_cube_cutout
from alma_stacking_pipeline.src.line_models import gaussian, gaussian_double

# ------------------------------------------------------------------------------------------------------------

def extract_from_sig_mask(Cube, nsig=3):
    """make a collapsed map of the source, then use a mask of pixels with flux greater than
    n-sigma to extract spectra.
    """

    # get indices of regions around
    i_min = find_nearest(Cube.nu, nu_cii/(1+Cube.z)*(1+Cube.fwhm_cii/3e05))[0]
    i_max = find_nearest(Cube.nu, nu_cii/(1+Cube.z)*(1-Cube.fwhm_cii/3e05))[0]
    img = np.nansum(Cube.cube[i_min:i_max], axis=0)

    mask_sig = img / np.nanstd(img) > nsig

    flux = np.array([
        np.sum(Cube.cube[i, mask_sig] / (np.pi * bmaj * bmin / (4 * np.log(2)) / Cube.pix_scale**2) * 1000)
        for i, (bmaj, bmin, _, _, _) in enumerate(Cube.beams)
    ])

    return flux, mask_sig

# ------------------------------------------------------------------------------------------------------------

def extract_spectra(tab, spec_method, mask_dir):
    print("Extracting spectra...")
    output_dir = Path(work_dir) / "data/spectra/contours"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Using contours
    if spec_method=="contours":
        print("Using contours...")

        for ID, file, z, ra, dec, fwhm_cii in tqdm(zip(tab["CRISTAL_ID_full"], tab["File"], tab["z"], tab["RA"],
                                                       tab["Dec"], tab["FWHM_CII"]), total=len(tab)):
            cube_file = Path(work_dir) / "data/cubes" / file
            cube_file = str(cube_file).replace("_fake", "")
            Cube = make_alma_cube(cube_file, ID, z, ra, dec, fwhm_cii)

            # DS9 region masks
            mask_path = Path(work_dir) / "data/masks/source_masks" / f"{ID}_mask.npy"
            if not mask_path.exists():
                print(f"Warning: Mask file {mask_path} not found. Skipping {ID}.")
                continue

            source_mask = np.load(mask_path)
            Cube_masked = make_alma_cube_cutout(Cube, size=4, centre=Cube.c)
            Cube_masked.cube[:, source_mask != 1] = np.nan

            flux, mask_sig = extract_from_sig_mask(Cube_masked, nsig=2)
            err = 0.1*flux.copy()

            # save mask
            np.save(f"{work_dir}data/spectra/contours/{ID}.npy", np.array([Cube.vel_axis, Cube.nu, flux, err]))
            np.save(Path(work_dir) / f"data/masks/contours/{ID}_mask.npy", mask_sig)

        print("Done.")

    # --------------------------------------------------------------------------------------------------
    # Using masks

    elif spec_method=="masks":
        print(f"Using pre-made masks from {mask_dir}...")
        mask_dir_path = Path(work_dir) / "spectra" / mask_dir
        for file in mask_dir_path.glob("*"):
            file.unlink()
        print("Old spectra removed.")

        for ID, file, z, ra, dec, fwhm_cii in tqdm(zip(tab["CRISTAL_ID_full"], tab["File"], tab["z"], tab["RA"],
                                                       tab["Dec"], tab["FWHM_CII"]), total=len(tab)):
            mask_path = Path(work_dir) / "data/masks" / mask_dir / f"{ID}_mask.npy"
            if not mask_path.exists():
                print(f"Warning: Mask file {mask_path} not found. Skipping {ID}.")
                continue

            mask = np.load(mask_path)
            cube_file = Path(work_dir) / "data/cubes" / file
            cube_file = str(cube_file).replace("_fake", "")
            Cube = make_alma_cube(cube_file, ID, z, ra, dec, fwhm_cii)
            Cube_cutout = make_alma_cube_cutout(Cube, size=4, centre=Cube.c)
            beam_aperture = Cube_cutout.get_beam_aperture()

            if len(np.where(mask.ravel()==True)[0])<beam_aperture.area:#np.sum(mask)==0:
                continue

            Cube_masked = Cube_cutout.cube.copy()
            Cube_masked.cube[:, mask != 1] = np.nan

            # do aperture photometry
            flux_masked = np.nansum(Cube_masked, axis=(1,2))
            err, apertures = get_errors(Cube, n_random=100, buffer=Cube.cube.shape[2]/4)
            bmaj, bmin, bpa = Cube.beams[i][0:3]
            pix_per_beam = np.pi * bmaj * bmin / (4 * np.log(2)) / Cube.pix_scale ** 2
            flux_masked /= pix_per_beam
            flux_masked *= 1000

            if np.sum(flux_masked)>0:
                np.save("tmp.npy", np.array([Cube.vel_axis, Cube.nu, flux_masked, err]))

        print("Done.")

# --------------------------------------------------------------------------------------------------

def center_spectra(spec_dir):
    print("Centering spectra by calculating the first moment...")
    print("Directory: {}".format(spec_dir))
    filelist = [file for file in Path(spec_dir).glob("*.npy") if "centered" not in file.name]

    # for i in tqdm(range(len(filelist))):
    #     vel, nu, flux, err = np.load(filelist[i])
    #
    #     # fit an approximate Gaussian
    #     popt, pcov = curve_fit(gaussian, vel, flux,
    #                            p0=[np.max(flux), 0, 100],
    #                            bounds=[[np.max(flux)/2, -100, 50],
    #                                    [np.max(flux), 100, 150]])
    #     # use to define limits for calculating moments
    #     i_min = find_nearest(vel, popt[1] - 1.5 * 2.35 * popt[2])[0]
    #     i_max = find_nearest(vel, popt[1] + 1.5 * 2.35 * popt[2])[0]
    #
    #     M0, dM0, M1, dM1, M2, dM2 = moments(vel[i_min:i_max],
    #                                         flux[i_min:i_max],
    #                                         err[i_min:i_max])
    #
    #     np.save(filelist[i].replace(".npy","_centered.npy"),
    #             np.array([vel - M1, nu, flux, err]))

    for file in tqdm(filelist):
        vel, nu, flux, err = np.load(file)
        try:
            popt, _ = curve_fit(gaussian, vel, flux, p0=[np.max(flux), 0, 100])
            i_min, i_max = find_nearest(vel, popt[1] - 1.5 * 2.35 * popt[2])[0], \
                           find_nearest(vel, popt[1] + 1.5 * 2.35 * popt[2])[0]
            _, _, M1, _, _, _ = moments(vel[i_min:i_max], flux[i_min:i_max], err[i_min:i_max])
            np.save(file.with_name(file.stem + "_centered.npy"), np.array([vel - M1, nu, flux, err]))
        except RuntimeError:
            print(f"Warning: Gaussian fitting failed for {file}. Skipping.")

    print("Done.")

# --------------------------------------------------------------------------------------------------