import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm
import os, glob

from alma_stacking_pipeline.src.config import cristal_tab, cube_dir, work_dir, nu_cii
from alma_stacking_pipeline.src.utils import find_nearest, moments
from alma_stacking_pipeline.src.data.alma import make_alma_cube, make_alma_cube_cutout
from alma_stacking_pipeline.src.line_models import gaussian, gaussian_double

# ------------------------------------------------------------------------------------------------------------

def extract_from_sig_mask(Cube, nsig=3):
    """make a collapsed map of the source, then use a mask of pixels with flux greater than
    n-sigma to extract spectra.
    N.B. currently using a velocity window of +/- 300 km/s for the collapsed image. might want to
    adjust this depending on linewidth
    """

    # get indices of regions around
    i_min = find_nearest(Cube.nu, nu_cii/(1+Cube.z)*(1+300/3e05))[0]
    i_max = find_nearest(Cube.nu, nu_cii/(1+Cube.z)*(1-300/3e05))[0]
    img = np.nansum(Cube.cube[i_min:i_max], axis=0)

    mask_sig = img / np.nanstd(img) > nsig

    flux = []
    for i in range(len(Cube.cube)):
        bmaj, bmin, bpa = Cube.beams[i][0:3]
        pix_per_beam = np.pi*bmaj*bmin/(4*np.log(2))/Cube.pix_scale**2
        flux.append(np.sum(Cube.cube[i,mask_sig] / pix_per_beam * 1000))
    flux = np.array(flux)

    return flux, mask_sig

# ------------------------------------------------------------------------------------------------------------

def extract_spectra(tab, spec_method, mask_dir):

    print("Extracting spectra...")

    # --------------------------------------------------------------------------------------------------
    # Using contours

    if spec_method=="contours":
        print("Using contours...")

        for nsig in [2,3]:
            print(str(nsig) + "-sigma")
            for i in tqdm(range(len(tab))):
                if not tab["in_stack?"][i]:
                    continue
                ID = tab["CRISTAL_ID_full"][i]
                file = cube_dir + tab["File"][i]
                z, ra, dec = tab["z"][i], tab["RA"][i], tab["Dec"][i]
                file = file.replace("10kms", "20kms")

                Cube = make_alma_cube(file, ID, z, ra, dec)
                Cube_cutout = make_alma_cube_cutout(Cube, size=4, centre=Cube.c)
                # DS9 region masks
                source_mask = np.load(f"{work_dir}data/masks/source_masks/{ID}_mask.npy")

                Cube_masked = make_alma_cube_cutout(Cube, size=4, centre=Cube.c)
                for j in range(len(Cube_masked.cube)):
                    Cube_masked.cube[j][source_mask!=1] = np.nan

                flux, mask_sig = extract_from_sig_mask(Cube_masked, nsig=nsig)
                err = 0.1*flux.copy()

                # save mask
                np.save(f"{work_dir}data/spectra/contours/{ID}.npy",
                        np.array([Cube.vel_axis, Cube.nu, flux, err]))

        print("Done.")

    # --------------------------------------------------------------------------------------------------
    # Using masks

    elif spec_method=="masks":
        print("Using pre-made masks from {}...".format(mask_dir))

        print("Removing old spectra...")
        for f in glob.glob(cristal_dir+"spectra/"+mask_dir+"/*"):
            os.remove(f)
        print("Done.")

        for i in tqdm(range(len(ctab))):
            if not ctab["Detect?"][i]:
                continue
            for vel_res in ["med"]:
                ID = ctab["CRISTAL_ID_full"][i]
                mask = np.load(cristal_dir+"data/masks/"+mask_dir+ID+"_mask.npy")

                file = cube_dir + ctab["File"][i]
                z, ra, dec = ctab["z"][i], ctab["RA"][i], ctab["Dec"][i]
                file = file.replace("10kms", "{}kms".format(ctab["vel_res_" + vel_res][i]))

                Cube = make_alma_cube(file, ID, z, ra, dec)
                Cube_cutout = make_alma_cube_cutout(Cube, size=4, centre=Cube.c)
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
                    np.save("tmp.npy", np.array([Cube.vel_axis, Cube.nu, flux_masked, err]))
                    # print(cristal_dir+"spectra/"+mask_dir.replace("/","")+"_vel_res_"+vel_res+"/"+ID)

        print("Done.")

# --------------------------------------------------------------------------------------------------

def center_spectra(spec_dir):
    print("Centering spectra by calculating the first moment...")
    print("Directory: {}".format(spec_dir))
    filelist = glob.glob(spec_dir+"*.npy")
    filelist = [file for file in filelist if not "centered" in file]
    print(np.load(filelist[0]))

    for i in tqdm(range(len(filelist))):
        vel, nu, flux, err = np.load(filelist[i])

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

        # print(filelist[i].split("/")[-1], M2)

        np.save(filelist[i].replace(".npy","_centered.npy"),
                np.array([vel - M1, nu, flux, err]))

    print("Done.")

# --------------------------------------------------------------------------------------------------