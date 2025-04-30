import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
from photutils import CircularAperture, aperture_photometry
from tqdm import tqdm

from src.utils.cristal_cube import cristal_cube, make_cristal_cube, make_cristal_cube_cutout, mask_cristal_cube
from src.utils.cristal_cube import cristal_spec
from src.utils.flux_extraction import extract_from_sig_mask, get_errors
from src.utils.tables import cristal_dir, cube_dir, ctab, nu_cii
from src.utils.line_models import gaussian, gaussian_double

# --------------------------------------------------------------------------------------------------
# take 3 sigma spectra and add broad component

for i in tqdm(range(len(ctab))):
    ID = ctab["CRISTAL_ID_full"][i]
    file = cube_dir + ctab["File"][i]
    z, ra, dec = ctab["z"][i], ctab["RA"][i], ctab["Dec"][i]
    fwhm_cii = ctab["FWHM_CII"][i]

    file = file.replace("10kms", "{}kms".format(ctab["vel_res_med"][i]))

    Cube = make_cristal_cube(file, ID, z, ra, dec, fwhm_cii)
    vel, nu, flux, err = np.load(cristal_dir+"spectra/contours_3sig_vel_res_med/"+ID+".npy")
    Spec = cristal_spec(vel=vel, nu=nu, flux=flux, err=err, z=z)

    # set up G20 outflow stack
    g20_outflow = gaussian(Cube.vel_axis, amp=0.25, mean=0, sig=533/2.35)
    # Spec.flux+=g20_outflow
    Spec.get_moments()

    if (flux+g20_outflow==np.inf).any():
        print("INFs found in "+ID)

    np.save(cristal_dir+"spectra/g20_sim_part2_vel_res_med/"+ID,
                        np.array([Cube.vel_axis, Cube.nu, flux+g20_outflow, err]))

    fig = plt.figure(figsize=(7.5,7.5))
    ax = plt.subplot(111)
    ax.set_xlim([-1000,1000])
    ax.set_ylim([np.nanmin(Spec.flux+g20_outflow) * 1.2, np.nanmax(Spec.flux+g20_outflow) * 1.5])
    X = np.linspace(-1000, 1000, 1000)
    ax.step(Spec.vel, Spec.flux, where="mid", label="CRISTAL")
    ax.step(Spec.vel, Spec.flux + g20_outflow, where="mid", label="+ G20 outflow")
    ax.step(Spec.vel, Spec.err, where="mid")
    ax.axvline(0, c="k", ls="--", alpha=0.5)
    ax.text(0.02, 0.98, Cube.ID + " (z={:.3f})".format(Cube.z), ha="left", va="top",
            transform=ax.transAxes, color="k", fontsize=17.5)
    ax.set_xlabel(r"Velocity / kms$^{-1}$")
    ax.set_ylabel(r"Flux density / mJy")
    plt.legend(loc=1)
    plt.savefig(cristal_dir + "spectra/g20_sim_part2_vel_res_med/" + ID + ".pdf", bbox_inches="tight")
    plt.close()

    # exit()

print("Done.")

# --------------------------------------------------------------------------------------------------