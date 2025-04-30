import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
from photutils import CircularAperture, EllipticalAperture, aperture_photometry
from tqdm import tqdm

from astro_cubes.defaults import colors
from src.utils.cristal_cube import cristal_cube, make_cristal_cube, make_cristal_cube_cutout, mask_cristal_cube
from src.utils.cristal_cube import cristal_spec
from src.utils.flux_extraction import extract_from_sig_mask, get_errors
from src.utils.tables import cristal_dir, cube_dir, ctab, nu_cii
from src.utils.line_models import gaussian, gaussian_double

# --------------------------------------------------------------------------------------------------
# extract spectra

vel_res = "med"

for i in tqdm(range(len(ctab))):
    ID = ctab["CRISTAL_ID_full"][i]
    file = cube_dir + ctab["File"][i]
    z, ra, dec = ctab["z"][i], ctab["RA"][i], ctab["Dec"][i]
    fwhm_cii = ctab["FWHM_CII"][i]
    file = file.replace("10kms", "{}kms".format(ctab["vel_res_" + vel_res][i]))

    Cube = make_cristal_cube(file, ID, z, ra, dec, fwhm_cii)
    # vel, nu, flux, err = np.load(cristal_dir+"spectra/"+spec_dir+ID+".npy")
    # Spec = cristal_spec(vel=vel, nu=nu, flux=flux, err=err, z=z)

    c = SkyCoord(ra, dec, unit=u.deg)
    x, y = c.to_pixel(Cube.wcs)
    x+=30

    bmaj_all, bmin_all, bpa_all = Cube.beams["BMAJ"], Cube.beams["BMIN"], Cube.beams["BPA"]
    bmaj = np.median(bmaj_all) / Cube.pix_scale  # correct for the pixel scale
    bmin = np.median(bmin_all) / Cube.pix_scale  # correct for the pixel scale
    bpa = np.median(bpa_all)  # unsure if astropy and ALMA use the same angle conventions
    beam_aperture = EllipticalAperture([x,y], a=bmaj / 2, b=bmin / 2, theta=bpa)
    pix_per_beam = beam_aperture.area / np.log(2)

    # do aperture photometry
    flux = []
    # set NaN values to zero to prevent NaN fluxes
    cube_to_use = Cube.cube.copy()
    cube_to_use[np.isnan(cube_to_use)] = 0
    for m in range(len(Cube.cube)):
        # phot = aperture_photometry(cube_to_use[m], aperture, error=error_cube_to_use[m], method='exact')
        phot = aperture_photometry(cube_to_use[m], beam_aperture, method='exact')
        flux.append(phot["aperture_sum"][0]/pix_per_beam*1000)
    err, apertures = get_errors(Cube, n_random=100)

    # set up G20 outflow stack
    g20_outflow = gaussian_double(Cube.vel_axis,
                                  amp1=0.25, mean1=0, sig1=533/2.35,
                                  amp2=1.75, mean2=0, sig2=230/2.35)

    Spec = cristal_spec(vel=Cube.vel_axis, nu=Cube.nu, flux=flux+g20_outflow, err=err, z=z)
    Spec.get_moments()

    if (flux+g20_outflow==np.inf).any():
        print("INFs found in "+ID)

    np.save(cristal_dir+"spectra/g20_sim_vel_res_"+vel_res+"/"+ID,
                        np.array([Cube.vel_axis, Cube.nu, flux+g20_outflow, err]))

# --------------------------------------------------------------------------------------------------

for i in range(len(ctab)):
    if not ctab["Detect?"][i]:
        continue
    ID = ctab["CRISTAL_ID_full"][i]
    file = cube_dir + ctab["File"][i]
    z, ra, dec = ctab["z"][i], ctab["RA"][i], ctab["Dec"][i]
    fwhm_cii = ctab["FWHM_CII"][i]

    file = file.replace("10kms", "{}kms".format(ctab["vel_res_" + vel_res][i]))

    Cube = make_cristal_cube(file, ID, z, ra, dec, fwhm_cii)
    vel, nu, flux, err = np.load(f"{cristal_dir}spectra/contours_2sig_vel_res_med/{ID}.npy")
    Spec = cristal_spec(vel=vel, nu=nu, flux=flux, err=err, z=z)
    Spec.get_rms()

    # set up G20 outflow stack
    g20_outflow = gaussian_double(Cube.vel_axis,
                                  amp1=0.25, mean1=0, sig1=533 / 2.35,
                                  amp2=1.75, mean2=0, sig2=230 / 2.35)

    noise = np.random.normal(loc=0, scale=Spec.rms, size=len(Spec.flux))
    Spec_g20 = cristal_spec(vel=Cube.vel_axis, nu=Cube.nu, flux=g20_outflow + noise, err=err, z=z)
    Spec_g20.get_rms()

    if (flux + g20_outflow == np.inf).any():
        print("INFs found in " + ID)

    np.save(cristal_dir + "spectra/g20_sim_add_rms_vel_res_med/" + ID,
            np.array([Spec_g20.vel, Spec_g20.nu, Spec_g20.flux, err]))

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.step(Spec_g20.vel, Spec_g20.flux, where="mid")
    ax.step(Spec_g20.vel, Spec_g20.rms, where="mid", c=colors[3])
    ax.text(0.02, 0.98, ID, va="top", ha="left", transform=ax.transAxes)
    plt.close()

# --------------------------------------------------------------------------------------------------

print("Done.")