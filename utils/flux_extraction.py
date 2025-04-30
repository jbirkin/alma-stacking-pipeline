import numpy as np
from photutils import aperture_photometry, CircularAperture, EllipticalAperture

from astro_cubes.tools import find_nearest
from src.utils.cristal_cube import cristal_cube, make_cristal_cube
from src.utils.tables import nu_cii

# ------------------------------------------------------------------------------------------------------------

class cristal_spec:
    def __init__(self, nu=None, vel=None, flux=None, err=None, ID=None, z=None):
        self.vel = vel
        self.nu = nu
        self.flux = flux
        self.err = err
        self.ID = ID
        self.z = z

    def crop_vel(self, v_min, v_max):
        i_min = find_nearest(self.vel, v_min)[0]
        i_max = find_nearest(self.vel, v_max)[0]
        self.vel = self.vel[i_min:i_max]
        self.nu = self.nu[i_min:i_max]
        self.flux = self.flux[i_min:i_max]
        self.err = self.err[i_min:i_max]

# ------------------------------------------------------------------------------------------------------------

def get_beam_aperture(Cube, positions):
    """from the FITS header of a cube, get the header information and make a beam-sized elliptical aperture. Takes
    themedian beam properties at all wavelengths"""
    bmaj_all, bmin_all, bpa_all = Cube.beams["BMAJ"], Cube.beams["BMIN"], Cube.beams["BPA"]
    bmaj = np.median(bmaj_all)/Cube.pix_scale       # correct for the pixel scale
    bmin = np.median(bmin_all)/Cube.pix_scale       # correct for the pixel scale
    bpa = np.median(bpa_all)                        # unsure if astropy and ALMA use the same angle conventions
    beam_aperture = EllipticalAperture(positions, a=bmaj/2, b=bmin/2, theta=bpa)
    return beam_aperture

# ------------------------------------------------------------------------------------------------------------

def extract_from_sig_mask(Cube, nsig=3):
    """make a collapsed map of the source, then use a mask of pixels with flux greater than
    n-sigma to extract spectra.
    N.B. currently using a velocity window of +/- 300 km/s for the collapsed image. might want to
    adjust this depending on linewidth
    """

    # get indices of regions around
    i_min = find_nearest(Cube.nu, nu_cii/(1+Cube.z)*(1+Cube.fwhm_cii/3e05))[0]
    i_max = find_nearest(Cube.nu, nu_cii/(1+Cube.z)*(1-Cube.fwhm_cii/3e05))[0]
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

def get_errors(Cube, n_random=100, buffer=0):
    sx, sy = Cube.cube.shape[1], Cube.cube.shape[2]
    # load beam
    bmaj, bmin, bpa = Cube.beams[int(Cube.beams.shape[0]/2)][0:3]
    bmaj_circ, bmin_circ = bmaj/np.abs(Cube.header["CDELT1"]*3600), bmin/np.abs(Cube.header["CDELT1"]*3600)
    aper_rad = (bmaj_circ*bmin_circ)**0.5

    # make random apertures (masking out central region)
    x_random = np.arange(buffer, sx - buffer, 1)
    x_random = np.delete(x_random, ((x_random > sx/2-(aper_rad+10)) &
                                    (x_random < sx/2+(aper_rad+10))))
    x_random = np.delete(x_random, ((x_random < (aper_rad+2)) | (x_random > sx-(aper_rad+10))))

    y_random = np.arange(buffer, sy - buffer, 1)
    y_random = np.delete(y_random, ((y_random > sy/2-(aper_rad+10)) &
                                    (y_random < sy/2+(aper_rad+10))))
    y_random = np.delete(y_random, ((y_random < (aper_rad+2)) | (y_random > sy-(aper_rad+10))))

    random_coords = np.array([np.random.choice(x_random, n_random), np.random.choice(y_random, n_random)]).T

    apertures = CircularAperture(random_coords, aper_rad)

    # get RMS of fluxes in the random apertures
    err = []
    for m in range(len(Cube.cube)):
        bmaj, bmin, bpa = Cube.beams[m][0:3]
        pix_per_beam = np.pi*bmaj*bmin/(4*np.log(2))/Cube.pix_scale**2
        phot = aperture_photometry(Cube.cube[m], apertures, method='exact')
        err.append(np.nanstd(phot["aperture_sum"]/pix_per_beam*1000))#/n_random**0.5)
    err = np.array(err)

    return err, apertures

# ------------------------------------------------------------------------------------------------------------