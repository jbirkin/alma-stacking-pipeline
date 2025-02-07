import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.stats import bootstrap
from photutils import EllipticalAperture
from scipy.optimize import curve_fit
import emcee
import corner

from alma_stacking_pipeline.src.config_loader import work_dir, cristal_tab, nu_cii, cosmo
from alma_stacking_pipeline.src.utils import find_nearest, moments, mc_errors
from alma_stacking_pipeline.src.line_models import gaussian
from alma_stacking_pipeline.src.line_models import log_prior_single, log_probability_single, log_likelihood_single
from alma_stacking_pipeline.src.line_models import log_prior_double, log_probability_double, log_likelihood_double

# ------------------------------------------------------------------------------------------------------------

class alma_cube:
    def __init__(self, nu=None, vel_axis=None, cube=None, error_cube=None, wcs=None, header=None, ID=None,
                 pix_scale=None, z=None, beams=None, ra=None, dec=None, fwhm_cii=None):
        self.nu = nu
        self.vel_axis = vel_axis
        self.cube = cube
        self.error_cube = error_cube
        self.wcs = wcs
        self.wcs2d = WCS.dropaxis(self.wcs, 2)
        self.header = header
        self.ID = ID
        self.pix_scale = pix_scale
        self.z = z
        self.Dl = cosmo.luminosity_distance(self.z).to(u.Mpc).value
        self.Da = cosmo.angular_diameter_distance(self.z).to(u.kpc).value
        self.pix_area = (self.Da * self.pix_scale / 206265) ** 2
        self.beams = beams
        self.ra = ra
        self.dec = dec
        self.fwhm_cii = fwhm_cii
        self.c = SkyCoord(ra, dec, unit=u.deg)

    def get_beam_aperture(self):
        """from the FITS header of a cube, get the header information and make a beam-sized elliptical aperture. Takes
        themedian beam properties at all wavelengths"""
        bmaj_all, bmin_all, bpa_all = self.beams["BMAJ"], self.beams["BMIN"], self.beams["BPA"]
        bmaj = np.median(bmaj_all) / self.pix_scale  # correct for the pixel scale
        bmin = np.median(bmin_all) / self.pix_scale  # correct for the pixel scale
        bpa = np.median(bpa_all)  # unsure if astropy and ALMA use the same angle conventions
        beam_aperture = EllipticalAperture([0,0], a=bmaj / 2, b=bmin / 2, theta=bpa)
        return beam_aperture

    def convert_to_jypixel(self):
        # convert from per beam to per pixel
        beam_aperture = self.get_beam_aperture()
        pix_per_beam = beam_aperture.area/np.log(2)
        self.cube /= pix_per_beam

    def get_flux_map(self):
        i_min = find_nearest(self.nu, nu_cii/(1+self.z)*(1+self.fwhm_cii/3e05))[0]
        i_max = find_nearest(self.nu, nu_cii/(1+self.z)*(1-self.fwhm_cii/3e05))[0]
        flux_map = np.nansum(self.cube[i_min:i_max], axis=0)
        # multiply by dv to get integrated [CII] flux
        dv = np.abs(self.vel_axis[1]-self.vel_axis[0])
        flux_map *= dv
        # convert from [CII] flux to [CII] luminosity
        flux_map *= 1.04e-03*nu_cii/(1+self.z)*self.Dl ** 2
        return flux_map

    def get_sfr_map(self):
        flux_map = self.get_flux_map()
        sfr_map = SFR_lagache(flux_map, self.z)
        return sfr_map

    class alma_spec:
        def __init__(self, nu=None, vel=None, flux=None, err=None, ID=None, z=None):
            self.nu = nu
            self.vel = vel
            self.flux = flux
            self.err = err
            self.ID = ID
            self.z = z
            self.Dl = cosmo.luminosity_distance(self.z).to(u.Mpc).value
            self.Da = cosmo.angular_diameter_distance(self.z).to(u.kpc).value

            self.M0 = None
            self.dM0 = None
            self.M1 = None
            self.dM1 = None
            self.M2 = None
            self.dM2 = None
            self.L_CII = None
            self.L_CII_err = None
            self.rms = None
            self.snr = None
            self.resid = None

        def get_line_limits(self, factor=1.5):
            # fit an approximate Gaussian
            popt, pcov = curve_fit(gaussian, self.vel, self.flux,
                                   p0=[np.max(self.flux), 0, 100],
                                   bounds=[[np.max(self.flux) / 2, -100, 50],
                                           [np.max(self.flux), 100, 150]])
            # use to define limits for calculating moments
            i_min = find_nearest(self.vel, popt[1] - factor * 2.35 * popt[2])[0]
            i_max = find_nearest(self.vel, popt[1] + factor * 2.35 * popt[2])[0]
            return i_min, i_max

        def get_moments(self):
            i_min, i_max = self.get_line_limits(factor=1.5)

            M0, dM0, M1, dM1, M2, dM2 = moments(self.vel[i_min:i_max],
                                                self.flux[i_min:i_max],
                                                self.err[i_min:i_max])
            self.M0 = M0 / 1000
            self.dM0 = dM0 / 1000
            self.M1 = M1
            self.dM1 = dM1
            self.M2 = M2
            self.dM2 = dM2
            f = lambda I_CII: 1.04e-03 * I_CII * nu_cii / (1 + self.z) * self.Dl ** 2
            self.L_CII = f(self.M0)
            self.L_CII_err = mc_errors(f=f, params=[self.M0], errors=[self.dM0])

        def get_rms(self):
            i_min, i_max = self.get_line_limits(factor=1)
            # extract "continuum-only" fluxes
            cont = np.concatenate([self.flux[:i_min], self.flux[i_max:]])
            self.rms = np.nanstd(cont) * np.ones(len(self.vel))

        def get_snr(self):
            i_min, i_max = self.get_line_limits(factor=1)
            # extract "continuum-only" fluxes
            signal = np.nansum(self.flux[i_min:i_max])
            noise = np.nansum(self.err[i_min:i_max] ** 2) ** 0.5
            self.snr = signal / noise

        def fit_gauss(self):
            popt, pcov = curve_fit(gaussian, self.vel, self.flux,
                                   p0=[np.max(self.flux), 0, 100 / 2.35],
                                   bounds=[[np.max(self.flux) / 2, -100, 50 / 2.35],
                                           [np.max(self.flux), 100, 400 / 2.35]],
                                   sigma=self.rms
                                   )
            perr = [pcov[i][i] ** 0.5 for i in range(len(popt))]
            return popt, perr

        def fit_gauss_emcee(self, bounds, n_walkers=100, verbose=True):
            ndim = 3  # number of parameters to fit
            nwalkers = n_walkers  # number of walkers to search the parameter space

            bounds_lower, bounds_upper = bounds
            amp1_l, mean1_l, sig1_l = bounds_lower
            amp1_u, mean1_u, sig1_u = bounds_upper

            pos_min = np.array([amp1_l, mean1_l, sig1_l])
            pos_max = np.array([amp1_u, mean1_u, sig1_u])
            psize = pos_max - pos_min
            pos = [pos_min + psize * np.random.rand(ndim) for i in range(nwalkers)]

            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_single,
                                            args=(bounds, self.vel, self.flux, self.err))

            # burnin phase
            pos, prob, state = sampler.run_mcmc(pos, 50)
            sampler.reset()
            # perform MCMCprint
            pos, prob, state = sampler.run_mcmc(pos, 500)

            samples = sampler.flatchain
            # fig = corner.corner(samples, quantiles=[0.16, 0.50, 0.84])
            # fig.set_size_inches(10, 10)
            # # plt.savefig("tmp.pdf")
            # # plt.close()
            # plt.show()
            # samples.shape
            if verbose:
                print('Median acceptance fraction = ' + str(np.median(sampler.acceptance_fraction)))

            popt = np.percentile(samples, 50, axis=0)

            # begin bootstrap
            n_bootstrap = 50
            fits = np.zeros([n_bootstrap, ndim])
            spec = np.transpose(np.array((self.vel, self.flux, self.err)))
            for j in range(n_bootstrap):
                bootspec = bootstrap(spec, 1)[0]

                pos = [pos_min + psize * np.random.rand(ndim) for i in range(nwalkers)]
                sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_single,
                                                args=(bounds, bootspec[:, 0], bootspec[:, 1], bootspec[:, 2]))

                pos, prob, state = sampler.run_mcmc(pos, 50)
                sampler.reset()
                pos, prob, state = sampler.run_mcmc(pos, 500)
                samples = sampler.flatchain

                fits[j] = np.percentile(samples, 50, axis=0)

            perr = (np.percentile(fits, 84, axis=0) - np.percentile(fits, 16, axis=0)) / 2

            return popt, perr

        def fit_gauss_broad(self):
            popt_b, pcov_b = curve_fit(gaussian_double, self.vel, self.flux,
                                       p0=[np.max(self.flux), 0, 100 / 2.35, np.max(self.flux) / 5, 0, 500 / 2.35],
                                       bounds=[[0, -100, 80 / 2.35, 0, -100, 300 / 2.35],
                                               [np.inf, 100, 400 / 2.35, np.inf, 100, 1500 / 2.35]],
                                       sigma=self.rms)
            perr_b = [pcov_b[i][i] ** 0.5 for i in range(len(popt_b))]
            return popt_b, perr_b

        def fit_gauss_broad_emcee(self, bounds, n_walkers=200, verbose=True):
            ndim = 6  # number of parameters to fit
            nwalkers = n_walkers  # number of walkers to search the parameter space

            bounds_lower, bounds_upper = bounds
            amp1_l, mean1_l, sig1_l, amp2_l, mean2_l, sig2_l = bounds_lower
            amp1_u, mean1_u, sig1_u, amp2_u, mean2_u, sig2_u = bounds_upper

            pos_min = np.array([amp1_l, mean1_l, sig1_l, amp2_l, mean2_l, sig2_l])
            pos_max = np.array([amp1_u, mean1_u, sig1_u, amp2_u, mean2_u, sig2_u])
            psize = pos_max - pos_min
            pos = [pos_min + psize * np.random.rand(ndim) for i in range(nwalkers)]

            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_double,
                                            args=(bounds, self.vel, self.flux, self.err))

            # burnin phase
            pos, prob, state = sampler.run_mcmc(pos, 50)
            sampler.reset()
            # perform MCMCprint
            pos, prob, state = sampler.run_mcmc(pos, 500)

            samples = sampler.flatchain
            fig = corner.corner(samples, quantiles=[0.16, 0.50, 0.84])
            fig.set_size_inches(10, 10)
            # plt.savefig("tmp.pdf")
            plt.close()
            # plt.show()
            # samples.shape
            if verbose:
                print('Median acceptance fraction = ' + str(np.median(sampler.acceptance_fraction)))

            popt_b = np.percentile(samples, 50, axis=0)

            # begin bootstrap
            n_bootstrap = 50
            fits = np.zeros([n_bootstrap, ndim])
            spec = np.transpose(np.array((self.vel, self.flux, self.err)))
            for j in range(n_bootstrap):
                bootspec = bootstrap(spec, 1)[0]

                pos = [pos_min + psize * np.random.rand(ndim) for i in range(nwalkers)]
                sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_double,
                                                args=(bounds, bootspec[:, 0], bootspec[:, 1], bootspec[:, 2]))

                pos, prob, state = sampler.run_mcmc(pos, 50)
                sampler.reset()
                pos, prob, state = sampler.run_mcmc(pos, 500)
                samples = sampler.flatchain

                fits[j] = np.percentile(samples, 50, axis=0)

            perr_b = (np.percentile(fits, 84, axis=0) - np.percentile(fits, 16, axis=0)) / 2

            return popt_b, perr_b

        def get_residuals(self):
            popt, perr = self.fit_gauss()
            self.resid = self.flux - gaussian(self.vel, *popt)

# ------------------------------------------------------------------------------------------------------------

class alma_spec:
    def __init__(self, nu=None, vel=None, flux=None, err=None, ID=None, z=None):
        self.nu = nu
        self.vel = vel
        self.flux = flux
        self.err = err
        self.ID = ID
        self.z = z
        self.Dl = cosmo.luminosity_distance(self.z).to(u.Mpc).value
        self.Da = cosmo.angular_diameter_distance(self.z).to(u.kpc).value

        self.M0 = None
        self.dM0 = None
        self.M1 = None
        self.dM1 = None
        self.M2 = None
        self.dM2 = None
        self.L_CII = None
        self.L_CII_err = None
        self.rms = None
        self.snr = None
        self.resid = None

    def get_line_limits(self, factor=1.5):
        # fit an approximate Gaussian
        popt, pcov = curve_fit(gaussian, self.vel, self.flux,
                               p0=[np.max(self.flux), 0, 100],
                               bounds=[[np.max(self.flux) / 2, -100, 50],
                                       [np.max(self.flux), 100, 150]])
        # use to define limits for calculating moments
        i_min = find_nearest(self.vel, popt[1] - factor * 2.35 * popt[2])[0]
        i_max = find_nearest(self.vel, popt[1] + factor * 2.35 * popt[2])[0]
        return i_min, i_max

    def get_moments(self):
        i_min, i_max = self.get_line_limits(factor=1.5)

        M0, dM0, M1, dM1, M2, dM2 = moments(self.vel[i_min:i_max],
                                            self.flux[i_min:i_max],
                                            self.err[i_min:i_max])
        self.M0 = M0/1000
        self.dM0 = dM0/1000
        self.M1 = M1
        self.dM1 = dM1
        self.M2 = M2
        self.dM2 = dM2
        f = lambda I_CII : 1.04e-03*I_CII*nu_cii/(1+self.z)*self.Dl**2
        self.L_CII = f(self.M0)
        self.L_CII_err = mc_errors(f=f, params=[self.M0], errors=[self.dM0])

    def get_rms(self):
        i_min, i_max = self.get_line_limits(factor=1)
        # extract "continuum-only" fluxes
        cont = np.concatenate([self.flux[:i_min],self.flux[i_max:]])
        self.rms = np.nanstd(cont)*np.ones(len(self.vel))

    def get_snr(self):
        i_min, i_max = self.get_line_limits(factor=1)
        # extract "continuum-only" fluxes
        signal = np.nansum(self.flux[i_min:i_max])
        noise = np.nansum(self.err[i_min:i_max]**2)**0.5
        self.snr = signal/noise

    def fit_gauss(self):
        popt, pcov = curve_fit(gaussian, self.vel, self.flux,
                               p0=[np.max(self.flux), 0, 100/2.35],
                               bounds=[[np.max(self.flux) / 2, -100, 50/2.35],
                                       [np.max(self.flux), 100, 400/2.35]],
                               sigma=self.rms
                               )
        perr = [pcov[i][i]**0.5 for i in range(len(popt))]
        return popt, perr

    def fit_gauss_emcee(self, bounds, n_walkers=100, verbose=True):
        ndim = 3                # number of parameters to fit
        nwalkers = n_walkers           # number of walkers to search the parameter space

        bounds_lower, bounds_upper = bounds
        amp1_l, mean1_l, sig1_l = bounds_lower
        amp1_u, mean1_u, sig1_u = bounds_upper

        pos_min = np.array([amp1_l, mean1_l, sig1_l])
        pos_max = np.array([amp1_u, mean1_u, sig1_u])
        psize = pos_max - pos_min
        pos = [pos_min + psize * np.random.rand(ndim) for i in range(nwalkers)]

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_single,
                                        args=(bounds, self.vel, self.flux, self.err))

        # burnin phase
        pos, prob, state = sampler.run_mcmc(pos, 50)
        sampler.reset()
        # perform MCMCprint
        pos, prob, state = sampler.run_mcmc(pos, 500)

        samples = sampler.flatchain
        # fig = corner.corner(samples, quantiles=[0.16, 0.50, 0.84])
        # fig.set_size_inches(10, 10)
        # # plt.savefig("tmp.pdf")
        # # plt.close()
        # plt.show()
        # samples.shape
        if verbose:
            print('Median acceptance fraction = ' + str(np.median(sampler.acceptance_fraction)))

        popt = np.percentile(samples, 50, axis=0)

        # begin bootstrap
        n_bootstrap = 50
        fits = np.zeros([n_bootstrap, ndim])
        spec = np.transpose(np.array((self.vel, self.flux, self.err)))
        for j in range(n_bootstrap):
            bootspec = bootstrap(spec, 1)[0]

            pos = [pos_min + psize * np.random.rand(ndim) for i in range(nwalkers)]
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_single,
                                            args=(bounds, bootspec[:, 0], bootspec[:, 1], bootspec[:, 2]))

            pos, prob, state = sampler.run_mcmc(pos, 50)
            sampler.reset()
            pos, prob, state = sampler.run_mcmc(pos, 500)
            samples = sampler.flatchain

            fits[j] = np.percentile(samples, 50, axis=0)

        perr = (np.percentile(fits, 84, axis=0) - np.percentile(fits, 16, axis=0)) / 2

        return popt, perr

    def fit_gauss_broad(self):
        popt_b, pcov_b = curve_fit(gaussian_double, self.vel, self.flux,
                                   p0=[np.max(self.flux), 0, 100/2.35, np.max(self.flux)/5, 0, 500/2.35],
                                   bounds=[[0, -100, 80/2.35, 0, -100, 300/2.35],
                                           [np.inf, 100, 400/2.35, np.inf, 100, 1500/2.35]],
                                   sigma=self.rms)
        perr_b = [pcov_b[i][i]**0.5 for i in range(len(popt_b))]
        return popt_b, perr_b

    def fit_gauss_broad_emcee(self, bounds, n_walkers=200, verbose=True):
        ndim = 6  # number of parameters to fit
        nwalkers = n_walkers  # number of walkers to search the parameter space

        bounds_lower, bounds_upper = bounds
        amp1_l, mean1_l, sig1_l, amp2_l, mean2_l, sig2_l = bounds_lower
        amp1_u, mean1_u, sig1_u, amp2_u, mean2_u, sig2_u = bounds_upper

        pos_min = np.array([amp1_l, mean1_l, sig1_l, amp2_l, mean2_l, sig2_l])
        pos_max = np.array([amp1_u, mean1_u, sig1_u, amp2_u, mean2_u, sig2_u])
        psize = pos_max - pos_min
        pos = [pos_min + psize * np.random.rand(ndim) for i in range(nwalkers)]

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_double,
                                        args=(bounds, self.vel, self.flux, self.err))

        # burnin phase
        pos, prob, state = sampler.run_mcmc(pos, 50)
        sampler.reset()
        # perform MCMCprint
        pos, prob, state = sampler.run_mcmc(pos, 500)

        samples = sampler.flatchain
        fig = corner.corner(samples, quantiles=[0.16, 0.50, 0.84])
        fig.set_size_inches(10, 10)
        # plt.savefig("tmp.pdf")
        plt.close()
        # plt.show()
        # samples.shape
        if verbose:
            print('Median acceptance fraction = ' + str(np.median(sampler.acceptance_fraction)))

        popt_b = np.percentile(samples, 50, axis=0)

        # begin bootstrap
        n_bootstrap = 50
        fits = np.zeros([n_bootstrap, ndim])
        spec = np.transpose(np.array((self.vel, self.flux, self.err)))
        for j in range(n_bootstrap):
            bootspec = bootstrap(spec, 1)[0]

            pos = [pos_min + psize * np.random.rand(ndim) for i in range(nwalkers)]
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_double,
                                            args=(bounds, bootspec[:, 0], bootspec[:, 1], bootspec[:, 2]))

            pos, prob, state = sampler.run_mcmc(pos, 50)
            sampler.reset()
            pos, prob, state = sampler.run_mcmc(pos, 500)
            samples = sampler.flatchain

            fits[j] = np.percentile(samples, 50, axis=0)

        perr_b = (np.percentile(fits, 84, axis=0) - np.percentile(fits, 16, axis=0)) / 2

        return popt_b, perr_b

    def get_residuals(self):
        popt, perr = self.fit_gauss()
        self.resid = self.flux - gaussian(self.vel, *popt)

# ------------------------------------------------------------------------------------------------------------

def make_alma_cube(file, ID, z, ra, dec, fwhm_cii):
    """generate a CRISTAL cube object from a given file"""
    hdul = fits.open(file)
    cube, header = hdul[0].data[0 ,: ,: ,:], hdul[0].header
    if len(hdul)==2:
        beams = hdul[1].data
    else:
        print("No beam information available.")
        beams = Table(data=np.zeros([cube.shape[0], 5]), names=["BMAJ", "BMIN", "BPA", "CHAN", "POL"])
        beams["BMAJ"] = 0.9
        beams["BMIN"] = 0.9

    wcs = WCS.dropaxis(WCS(header), dropax=3)       # get WCS from header
    # wcs = WCS.dropaxis(wcs, dropax=2)               # drop frequency axis
    pix_scale = np.abs(header["CDELT1"]) * 3600     # get pixel scale in arcsec/pixel

    nu = np.linspace(header["CRVAL3"], header["CRVAL3"]+(header["NAXIS3"]-1)*header["CDELT3"],
                     header["NAXIS3"]) / 1e09                       # frequency axis in GHz
    vel_axis = ((nu_cii/(1+z))-nu)/(nu_cii/(1+z))*3e05    # velocity axis in km/s

    Cube = alma_cube(nu=nu, vel_axis=vel_axis, cube=cube, wcs=wcs, header=header, ID=ID,
                        pix_scale=pix_scale, z=z, beams=beams, ra=ra, dec=dec, fwhm_cii=fwhm_cii)

    return Cube

def make_alma_cube_cutout(Cube, size, centre):
    """size in arcsec"""
    cy, cx = centre.to_pixel(wcs=Cube.wcs)

    # slice region in the middle
    size /= Cube.pix_scale
    wcs_cut = Cube.wcs[:,int(cx-size/2):int(cx+size/2), int(cy-size/2):int(cy+size/2)] # sliced here
    cube_cut = Cube.cube[:,int(cx-size/2):int(cx+size/2), int(cy-size/2):int(cy+size/2)]

    Cube_cutout = alma_cube(nu=Cube.nu, vel_axis=Cube.vel_axis,
                               cube=cube_cut, wcs=wcs_cut, ID=Cube.ID,
                                pix_scale=Cube.pix_scale, z=Cube.z, beams=Cube.beams,
                               ra=Cube.ra, dec=Cube.dec, fwhm_cii=Cube.fwhm_cii)

    return Cube_cutout

# ------------------------------------------------------------------------------------------------------------