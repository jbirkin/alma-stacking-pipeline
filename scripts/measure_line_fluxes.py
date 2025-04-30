import numpy as np
import glob

from src.utils.tables import ctab, ctab_file, nu_cii, cristal_dir, cube_dir
from src.utils.cristal_cube import cristal_spec, make_cristal_cube, cristal_cube
from src.utils.formulae import SFR_lagache

spec_dir = "spectra/contours_2sig_vel_res_med/"
filelist = glob.glob(cristal_dir+spec_dir+"*.npy")

for i in range(len(ctab)):
    if not ctab["Detect?"][i]:
        continue
    ID = ctab["CRISTAL_ID_full"][i]
    if cristal_dir+spec_dir+ID+".npy" not in filelist:
        continue
    z, ra, dec = ctab["z"][i], ctab["RA"][i], ctab["Dec"][i]
    fwhm_cii = ctab["FWHM_CII"][i]
    file = cube_dir + ctab["File"][i]

    Cube = make_cristal_cube(file, ID, z, ra, dec, fwhm_cii)
    vel, nu, flux, err = np.load(cristal_dir + spec_dir + ID + ".npy")
    Spec = cristal_spec(vel=vel, flux=flux, err=err, z=z)

    Spec.get_moments()
    Spec.get_rms()
    Spec.get_snr()

    bmaj, bmin, bpa = Cube.beams[100][0:3]
    pix_per_beam = np.pi * bmaj * bmin / (4 * np.log(2)) / Cube.pix_scale ** 2

    ctab["M0"][i], ctab["M0_err"][i] = Spec.M0, Spec.dM0
    ctab["M1"][i], ctab["M1_err"][i] = Spec.M1, Spec.dM1
    ctab["M2"][i], ctab["M2_err"][i] = Spec.M2, Spec.dM2
    ctab["L_CII"][i], ctab["L_CII_err"][i] = Spec.L_CII, Spec.L_CII_err
    ctab["SFR_CII"][i] = SFR_lagache(Spec.L_CII, Spec.z)
    ctab["S/N"][i] = Spec.snr

    # SNR = Spec.M0*1000/(Spec.vel[1]-Spec.vel[0])/np.median(Spec.rms)
    # print(SNR)
    # print(ID, np.median(Spec.rms), Spec.M0*1000/(Spec.vel[1]-Spec.vel[0]))

ctab["M1", "M1_err"].pprint()

ctab.write(ctab_file, overwrite=True)