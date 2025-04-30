import numpy as np
from scipy.optimize import curve_fit
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")
warnings.filterwarnings("ignore", message="invalid value encountered in scalar power")
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice.")

from astro_cubes.tools import find_nearest, moments
from src.utils.tables import ctab, cristal_dir
from src.utils.line_models import gaussian

# --------------------------------------------------------------------------------------------------

print("Centering spectra by calculating the first moment...")
spec_dir = cristal_dir+"spectra/"+str(sys.argv[1])+"/"
print("Directory: {}".format(spec_dir))
filelist = glob.glob(spec_dir+"*.npy")
filelist = sorted([file for file in filelist if not "centered" in file])

in_stack = ctab["CRISTAL_ID_full"][ctab["in_stack?"]]

for i in tqdm(range(len(filelist))):
    ID = filelist[i].split("/")[-1].replace(".npy","")
    if ID not in in_stack:
        continue
    vel, nu, flux, err = np.load(filelist[i])

    # fit an approximate Gaussian
    popt, pcov = curve_fit(gaussian, vel, flux,
                           p0=[np.max(flux), 0, 100],
                           bounds=[[np.max(flux)/2, -400, 50],
                                   [np.max(flux), 400, 150]])
    # use to define limits for calculating moments
    i_min = find_nearest(vel, popt[1] - 1.5 * 2.35 * popt[2])[0]
    i_max = find_nearest(vel, popt[1] + 1.5 * 2.35 * popt[2])[0]

    M0, dM0, M1, dM1, M2, dM2 = moments(vel[i_min:i_max],
                                        flux[i_min:i_max],
                                        err[i_min:i_max])

    tqdm.write(f'{ID} flux: {M0:.1f} +/- {dM0:.1f}')
    tqdm.write(f'{ID} centroid: {M1:.1f} +/- {dM1:.1f}')
    tqdm.write(f'{ID} linewidth: {M2:.1f} +/- {dM2:.1f}')

    # np.save(filelist[i].replace(".npy","_centered.npy"),
    #         np.array([vel - M1, nu, flux, err]))

print("Done.")

# --------------------------------------------------------------------------------------------------