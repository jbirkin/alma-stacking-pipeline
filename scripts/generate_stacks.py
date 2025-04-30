import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob
from scipy.optimize import curve_fit
from scipy.integrate import simpson
from scipy import stats
import sys
import pickle
import os
import time
import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")

from astro_cubes.tools import disable_ax_ticks, find_nearest
from astro_cubes.defaults import colors
from src.utils.tables import cristal_dir, ctab, cube_dir, nu_cii, cosmo
from src.utils.stacking import binned_stack_IDs, bic, bic_check, cristal_stack
from src.utils.line_models import gaussian, gaussian_double, line_model_single, line_model_double
from src.utils.cristal_cube import cristal_cube, make_cristal_cube, make_cristal_cube_cutout, cristal_spec
from src.utils.formulae import m_atom_out, m_out_dot

# --------------------------------------------------------------------------------------------------
# Establish the sample and its median properties

# Error message if a non-existent directory is passed
if len(sys.argv)==1:
    print("Enter the directory containing the spectra you wish to use. Available choices are:")
    print(np.array(os.listdir(cristal_dir+"spectra/")))
    exit()

# Set directory containing the spectra, and the outfile to be written
print("Generating stacks...")
spec_dir = str(sys.argv[1])+"/"
if len(sys.argv)==2:
    outfile = str(sys.argv[1])
else:
    outfile = str(sys.argv[2])
print("Directory: {}".format(spec_dir))
filelist = glob.glob(cristal_dir+"spectra/"+spec_dir+"*.npy")

# Determine list of files to be used
ID_list = sorted(set([file.split("/")[-1].split(".")[0].replace("_centered","") for file in filelist]))
# ID_list = list(ctab["CRISTAL_ID_full"])#[(ctab["kinematic_class"]=="Disk I") | (ctab["kinematic_class"]=="Disk II")]

# Manually set sources to be excluded
# bad_list = ["CRISTAL-01a","CRISTAL-04a","CRISTAL-04b","CRISTAL-06a","CRISTAL-06b","CRISTAL-07a","CRISTAL-07b",
#             "CRISTAL-05", "CRISTAL-17","CRISTAL-22a","CRISTAL-22b","CRISTAL-23a","CRISTAL-23b"]
# bad_list = list(ctab["CRISTAL_ID_full"][(ctab["kinematic_class"]=="Non-disk") | (ctab["kinematic_class"]=="???")])
bad_list = list(ctab["CRISTAL_ID_full"][ctab["in_stack?"]==False])

if spec_dir == "g20_sim_add_rms_vel_res_med/":
    bad_list.append("CRISTAL-08")
    bad_list.append("CRISTAL-23b")
if "_noc2" in outfile:
    bad_list.append("CRISTAL-02")

# bad_list = list(ctab["CRISTAL_ID_full"][(ctab["kinematic_class"]=="???")])

# Remove these from the file list
for item in bad_list:
    if item in ID_list:
        ID_list.remove(item)

if spec_dir == "g20_sim_add_rms_vel_res_med/":
    N_gal = 15
else:
    N_gal = len(ID_list)

# Get subset of CRISTAL master table which contains the correct sources
ii = []
unused = []
for i in range(len(ID_list)):
    if ID_list[i] in ctab["CRISTAL_ID_full"]:
        i_gal = np.where(ID_list[i]==ctab["CRISTAL_ID_full"])[0][0]
        if ctab["Detect?"][i_gal]:
            ii.append(i_gal)
        else:
            unused.append(ID_list[i])
    else:
        unused.append(ID_list[i])
for i in range(len(unused)):
    ID_list.remove(unused[i])
ctab_sub = ctab[ii]
ctab_sub = ctab_sub[ctab_sub["Detect?"]]

# Get median redshift of the subsample
z_med = np.median(ctab_sub["z"])
# Get median SFR of the subsample, ignoring those with no measurements
sfr_med = 10**np.median(ctab_sub["logSFR"][ctab_sub["logSFR"]>0])
sfr16 = np.nanpercentile(10 ** ctab_sub["logSFR"][ctab_sub["logSFR"]>0], 16)
sfr84 = np.nanpercentile(10 ** ctab_sub["logSFR"][ctab_sub["logSFR"]>0], 84)
# Get median stellar mass of the subsample, ignoring those with no measurements
mstar_med = 10**np.median(ctab_sub["logMstar"][ctab_sub["logMstar"]>0])
mstar16 = np.nanpercentile(10 ** ctab_sub["logMstar"][ctab_sub["logMstar"]>0], 16)
mstar84 = np.nanpercentile(10 ** ctab_sub["logMstar"][ctab_sub["logMstar"]>0], 84)

# --------------------------------------------------------------------------------------------------
# Do the stacking

# Set type of normalization to use, and velocity resolution
if int(sys.argv[3])==0:
    fwhm_norm = False
else:
    fwhm_norm = True
flux_norm = str(sys.argv[4])
vel_res_stack = float(sys.argv[5])

# Combine spectra
stack_vel, stack_flux, stack_err, stack_bins, vels, fluxes, errs, norm_values = binned_stack_IDs(
                IDs=ID_list, spec_dir=spec_dir, vel_res_stack = vel_res_stack, fwhm_norm = fwhm_norm,
                flux_norm = flux_norm, use_rms=True)

# Get the median value used to normalize fluxes in stacking
if fwhm_norm and flux_norm!="none":
    norm_low = np.percentile(norm_values, 16)
    norm_med = np.percentile(norm_values, 50)
    norm_high = np.percentile(norm_values, 84)
else:
    norm_med = 1
# Reapply this value
stack_flux*=norm_med
stack_err*=norm_med

# Trim arrays for BIC calculation
i_min = find_nearest(stack_vel, -750)[0]
i_max = find_nearest(stack_vel, 750)[0]
stack_vel = stack_vel[i_min:i_max+1]
stack_flux = stack_flux[i_min:i_max+1]
stack_err = stack_err[i_min:i_max+1]
stack_bins = stack_bins[i_min:i_max+1]

Spec_stack = cristal_spec(vel=stack_vel, flux=stack_flux, err=stack_err, z=z_med) # generate spec class for stack

popts, perrs = [], []
popt_bs, perr_bs = [], []
delta_bics = []

N_iter = 1

for n in range(N_iter):
    print(f"Fitting iteration number {n+1}")
    # Fit single-Gaussian model to the composite, using emcee
    popt, perr = Spec_stack.fit_gauss_emcee(bounds=[[np.max(Spec_stack.flux)/2, -100, 50/2.35],
                                                    [np.max(Spec_stack.flux)*1.5, 100, 400/2.35]],
                                            n_walkers=20
                                            )
    # Fit double-Gaussian model to the composite, using emcee
    popt_b, perr_b = Spec_stack.fit_gauss_broad_emcee(
        bounds=[[np.max(Spec_stack.flux)*0.5, -200, 80/2.35, 0, -200, 400/2.35],
        [np.max(Spec_stack.flux)*1.5, 200, 400/2.35, np.max(Spec_stack.flux)*0.5, 200, 1000/2.35]],
        n_walkers=30#200
    )
    popts.append(popt)
    perrs.append(perr)
    popt_bs.append(popt_b)
    perr_bs.append(perr_b)
    bic_single = bic(data=stack_flux, model=gaussian(stack_vel, *popt), error=stack_err, n_param=3)
    bic_double = bic(data=stack_flux, model=gaussian_double(stack_vel, *popt_b), error=stack_err, n_param=6)
    print(bic_single, bic_double, popt_b[2], popt_b[5])
    delta_bics.append(bic_single - bic_double)

popts, perrs = np.array(popts), np.array(perrs)
popt_bs, perr_bs = np.array(popt_bs), np.array(perr_bs)
delta_bics = np.array(delta_bics)

np.save(f"{cristal_dir}stacks/{outfile}_single_fit_params.npy", [popts, perrs])
np.save(f"{cristal_dir}stacks/{outfile}_double_fit_params.npy", [popt_bs, perr_bs])
np.save(f"{cristal_dir}stacks/{outfile}_bics.npy", delta_bics)

popt = np.median(popts, axis=0)
perr = np.median(perrs, axis=0)
popt_b = np.median(popt_bs, axis=0)
perr_b = np.median(perr_bs, axis=0)

# Measure the BIC for both models, then work out which is best
bic_single = bic(data=stack_flux, model=gaussian(stack_vel, *popt), error=stack_err, n_param=3)
bic_double = bic(data=stack_flux, model=gaussian_double(stack_vel, *popt_b), error=stack_err, n_param=6)
bic_result = bic_check(bic_single, bic_double)

# Make a cristal_stack object
Stack = cristal_stack(spec_dir=spec_dir, N_gal=N_gal, ID_list=ID_list, bad_list=bad_list, table=ctab_sub, fwhm_norm=fwhm_norm,
                      flux_norm=flux_norm, stack_vel=stack_vel, stack_flux=stack_flux, stack_err=stack_err,
                      stack_bins=stack_bins, vels=vels, fluxes=fluxes, errs=errs, norm_med=norm_med, popt=popt,
                      perr=perr, popt_b=popt_b, perr_b=perr_b, bic_single=bic_single, bic_double=bic_double,
                      bic_result=bic_result, outfile=outfile)
# Save the cristal_spec to a file
with open(f'{outfile}.pkl', 'wb') as file:
    pickle.dump(Stack, file)