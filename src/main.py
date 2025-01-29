from alma_stacking_pipeline.src.data.masking import make_source_masks, make_sfr_masks
from alma_stacking_pipeline.src.data.spectral_extraction import extract_spectra, center_spectra
from alma_stacking_pipeline.src.stacking import binned_stack_IDs
from alma_stacking_pipeline.src.config import cristal_tab, work_dir


# Make general source masks
make_source_masks(tab=cristal_tab)

# Make SFR masks
make_sfr_masks(tab=cristal_tab, SFR_threshold=1.7)

# Extract 1-D spectra
extract_spectra(tab=cristal_tab, spec_method="contours", mask_dir=f"{work_dir}")

# Center spectra
center_spectra(spec_dir=f"{work_dir}/data/spectra/contours/")

# # Combine spectra
ID_list = ["CRISTAL-02", "CRISTAL-03"]
stack_vel, stack_flux, stack_err, stack_bins, vels, fluxes, errs, norm_values = binned_stack_IDs(
                IDs=ID_list, tab=cristal_tab, spec_dir=f"{work_dir}/data/spectra/contours/", vel_res_stack = 50,
                fwhm_norm = True, flux_norm = True, use_rms=True)