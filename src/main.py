import yaml
import argparse
from pathlib import Path

from alma_stacking_pipeline.src.config_loader import work_dir, cristal_tab
from alma_stacking_pipeline.src.data.masking import make_source_masks, make_sfr_masks
from alma_stacking_pipeline.src.data.spectral_extraction import (
    extract_spectra, center_spectra
)
from alma_stacking_pipeline.src.stacking import binned_stack_IDs
from alma_stacking_pipeline.src.plotting import plot_stack

print(f"Working directory: {work_dir}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run the ALMA stacking pipeline.")
    parser.add_argument("--SFR_threshold", type=float, default=1.7, help="Threshold for SFR masking")
    parser.add_argument("--vel_res_stack", type=int, default=50, help="Velocity resolution for stacking")

    args = parser.parse_args()

    # Make general source masks
    make_source_masks(tab=cristal_tab)

    # Make SFR masks
    make_sfr_masks(tab=cristal_tab, SFR_threshold=args.SFR_threshold)

    # Extract 1-D spectra
    extract_spectra(tab=cristal_tab, spec_method="contours", mask_dir=f"{work_dir}")

    # Center spectra
    center_spectra(spec_dir=f"{work_dir}data/spectra/contours/")

    # Combine spectra
    ID_list = cristal_tab["CRISTAL_ID_full"]
    binned_stack_IDs(
        IDs=ID_list, tab=cristal_tab, spec_dir=f"data/spectra/contours/", vel_res_stack =
        args.vel_res_stack, fwhm_norm = True, flux_norm = "unity", use_rms=True)
    plot_stack("stack.pkl")

if __name__ == "__main__":
    main()