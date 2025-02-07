import yaml
from pathlib import Path
from astropy.table import Table
from astropy.cosmology import Planck18

# Load YAML configuration
with open(Path(__file__).parent / "config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Define paths
work_dir = str(Path(config["work_dir"]).resolve()) + "/"
cristal_tab_file = f"{work_dir}data/{config['cristal_tab_file']}"
cristal_tab = Table.read(cristal_tab_file)

# Cosmology model
if config["cosmology"] == "Planck18":
    cosmo = Planck18

nu_cii = config["nu_cii"]