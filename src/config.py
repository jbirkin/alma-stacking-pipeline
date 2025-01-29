from astropy.table import Table
from astropy.cosmology import Planck18

work_dir = "../"
cube_dir = work_dir+"data/cubes/"

cristal_tab_file = work_dir+"data/cristal_table.fits"         # file in which main table is stored
cristal_tab = Table.read(cristal_tab_file)                    # load the main table as an astropy Table

cosmo = Planck18

nu_cii = 1900.53690000              # frequency of [CII] in GHz